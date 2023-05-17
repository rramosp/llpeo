import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np
from tensorflow.keras.layers import Conv2D, Resizing, InputLayer, Flatten, Dense
from tensorflow.keras.models import Sequential


# Patch extraction as a layer 
# from https://keras.io/examples/vision/mlp_image_classification/
class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size, image_size, strides):
        super(Patches, self).__init__()
        self.patch_size = patch_size
        self.strides = strides
        self.num_patches = (image_size - patch_size) // strides + 1 

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.strides, self.strides, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, self.num_patches ** 2, patch_dims])
        return patches


def dm2comp(dm):
    '''
    Extract vectors and weights from a factorized density matrix representation
    Arguments:
     dm: tensor of shape (bs, n, d + 1)
    Returns:
     w: tensor of shape (bs, n)
     v: tensor of shape (bs, n, d)
    '''
    return dm[:, :, 0], dm[:, :, 1:]


def comp2dm(w, v):
    '''
    Construct a factorized density matrix from vectors and weights
    Arguments:
     w: tensor of shape (bs, n)
     v: tensor of shape (bs, n, d)
    Returns:
     dm: tensor of shape (bs, n, d + 1)
    '''
    return tf.concat((w[:, :, tf.newaxis], v), axis=2)


def pure2dm(psi):
    '''
    Construct a factorized density matrix to represent a pure state
    Arguments:
     psi: tensor of shape (bs, d)
    Returns:
     dm: tensor of shape (bs, 1, d + 1)
    '''
    ones = tf.ones_like(psi[:, 0:1])
    dm = tf.concat((ones[:,tf.newaxis, :],
                    psi[:,tf.newaxis, :]),
                   axis=2)
    return dm


def dm2distrib(dm, sigma):
    '''
    Creates a Gaussian mixture distribution from the components of a density
    matrix with an RBF kernel 
    Arguments:
     dm: tensor of shape (bs, n, d + 1)
     sigma: sigma parameter of the RBF kernel 
    Returns:
     gm: mixture of Gaussian distribution with shape (bs, )
    '''
    w, v = dm2comp(dm)
    gm = tfd.MixtureSameFamily(reparameterize=True,
            mixture_distribution=tfd.Categorical(
                                    probs=w),
            components_distribution=tfd.Independent( tfd.Normal(
                    loc=v,  # component 2
                    scale=sigma * np.sqrt(2.)),
                    reinterpreted_batch_ndims=1))
    return gm


def pure_dm_overlap(x, dm, kernel):
    '''
    Calculates the overlap of a state  \phi(x) with a density 
    matrix in a RKHS defined by a kernel
    Arguments:
      x: tensor of shape (bs, d)
     dm: tensor of shape (bs, n, d + 1)
     kernel: kernel function 
              k: (bs, d) x (bs, n, d) -> (bs, n)
    Returns:
     overlap: tensor with shape (bs, )
    '''
    w, v = dm2comp(dm)
    overlap = tf.einsum('...i,...i->...', w, kernel(x, v) ** 2)
    return overlap

## Kernels

def create_comp_trans_kernel(transform, kernel):
    '''
    Composes a transformation and a kernel to create a new
    kernel.
    Arguments:
        transform: a function f that transform the input before feeding it to the 
                   kernel
                   f:(bs, d) -> (bs, D) 
        kernel: a kernel function
                k:(bs, n, D)x(m, D) -> (bs, n, m)
    Returns:
        a function that receives 2 tensors with the following shapes
            Input:
                A: tensor of shape (bs, n, d)
                B: tensor of shape (m, d)
            Result:
                K: tensor of shape (bs, n, m)
    '''
    def comp_kernel(A, B):
        shape = tf.shape(A) # (bs, n, d)
        A = tf.reshape(A, [shape[0] * shape[1], shape[2]])
        A = transform(A)
        dim_out = tf.shape(A)[1]
        A = tf.reshape(A, [shape[0], shape[1], dim_out])
        B = transform(B)
        return kernel(A, B)
    return comp_kernel

def create_rbf_kernel(sigma):
    '''
    Builds a function that calculates the rbf kernel between two set of vectors
    Arguments:
        sigma: RBF scale parameter
    Returns:
        a function that receives 2 tensors with the following shapes
        Input:
            A: tensor of shape (bs, n, d)
            B: tensor of shape (m, d)
        Result:
            K: tensor of shape (bs, n, m)
    '''

    def rbf_kernel(A, B):
        shape_A = tf.shape(A)
        shape_B = tf.shape(B)
        A_norm = tf.norm(A, axis=-1)[..., tf.newaxis] ** 2
        B_norm = tf.norm(B, axis=-1)[tf.newaxis, tf.newaxis, :] ** 2
        A_reshaped = tf.reshape(A, [-1, shape_A[2]])
        AB = tf.matmul(A_reshaped, B, transpose_b=True) 
        AB = tf.reshape(AB, [shape_A[0], shape_A[1], shape_B[0]])
        dist2 = A_norm + B_norm - 2. * AB
        dist2 = tf.clip_by_value(dist2, 0., np.inf)
        K = tf.exp(-dist2 / (2 * sigma ** 2))
        return K
    return rbf_kernel


## Layers and models

def l1_loss(vals):
    '''
    Calculate the l1 loss for a batch of vectors
    Arguments:
        vals: tensor with shape (b_size, n)
    '''
    b_size = tf.cast(tf.shape(vals)[0], dtype=tf.float32)
    vals = vals / tf.norm(vals, axis=1)[:, tf.newaxis]
    loss = tf.reduce_sum(tf.abs(vals)) / b_size
    return loss

class KQMUnit(tf.keras.layers.Layer):
    """Kernel Quantum Measurement Unit
    Receives as input a factored density matrix represented by a set of vectors
    and weight values. 
    Returns a resulting factored density matrix.
    Input shape:
        (batch_size, n_comp_in, dim_x + 1)
        where dim_x is the dimension of the input state
        and n_comp_in is the number of components of the input factorization. 
        The weights of the input factorization of sample i are [i, :, 0], 
        and the vectors are [i, :, 1:dim_x + 1].
    Output shape:
        (batch_size, n_comp, dim_y)
        where dim_y is the dimension of the output state
        and n_comp is the number of components used to represent the train
        density matrix. The weights of the
        output factorization for sample i are [i, :, 0], and the vectors
        are [i, :, 1:dim_y + 1].
    Arguments:
        dim_x: int. the dimension of the input state
        dim_y: int. the dimension of the output state
        x_train: bool. Whether to train or not the x compoments of the train
                       density matrix.
        x_train: bool. Whether to train or not the y compoments of the train
                       density matrix.
        w_train: bool. Whether to train or not the weights of the compoments 
                       of the train density matrix. 
        n_comp: int. Number of components used to represent 
                 the train density matrix
        l1_act: float. Coefficient of the regularization term penalizing the l1
                       norm of the activations.
        l1_x: float. Coefficient of the regularization term penalizing the l1
                       norm of the x components.
        l1_y: float. Coefficient of the regularization term penalizing the l1
                       norm of the y components.
    """
    def __init__(
            self,
            kernel,
            dim_x: int,
            dim_y: int,
            x_train: bool = True,
            y_train: bool = True,
            w_train: bool = True,
            n_comp: int = 0, 
            l1_x: float = 0.,
            l1_y: float = 0.,
            l1_act: float = 0.,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.kernel = kernel
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.x_train = x_train
        self.y_train = y_train
        self.w_train = w_train
        self.n_comp = n_comp
        self.l1_x = l1_x
        self.l1_y = l1_y
        self.l1_act = l1_act

    def build(self, input_shape):
        if (input_shape[1] and input_shape[2] != self.dim_x + 1 
            or len(input_shape) != 3):
            raise ValueError(
                f'Input dimension must be (batch_size, m, {self.dim_x + 1} )'
                f' but it is {input_shape}'
                )
        self.c_x = self.add_weight(
            "c_x",
            shape=(self.n_comp, self.dim_x),
            #initializer=tf.keras.initializers.orthogonal(),
            initializer=tf.keras.initializers.random_normal(),
            trainable=self.x_train)
        self.c_y = self.add_weight(
            "c_y",
            shape=(self.n_comp, self.dim_y),
            initializer=tf.keras.initializers.Constant(0.05),
            #initializer=tf.keras.initializers.random_normal(),
            trainable=self.y_train)
        self.comp_w = self.add_weight(
            "comp_w",
            shape=(self.n_comp,),
            initializer=tf.keras.initializers.constant(1./self.n_comp),
            trainable=self.w_train) 
        self.eps = 1e-10
        self.built = True

    def call(self, inputs):
        
        # Weight regularizers
        if self.l1_x != 0:
            self.add_loss(self.l1_x * l1_loss(self.c_x))
        if self.l1_y != 0:
            self.add_loss(self.l1_y * l1_loss(self.c_y))
        comp_w = tf.nn.softmax(self.comp_w)
        in_w = inputs[:, :, 0]  # shape (b, n_comp_in)
        in_v = inputs[:, :, 1:] # shape (b, n_comp_in, dim_x)
        out_vw = self.kernel(in_v, self.c_x)  # shape (b, n_comp_in, n_comp)
        out_w = (tf.expand_dims(tf.expand_dims(comp_w, axis=0), axis=0) *
                 tf.square(out_vw)) # shape (b, n_comp_in, n_comp)
        out_w = tf.maximum(out_w, self.eps) #########
        # out_w_sum = tf.maximum(tf.reduce_sum(out_w, axis=2), self.eps)  # shape (b, n_comp_in)
        out_w_sum = tf.reduce_sum(out_w, axis=2) # shape (b, n_comp_in)
        out_w = out_w / tf.expand_dims(out_w_sum, axis=2)
        out_w = tf.einsum('...i,...ij->...j', in_w, out_w, optimize="optimal")
                # shape (b, n_comp)
        if self.l1_act != 0:
            self.add_loss(self.l1_act * l1_loss(out_w))
        out_w = tf.expand_dims(out_w, axis=-1) # shape (b, n_comp, 1)
        out_y_shape = tf.shape(out_w) + tf.constant([0, 0, self.dim_y - 1])
        out_y = tf.broadcast_to(tf.expand_dims(self.c_y, axis=0), out_y_shape)
        out = tf.concat((out_w, out_y), 2)
        return out

    def get_config(self):
        config = {
            "dim_x": self.dim_x,
            "dim_y": self.dim_y,
            "n_comp": self.n_comp,
            "x_train": self.x_train,
            "y_train": self.y_train,
            "w_train": self.w_train,
            "l1_x": self.l1_x,
            "l1_y": self.l1_y,
            "l1_act": self.l1_act,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return (self.dim_y + 1, self.n_comp)


class KQMClassModel(tf.keras.Model):
    def __init__(self,
                 dim_x,
                 dim_y,
                 n_comp,
                 x_train=True):
        super().__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.n_comp = n_comp
        self.sigma = tf.Variable(0.1, dtype=tf.float32)
        kernel_x = create_rbf_kernel(self.sigma)
        self.kqmu = KQMUnit(kernel_x,
                            dim_x=dim_x,
                            dim_y=dim_y,
                            n_comp=n_comp,
                            x_train=x_train)

    def call(self, inputs):
        rho_x = pure2dm(inputs)
        rho_y = self.kqmu(rho_x)
        y_w, y_v = dm2comp(rho_y)
        norms_y = tf.expand_dims(tf.linalg.norm(y_v, axis=-1), axis=-1)
        y_v = y_v / norms_y
        probs = tf.einsum('...j,...ji->...i', y_w, y_v ** 2, optimize="optimal")
        return probs

class BagKQMClassModel(tf.keras.Model):
    def __init__(self,
                 dim_x,
                 dim_y,
                 n_comp_in,
                 n_comp,
                 x_train=True,
                 l1_y=0.):
        super().__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.n_comp = n_comp
        self.n_comp_in = n_comp_in
        self.sigma = tf.Variable(0.1, dtype=tf.float32)
        kernel_x = create_rbf_kernel(self.sigma)
        self.kqmu = KQMUnit(kernel_x,
                            dim_x=dim_x,
                            dim_y=dim_y,
                            n_comp=n_comp,
                            x_train=x_train,
                            l1_y=l1_y)

    def call(self, inputs):
        w = tf.ones_like(inputs[:, :, 0]) / self.n_comp_in
        rho_x = comp2dm(w, inputs)
        rho_y = self.kqmu(rho_x)
        y_w, y_v = dm2comp(rho_y)
        norms_y = tf.expand_dims(tf.linalg.norm(y_v, axis=-1), axis=-1)
        y_v = y_v / norms_y
        rho_y = comp2dm(y_w, y_v)
        return rho_y


@tf.function
def overlap_kernel(A, B):
    '''
    Calculates the identity kernel between 
    two set of vectors.
    Input:
        A: tensor of shape (bs, d)
        B: tensor of shape (bs, n, d)
    Result:
        K: tensor of shape (bs, n)
    '''
    K = tf.einsum("...d,...nd->...n", A, B)
    return K

def overlap_loss(y_true, y_pred):
    y_true = tf.math.sqrt(y_true)
    overlap = pure_dm_overlap(y_true, y_pred, overlap_kernel)
    #return -tf.reduce_mean(tf.math.log(overlap + 0.0000001), axis=-1) 
    return -tf.reduce_mean(overlap , axis=-1) 

def create_resize_encoder(input_shape, encoded_size=64, filter_size=[32, 64, 128]):
    encoder = Sequential([
        InputLayer(input_shape=input_shape),
        #tfkl.Lambda(lambda x: tf.cast(x, tf.float32) - 0.5),
        Resizing(16, 16),
        Conv2D(filter_size[0], 3, 2,
                    padding='same', activation=tf.nn.relu),
        Conv2D(filter_size[1], 3, 2,
                    padding='same', activation=tf.nn.relu),
        Conv2D(filter_size[2], 3, 2,
                    padding='same', activation=tf.nn.relu),
        #tfk.layers.LayerNormalization(),
        Flatten(),
        Dense(encoded_size,
                activation=None),
                #activity_regularizer=tf.keras.regularizers.l2(1e-4)),
    ])
    return encoder
