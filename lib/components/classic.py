import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Resizing, InputLayer, \
                                    Conv2DTranspose, Input, Flatten, concatenate, Lambda, Dense
from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.models import Model
import tensorflow_probability as tfp
import wget
import os
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
from rlxutils import subplots

tfd = tfp.distributions


def get_alexnet_weights(kernel_shape=None, dtype=np.float32):
    """
    Downloads (if needed) alexnet weights of first layer and resizes them
    to (kernel_size, kernel_size)
    """ 
    assert kernel_shape[0]==kernel_shape[1], "initializing alexnet weights: kernels must be square"
    assert kernel_shape[2]==3, "initializing alexnet weights: images must have 3 channels"

    kernel_size = kernel_shape[0]
    home = os.environ['HOME']
    dest_dir = f"{home}/.alexnet"
    weights_file = f"{dest_dir}/bvlc_alexnet.npy"
    url = 'https://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy'
    os.makedirs(dest_dir, exist_ok=True)

    do_download = True
    if os.path.isfile(weights_file):
        if os.path.getsize(weights_file)!=243861814:
            print ("alexnet weights are corrupt! removing them")
            os.remove(weights_file)
        else:
            do_download = False

    if do_download:
        print ("downloading alexnet weights", flush=True)
        wget.download(url, weights_file)
        
    w = np.load(open(weights_file, "rb"), allow_pickle=True, encoding="latin1").item()
    conv1_weights, conv1_bias =  w['conv1']
    # normalize to -1, 1
    conv1_weights = 2 * (conv1_weights-np.min(conv1_weights))/(np.max(conv1_weights)-np.min(conv1_weights)) - 1
    conv1_weights = np.transpose(conv1_weights, [3,0,1,2])
    
    if kernel_size is not None:
        conv1_weights = np.r_[[resize(i, (kernel_size, kernel_size)) for i in conv1_weights]]
    
    conv1_weights = np.transpose(conv1_weights, [1,2,3,0])
    return tf.convert_to_tensor(conv1_weights, dtype=tf.float32)

def plot_rgb_weights(w, n_cols=10):
    print (f"original input pixel values in range [{w.min()} - {w.max()}]")
    w = (w-w.min())/(w.max()-w.min())
    for ax, i in subplots(w.shape[-1], usizex=0.5, usizey=0.5, n_cols=n_cols):
        plt.imshow(w[:,:,:,i])
        plt.axis("off")

class Conv2DBlock(tf.keras.layers.Layer):
    """
    example execution:
    
        conv_layers = [
            dict(kernel_size=6, filters=3, activation='relu', padding='same', dropout=0.1, maxpool=2),
            dict(kernel_size=6, filters=3, activation='relu', dropout=0.1, maxpool=2)
        ]

        c = Conv2DBlock(conv_layers)

    start_n, name_prefix: used to build the layer names
    conv_layers: a list of kwargs which will be passed to Conv2D.
                'dropout' or 'maxpool' keywords will be extracted before calling Conv2D
                and, if present, Dropout and MaxPooling2D layers will be added with the 
                value specified.
    """
    def __init__(self, conv_layers, use_alexnet_weights=False, start_n=1, name_prefix=""):
        super().__init__()
        
        conv_layers = conv_layers.copy()
        self.use_alexnet_weights = use_alexnet_weights

        n = start_n - 1
        self.layers = []
        first_layer = True
        for kwargs in conv_layers:
            n += 1
            kwargs = kwargs.copy()
            dropout = maxpool = None

            print ("convlayer", kwargs, flush=True)

            if 'dropout' in kwargs.keys():
                dropout = kwargs['dropout']
                del(kwargs['dropout'])

            if 'maxpool' in kwargs.keys():
                maxpool = kwargs['maxpool']
                del(kwargs['maxpool'])

            if not 'name' in kwargs.keys():
                kwargs['name'] = f"{name_prefix}{n:02d}_conv2d"

            if first_layer and self.use_alexnet_weights:
                kwargs['kernel_initializer'] = get_alexnet_weights

            self.layers.append(Conv2D(**kwargs))

            if dropout is not None:
                self.layers.append(Dropout(dropout, name=f'{name_prefix}{n:02d}_dropout'))

            if maxpool is not None:
                self.layers.append(MaxPooling2D(pool_size=maxpool, name=f'{name_prefix}{n:02d}_maxpool'))
            
            first_layer = False
        self.end_n = n
                
    def call(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x
    
class DenseBlock(tf.keras.layers.Layer):
    
    """
    example:

        dense_layers = [
            dict(units=100, activation='relu', dropout=0.2),
            dict(units=50, activation='relu')
        ]    
    
        cc = Conv2DBlock(conv_layers, start_n = 1)
        dd = DenseBlock(dense_layers, start_n = cc.end_n+1)
        x = Input((96,96,3))
        x = cc(x)
        x = Flatten()(x)
        x = dd(x)
        
    start_n, name_prefix: used to build the layer names
    
    dense_layers: a list of kwargs which will be passed to Dense.
                  the 'dropout' keyword will be extracted before calling Dense and, 
                  if present, a Dropout layer will be added with the value specified

    """
    
    def __init__(self, dense_layers, start_n=1, name_prefix=""):
        super().__init__()

        dense_layers = dense_layers.copy()
        self.layers = []
        
        n = start_n-1

        for kwargs in dense_layers:
            n += 1      
            kwargs = kwargs.copy()
            dropout = None
            if 'dropout' in kwargs.keys():
                dropout = kwargs['dropout']
                del(kwargs['dropout'])

            if not 'name' in kwargs.keys():
                kwargs['name'] = f"{name_prefix}{n:02d}_dense"

            self.layers.append(Dense(**kwargs))

            if dropout is not None:
                self.layers.append(Dropout(dropout, name=f'{name_prefix}{n:02d}_dropout'))

        self.end_n = n
        
    def call(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x


class GaussianMixtureLayer(tf.keras.layers.Layer):

    def __init__(self, nb_gaussians, name=None ):
        super().__init__(name=name)
        self.nb_gaussians  = nb_gaussians        
        #self._mu    = tf.Variable(np.random.random((nb_gaussians, dim)), dtype=tf.float32, name="_mu")
        #self._sigma = tf.Variable(np.random.random((nb_gaussians,1)), dtype=tf.float32, name="_sigma")
        
        
    def build(self, input_shape):
        
        self.dim = input_shape[-1]
        self._mu = self.add_weight(
            shape=(self.nb_gaussians, self.dim),
            initializer="random_normal",
            trainable=True,
            name="gmmu"
        )

        self._sigma = self.add_weight(
            shape=(self.nb_gaussians,1),
            initializer="random_normal",
            trainable=True,
            name="gmsigma"
        )        

    def call(self, inputs):

        pi = (np.ones(self.nb_gaussians)/self.nb_gaussians).astype(np.float32)
        self._gmm = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=pi),
            components_distribution = tfd.MultivariateNormalDiag(
                    loc = self._mu,
                    scale_diag = tf.repeat(tf.math.abs(self._sigma) + 1e-5,
                                           2, axis=1)
                )
        )
        
        outputs =  self._gmm.log_prob(inputs)
        
        repeated_inputs = tf.reshape(tf.repeat(inputs, [self.nb_gaussians], axis=0), 
                                     [-1,self.nb_gaussians,self.dim])
        output_per_distribution = self._gmm.components_distribution.log_prob(repeated_inputs)
        
        return outputs, output_per_distribution



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