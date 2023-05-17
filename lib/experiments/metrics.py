import tensorflow as tf
import numpy as np
from rlxutils import subplots
import matplotlib.pyplot as plt
from skimage.io import imread
import seaborn as sns
import os

def to_onehot_argmax(x):
    """
    converts probabilities to one_hot by selecting the item with highest probability,
    setting it to 1 and the rest to zero.
    selection is made on the last dimenstion

    x: tensor of shape [i, ..., number_of_classes]

    returns: a tensor with the same shape with zeros and ones

    warn: this is not a differentiable operation, since tf.argmax is not
    """
    number_of_classes = x.shape[-1]
    amax = tf.argmax(x, axis=-1)
    r = tf.reshape(tf.one_hot(tf.reshape(amax,[-1]), depth=number_of_classes), x.shape)
    return r

class PixelClassificationMetrics:
    """
    accumulates tp,tn,fp,fn per class and computes metrics on them
    accepts batches of images
    
    y_true: [batch_size, sizex, sizey]                     class labels
    y_pred: [batch_size, sizex, sizey, number_of_classes]  probabilities per class
    
    if sizex,sizey are different in y_true and y_pred the smaller is resized to the larger
    
    """
    def __init__(self, number_of_classes, exclude_classes=[]):
        self.number_of_classes = number_of_classes
        self.classes = [i for i in range(self.number_of_classes) if not i in exclude_classes]
        self.exclude_classes = exclude_classes
        self.reset_state()
        
    def reset_state(self):
        self.tp = {i:0 for i in self.classes}
        self.tn = {i:0 for i in self.classes}
        self.fp = {i:0 for i in self.classes}
        self.fn = {i:0 for i in self.classes}
        self.cm = np.zeros((self.number_of_classes, self.number_of_classes))
        self.number_of_pixels = {i:0 for i in self.classes}
        self.total_pixels = 0
        
    def update_state(self, y_true, y_pred):
        """
        y_true: input of shape [n_batches, x, y] with multiclass label for each pixel
        y_pred: input of shape [n_batches, x, y, n_classes] with class probabilities for each pixel
        """
        # resize smallest
        if y_true.shape[1]<y_pred.shape[1]:
            y_true = tf.image.resize(y_true[..., tf.newaxis], y_pred.shape[1:3], method='nearest')[:,:,:,0]

        if y_pred.shape[1]<y_true.shape[1]:
            y_pred = tf.image.resize(y_pred, y_true.shape[1:3], method='nearest')
        
        # choose class with highest probability for prediction
        y_pred = tf.argmax(y_pred, axis=-1)
        self.total_pixels += tf.cast(tf.reduce_prod(y_true.shape), tf.float32)
        for i in self.classes:
            y_true_ones  = tf.cast(y_true==i, tf.float32)
            y_true_zeros = tf.cast(y_true!=i, tf.float32)
            y_pred_ones  = tf.cast(y_pred==i, tf.float32)
            y_pred_zeros = tf.cast(y_pred!=i, tf.float32)
            
            self.tp[i] += tf.reduce_sum(y_true_ones  * y_pred_ones)
            self.tn[i] += tf.reduce_sum(y_true_zeros * y_pred_zeros)
            self.fp[i] += tf.reduce_sum(y_true_zeros * y_pred_ones)
            self.fn[i] += tf.reduce_sum(y_true_ones  * y_pred_zeros)
            self.number_of_pixels[i] += tf.reduce_sum(y_true_ones)

        # update confusion matrix
        for classid in range(self.number_of_classes):
            y_pred_classid = y_pred[y_true==classid]
            for c,n in zip(*np.unique(y_pred_classid, return_counts=True)):
                self.cm[classid, c] += n

        return self
            
    def accuracy(self, tp, tn, fp, fn):
        denominator = tp + tn + fp + fn
        if denominator == 0:
            return np.nan
        else:
            return (tp + tn)/denominator
    
    def f1(self, tp, tn, fp, fn):
        denominator = tp + tf.cast(fp+fn, tf.float32).numpy() / 2
        if denominator == 0:
            return np.nan
        return tp / denominator
    
    def precision(self, tp, tn, fp, fn):
        denominator = tp+fp
        if denominator == 0:
            return np.nan
        return tp / denominator

    def iou(self, tp, tn, fp, fn):
        denominator = tp + fp + fn
        if denominator == 0:
            return np.nan
        else:
            return tp / denominator

    def result(self, metric_name, mode='micro'):
        if metric_name=='confusion_matrix':
            return self.cm

        if not mode in ['per_class', 'micro', 'macro', 'weighted']:
            raise ValueError(f"invalid mode '{mode}'")
        
        m = eval(f'self.{metric_name}')
        
        if mode=='per_class':
            r = {i: m(self.tp[i], self.tn[i], self.fp[i], self.fn[i]) for i in self.classes}
            return r
        
        if mode=='macro':
            r = {i: m(self.tp[i], self.tn[i], self.fp[i], self.fn[i]) for i in self.classes}
            r = [i for i in r.values() if not np.isnan(i)]
            if len(r)==0:
                return 0.
            else:
                return tf.reduce_mean(r)

        if mode=='weighted':
            total_pixels = tf.reduce_sum(list(self.number_of_pixels.values()))
            if total_pixels == 0:
                return 1.
            r = []
            for i in self.classes:
                metric = m(self.tp[i], self.tn[i], self.fp[i], self.fn[i])
                if not np.isnan(metric):
                    r.append(metric*self.number_of_pixels[i] / total_pixels)

            if len(r)==0:
                return 0.
            else:
                return tf.reduce_sum(r)
        
        if mode=='micro':
            tp = tf.reduce_sum(list(self.tp.values()))
            tn = tf.reduce_sum(list(self.tn.values()))
            fp = tf.reduce_sum(list(self.fp.values()))
            fn = tf.reduce_sum(list(self.fn.values()))
            r = m(tp, tn, fp, fn)
            if np.isnan(r):
                return 0
            else:
                return r

def pnorm(v,p):
    """
    the p-norm of a vector 

    v: a batch of vectors of shape [num_vectors, vector_length]
    p: the p of the norm, can be < 1
    """
    #return tf.math.pow(tf.reduce_sum(tf.math.pow(np.r_[v], p), axis=-1), 1/p)
    return tf.norm(v, ord=p, axis=-1)


def plot_confusion_matrix(cm):

    nc = cm.shape[0]

    fig = plt.figure(figsize=(6+nc/5, 6+nc/5))
    # Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
    # the size of the marginal axes and the main axes in both directions.
    # Also adjust the subplot parameters for a square plot.
    gs = fig.add_gridspec(2, 2,  width_ratios=(6, 1), height_ratios=(1, 6),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.05)
    # Create the Axes.
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)


    sns.heatmap(cm/(np.sum(cm, axis=1).reshape(-1,1)+1e-10), fmt='.1%', annot=True, 
                cmap='Blues', cbar=False, ax=ax,
                annot_kws={"size": 6})
    ax.set_xlabel("y_pred", fontdict = {'fontsize': 8})
    ax.set_ylabel("y_true\n(rows add up to one)", fontdict = {'fontsize': 8})
    ax.set_xticklabels(range(nc), fontdict = {'fontsize': 8})
    ax.set_yticklabels(range(nc), fontdict = {'fontsize': 8})


    y_pred_distrib = cm.sum(axis=0)/cm.sum()
    y_true_distrib = cm.sum(axis=1)/cm.sum()
    ax_histx.bar(np.arange(nc) + 0.5, y_pred_distrib, alpha=.5, color="steelblue")
    ax_histx.set_title("y_pred class distribution")
    ax_histx.title.set_fontsize(8)
    ax_histx.grid()

    xy_ticks = [0.2, 0.4, 0.6, 0.8]
    ax_histx.set_yticks(xy_ticks)
    ax_histx.set_yticklabels(xy_ticks, fontdict = {'fontsize': 8})
    ax_histx.set_ylim(0,1)
    for spine in ax_histx.spines.values():
        spine.set_edgecolor('gray')
    ax_histx.tick_params(axis='x', pad=0, direction='out', labelbottom=False)

    ax_histy.barh(np.arange(nc) + 0.5, y_true_distrib, alpha=.5, color="steelblue")
    ax_histy.set_ylabel("y_true class distribution", 
                        rotation=270, labelpad=-60, 
                        fontdict = {'fontsize': 8})
    ax_histy.grid()
    yx_ticks = [0.2, 0.4, 0.6, 0.8]
    ax_histy.set_xticks(yx_ticks)
    ax_histy.set_xticklabels(yx_ticks, fontdict = {'fontsize': 8}, rotation=270)
    ax_histy.set_xlim(0,1)
    for spine in ax_histy.spines.values():
        spine.set_edgecolor('gray')
    ax_histy.tick_params(axis='y', pad=0, direction='out', labelleft=False)

    tmpfname = f"/tmp/{np.random.randint(1000000)}.png"
    plt.savefig(tmpfname)
    plt.close(fig)       
    img = imread(tmpfname)
    os.remove(tmpfname)
    return img

class ProportionsMetrics:
    """
    class containing methods for label proportions metrics and losses
    """

    def __init__(self, class_weights_values, 
                       mse_proportions_argmax=False, 
                       rmse_proportions_argmax=False,
                       kldiv_proportions_argmax=False, 
                       mae_proportions_argmax=True,
                       rmae_proportions_argmax=True,
                       p_for_norm = 1, lambda_reg=0.0):
        """
        proportions_argmax: see get_y_pred_as_proportions
        """
        self.class_weights_values = np.r_[class_weights_values]
        self.number_of_classes = len(self.class_weights_values)
        self.mse_proportions_argmax = mse_proportions_argmax
        self.rmse_proportions_argmax = rmse_proportions_argmax
        self.mae_proportions_argmax = mae_proportions_argmax
        self.rmae_proportions_argmax = rmae_proportions_argmax
        self.kldiv_proportions_argmax = kldiv_proportions_argmax
        self.p_for_norm = p_for_norm
        self.lambda_reg = lambda_reg

        self.max_pnorm = tf.cast(pnorm(np.ones(self.number_of_classes) / self.number_of_classes, self.p_for_norm), tf.float32)
        

    def generate_y_true(self, batch_size, pixel_size=96, max_samples=5):
        """
        generate a sample of label masks of shape batch_size x pixel_size x pixel_size
        each pixel being an integer value in [0 .. number_of_classes]
        returns a numpy array
        """
        y_true = np.zeros((batch_size,pixel_size,pixel_size))
        for i in range(batch_size):
            for class_id in range(1,self.number_of_classes):
                for _ in range(np.random.randint(max_samples-j)+1):
                    size = np.random.randint(30)+10
                    x,y = np.random.randint(y_true.shape[1]-size, size=2)
                    y_true[i,y:y+size,x:x+size] = class_id
        return y_true.astype(np.float32)

    def generate_y_pred(self, batch_size, pixel_size=96, max_samples=5, noise=2):
        """
        generate a sample of probability predictions of shape [batch_size, pixel_size, pixel_size, number_of_classes]
        each pixel being a probability in [0,1] softmaxed so the each pixel probabilities add up to 1.
        returns a numpy array
        """        
        y_true = self.generate_y_true(batch_size=batch_size, pixel_size=pixel_size, max_samples=max_samples)
        y_pred = np.zeros((batch_size, pixel_size, pixel_size, self.number_of_classes))

        # set classes
        for class_id in range(self.number_of_classes):
            y_pred[:,:,:,class_id][y_true==class_id] = 1.

        # add random noise    
        y_pred += np.random.random(y_pred.shape)*noise

        # softmax for probabilities across pixels
        y_pred = np.exp(y_pred) / np.exp(y_pred).sum(axis=-1)[:,:,:,np.newaxis]
        return y_pred.astype(np.float32)

    def get_class_proportions_on_masks(self, y_true):
        """
        obtains the class proportions in a label mask.
        y_true: int array of shape [batch_size, pixel_size, pixel_size]
        returns an array [batch_size, number_of_classes]
        """
        return np.r_[[[np.mean(y_true[i]==class_id)  for class_id in range(self.number_of_classes)] for i in range(len(y_true))]].astype(np.float32)

    def get_class_proportions_on_probabilities(self, y_pred):
        """
        obtains the class proportions on probability predictions
        y_pred: float array of shape [batch_size, pixel_size, pixel_size, number_of_classes]
        returns: a tf tensor of shape [batch_size, number_of_classes]
        """
        assert y_pred.shape[-1] == self.number_of_classes
        return tf.reduce_mean(y_pred, axis=[1,2])
    
    def get_y_pred_as_masks(self, y_pred):
        """
        converts probability predictions to masks by selecting in each pixel the class with highest probability.
        """
        assert y_pred.shape[-1] == self.number_of_classes
        y_pred_as_label = np.zeros(y_pred.shape[:-1]).astype(int)
        t = y_pred.argmax(axis=-1)
        for i in range(self.number_of_classes):
            y_pred_as_label[t==i] = i
        return y_pred_as_label
    
    def get_y_true_as_probabilities(self, y_true):
        """
        converts masks to probability predictions by setting prob=1 or 0
        """
        r = np.zeros(list(y_true.shape) + [self.number_of_classes])
        for class_id in range(self.number_of_classes):
            r[:,:,:,class_id] = y_true==class_id
        return r.astype(np.float32)
    
    
    def get_y_pred_as_proportions(self, y_pred, argmax=False):
        """
        y_pred: a tf tensor of shape [batch_size, pixel_size, pixel_size, number_of_classes] with 
                probability predictions per pixel (such as the output of a softmax layer,so that class 
                proportions will be computed from it), or with shape [batch_size, number_of_classes] 
                directly with the class proportions (must add up to 1).
        argmax: if true compute proportions by selecting the class with highest assigned probability 
                in each pixel and then computing the proportions of selected classes across each image.
                If False, the class proportions will be computed by averaging the probabilities in 
                each class channel. If none, it will use self.proportions_argmax
        
        returns: a tf tensor of shape [batch_size, number_of_classes]
                 if input has shape [batch_size, number_of_classes], the input is returned untouched

        warn: if argmax=True the result will not be differentiable since tf.argmax is not
        """
        assert (len(y_pred.shape)==4 or len(y_pred.shape)==2) and y_pred.shape[-1]==self.number_of_classes

        if argmax is None:
            argmax = False

        # compute the proportions on prediction
        if len(y_pred.shape)==4:
            # if we have probability predictions per pixel (softmax output)
            if argmax:
                r = to_onehot_argmax(y_pred)
                r = tf.reduce_mean(r, axis=[1,2])
            else:
                # compute the proportions by averaging each class. requires probabilities across each pixel
                # to add up to 1. Previous softmax output guarantees all will add up to one.
                r = tf.reduce_mean(y_pred, axis=[1,2])
        else:
            # if we already have a probabilities vector, return it as such
            r = y_pred

        return r        
    
    def pixel_level_categorical_cross_entropy(self, y_true, y_pred):
        """
        computes the categorical cross entropy between to segmentation maps
        of, possibly, different sizes by resizing the smaller to the size
        of the larger.

        y_pred: must have shape (batch_size, pixels_width, pixels_heigh, number_of_classes) 
                with probabilities in each pixel adding up to one
        y_true: must have shape (batch_size, pixels_width, pixels_heigh, number_of_classes)
                with a one hot encoding of the target labels
                or shape (batch_size, pixels_width, pixels_heigh) with integer 0..number_of_classes
                which will be transformed to one_hot
        """
        # resize smallest
        if y_true.shape[1]<y_pred.shape[1]:
            y_true = tf.image.resize(y_true[..., tf.newaxis], y_pred.shape[1:3], method='nearest')[:,:,:,0]

        if y_pred.shape[1]<y_true.shape[1]:
            y_pred = tf.image.resize(y_pred, y_true.shape[1:3], method='nearest')

        if len(y_true.shape)==3:
            y_true = tf.one_hot(tf.cast(y_true, tf.uint8), self.number_of_classes)

        if not y_pred.shape[-1]==self.number_of_classes:
            raise ValueError(f"incorrect number of classes in y_pred, found {y_pred.shape[-1]} but expected {self.number_of_classes}")

        if not y_true.shape[-1]==self.number_of_classes:
            raise ValueError(f"incorrect number of classes in y_trye, found {y_true.shape[-1]} but expected {self.number_of_classes}")

        cce = -tf.reduce_mean(tf.reduce_sum(tf.math.log(y_pred+1e-5)*y_true, axis=-1))        
        return cce
    
    def multiclass_proportions_mse(self, true_proportions, y_pred):
        """
        computes the mse between proportions on probability predictions (y_pred)
        and target_proportions, using the class_weights in this instance.
        
        y_pred:  see y_pred in get_y_pred_as_proportions

        returns: a float with mse.
        """
                        
        assert len(true_proportions.shape)==2 and true_proportions.shape[-1]==self.number_of_classes

        # compute the proportions on prediction
        proportions_y_pred = self.get_y_pred_as_proportions(y_pred, argmax = self.mse_proportions_argmax)

        # regularization promoting sparsity. divide by max_pnorm to ensure value in [0,1]
        reg = tf.reduce_mean(
                    self.lambda_reg * pnorm(proportions_y_pred, self.p_for_norm) / self.max_pnorm
                )

        # compute mse using class weights
        r = tf.reduce_mean(
                tf.reduce_sum(
                    (true_proportions - proportions_y_pred)**2 * self.class_weights_values, 
                    axis=-1
                )
        )
        return r + reg

    def multiclass_proportions_rmse(self, true_proportions, y_pred):
        """
        computes the mse between proportions on probability predictions (y_pred)
        and target_proportions, using the class_weights in this instance.
        
        y_pred:  see y_pred in get_y_pred_as_proportions

        returns: a float with mse.
        """
                        
        assert len(true_proportions.shape)==2 and true_proportions.shape[-1]==self.number_of_classes

        # compute the proportions on prediction
        proportions_y_pred = self.get_y_pred_as_proportions(y_pred, argmax = self.rmse_proportions_argmax)

        # regularization promoting sparsity. divide by max_pnorm to ensure value in [0,1]
        reg = tf.reduce_mean(
                    self.lambda_reg * pnorm(proportions_y_pred, self.p_for_norm) / self.max_pnorm
                )

        # compute mse using class weights
        r = tf.reduce_mean(
                tf.reduce_sum(
                    ((true_proportions - proportions_y_pred)/(proportions_y_pred + 1e-5))**2 * self.class_weights_values, 
                    axis=-1
                )
        )
        return r + reg


    def kldiv(self, true_proportions, y_pred):
        """
        computes the kl divergence between two proportions of labels
        y_pred:  see y_pred in get_y_pred_as_proportions

        returns: a float .
        """

        assert len(true_proportions.shape)==2 and true_proportions.shape[-1]==self.number_of_classes

        # compute the proportions on prediction
        proportions_y_pred = self.get_y_pred_as_proportions(y_pred, argmax = self.kldiv_proportions_argmax)
        # compute mse using class weights
        r = tf.reduce_mean(
                tf.reduce_sum(
                    true_proportions \
                    * (tf.math.log(true_proportions + 1e-5) - tf.math.log(proportions_y_pred + 1e-5)) \
                    * self.class_weights_values, 
                    
                    axis=-1
                )
        )
        return r    

    def ilogkldiv(self, true_proportions, y_pred):
        """
        computes the kl divergence between two proportions of labels
        y_pred:  see y_pred in get_y_pred_as_proportions

        returns: a float .
        """

        assert len(true_proportions.shape)==2 and true_proportions.shape[-1]==self.number_of_classes

        # compute the proportions on prediction
        proportions_y_pred = self.get_y_pred_as_proportions(y_pred, argmax = self.kldiv_proportions_argmax)
        # compute mse using class weights
        r = tf.reduce_mean(
                -1 / (
                    tf.math.log( 1e-5 + \
                        tf.reduce_sum(
                                    true_proportions \
                                    * (tf.math.log(true_proportions + 1e-5) - tf.math.log(proportions_y_pred + 1e-5)) \
                                    * self.class_weights_values, 
                                axis=-1
                            )
                    )
                )                    
        )
        return r         
        
    def multiclass_proportions_mae(self, true_proportions, y_pred, perclass=False):
        """
        computes the mae between proportions on probability predictions (y_pred)
        and target_proportions. NO CLASS WEIGHTS ARE USED.

        y_pred: see y_pred in get_y_pred_as_proportions
        perclass: if true returns a vector of length num_classes with the mae on each class

        returns: a float with mse if perclass=False, otherwise a vector
        """
                        
        assert len(true_proportions.shape)==2 and true_proportions.shape[-1]==self.number_of_classes

        # compute the proportions on prediction
        proportions_y_pred = self.get_y_pred_as_proportions(y_pred, argmax = self.mae_proportions_argmax)

        # compute mae per class
        r = tf.reduce_mean(
            tf.sqrt((true_proportions - proportions_y_pred)**2),
            axis=0
        )

        # return mean if perclass is not required
        if not perclass:
            r = tf.reduce_mean(r)
            
        return r
    

    def proportions_cross_entropy(self, true_proportions, y_pred, perclass=False):
        """
        computes the categorical cross extropy between proportions on probability predictions (y_pred)
        and target_proportions. NO CLASS WEIGHTS ARE USED.

        y_pred: see y_pred in get_y_pred_as_proportions
        perclass: not used

        returns: a float with mse if perclass=False, otherwise a vector
        """
                        
        assert len(true_proportions.shape)==2 and true_proportions.shape[-1]==self.number_of_classes

        # compute the proportions on prediction
        proportions_y_pred = self.get_y_pred_as_proportions(y_pred, argmax = self.mae_proportions_argmax)

        # compute categorical cross entropy per class
        r = tf.reduce_mean(
                -tf.reduce_sum(
                            true_proportions*tf.math.log(proportions_y_pred+1e-5), 
                            axis=1
                            )
                        )            
        return r    
    
    def multiclass_proportions_rmae(self, true_proportions, y_pred, perclass=False):
        """
        computes the relative mae between proportions on probability predictions (y_pred)
        and target_proportions. CLASS WEIGHTS ARE USED.
        
        relative mae is defined as sqrt( ( (y_true-y_pred)/(y_pred+1e-5) )**2 )

        y_pred: see y_pred in get_y_pred_as_proportions
        perclass: if true returns a vector of length num_classes with the mae on each class

        returns: a float with mse if perclass=False, otherwise a vector
        """
                        
        assert len(true_proportions.shape)==2 and true_proportions.shape[-1]==self.number_of_classes

        # compute the proportions on prediction
        proportions_y_pred = self.get_y_pred_as_proportions(y_pred, argmax = self.rmae_proportions_argmax)

        cw = 1 if perclass else self.class_weights_values

        # compute mae per class
        r = tf.reduce_mean(
            tf.sqrt(((true_proportions - proportions_y_pred) / ( proportions_y_pred + 1e-5) )**2) * cw,
            axis=0
        )
        # return mean if perclass is not required
        if not perclass:
            r = tf.reduce_sum(r)
            
        return r    


    def multiclass_LSRN_loss(self, true_proportions, y_pred):
        """
        computes the loss proposed in:
        
         Malkin, Kolya, et al. "Label super-resolution networks." 
         International Conference on Learning Representations. 2018

        y_pred: a tf tensor of shape [batch_size, pixel_size, pixel_size, number_of_classes] with probability predictions
        target_proportions: a tf tensor of shape [batch_size, number_of_classes]
        
        
        returns: a float with the loss.
        """

        assert len(y_pred.shape)==4 and y_pred.shape[-1]==self.number_of_classes
        assert len(true_proportions.shape)==2 and true_proportions.shape[-1]==self.number_of_classes
        
        eta = true_proportions

        # compute the proportions on prediction (mu)
        mu = tf.reduce_mean(y_pred, axis=[1,2])

        # compute variances (sigma^2)
        block_size = y_pred.shape[1] * y_pred.shape[2]
        sigma_2 = (tf.reduce_sum(y_pred * (1 - y_pred), 
                                axis=[1,2]) / block_size ** 2)
        # compute loss
        loss = tf.reduce_mean(
                tf.reduce_sum(
                    0.8 * (eta - mu)**2 / (sigma_2) +
                    0.2 * sigma_2, 
                    #0.5 * tf.math.log(2 * np.pi * sigma_2), 
                    axis=-1
                )
        )
        return loss

    def multiclass_proportions_mae_on_chip(self, y_true, y_pred, perclass=False):
        """
        computes the mse between the proportions observed in a prediction wrt to a mask
        y_pred: a tf tensor of shape [batch_size, pixel_size, pixel_size, number_of_classes] with probability predictions
        y_true: int array of shape [batch_size, pixel_size, pixel_size]
        perclass: see multiclass_proportions_mae
        argmax: see multiclass_proportions_mae

        returns: a float with mse
        """
        p_true = self.get_class_proportions_on_masks(y_true)
        return self.multiclass_proportions_mae(p_true, y_pred, perclass=perclass)

    def proportions_cross_entropy_on_chip(self, y_true, y_pred, perclass=False):
        """
        computes the categorical cross entropy between the proportions observed in a prediction wrt to a mask
        y_pred: a tf tensor of shape [batch_size, pixel_size, pixel_size, number_of_classes] with probability predictions
        y_true: int array of shape [batch_size, pixel_size, pixel_size]
        perclass: not used
        argmax: see multiclass_proportions_mae

        returns: a float with mse
        """
        p_true = self.get_class_proportions_on_masks(y_true)
        return self.proportions_cross_entropy(p_true, y_pred)



    def multiclass_proportions_rmae_on_chip(self, y_true, y_pred, perclass=False):
        """
        computes the mse between the proportions observed in a prediction wrt to a mask
        y_pred: a tf tensor of shape [batch_size, pixel_size, pixel_size, number_of_classes] with probability predictions
        y_true: int array of shape [batch_size, pixel_size, pixel_size]
        perclass: see multiclass_proportions_mae
        argmax: see multiclass_proportions_mae

        returns: a float with mse
        """
        p_true = self.get_class_proportions_on_masks(y_true)
        return self.multiclass_proportions_rmae(p_true, y_pred, perclass=perclass)


    def compute_iou(self, y_true, y_pred):
        """
        computes iou using the formula tp / (tp + fp + fn) for each individual image.
        for each image, it computes the iou for each class and then averages only over
        the classes containing pixels in that image in y_true or y_pred.
        NO CLASS WEIGHTS ARE USED.
        """

        if y_true.shape[1]<y_pred.shape[1]:
            y_true = tf.image.resize(y_true[..., tf.newaxis], y_pred.shape[1:], method='nearest')[:,:,:,0]

        if y_pred.shape[1]<y_true.shape[1]:
            y_pred = tf.image.resize(y_pred, y_true.shape[1:], method='nearest')

        itemclass_iou = []
        itemclass_true_or_pred_ones = []
        y_pred = tf.argmax(y_pred, axis=-1)
        for class_id in range(self.number_of_classes):
            y_true_ones  = tf.cast(y_true==class_id, tf.float32) 
            y_pred_ones  = tf.cast(y_pred==class_id, tf.float32)
            y_true_zeros = tf.cast(y_true!=class_id, tf.float32) 
            y_pred_zeros = tf.cast(y_pred!=class_id, tf.float32)

            tp = tf.reduce_sum(y_true_ones  * y_pred_ones, axis=[1,2])
            fp = tf.reduce_sum(y_true_zeros * y_pred_ones, axis=[1,2])
            fn = tf.reduce_sum(y_true_ones  * y_pred_zeros, axis=[1,2])

            true_or_pred_ones = tf.cast(tf.reduce_sum(y_true_ones + y_pred_ones, axis=[1,2])>0, tf.float32)
            iou_this_class = tp / (tp + fp + fn)

            # substitute nans with zeros
            iou_this_class = tf.where(tf.math.is_nan(iou_this_class), tf.zeros_like(iou_this_class), iou_this_class)

            itemclass_iou.append(iou_this_class)
            itemclass_true_or_pred_ones.append(true_or_pred_ones)
            

        itemclass_iou = tf.convert_to_tensor(itemclass_iou)
        itemclass_true_or_pred_ones = tf.convert_to_tensor(itemclass_true_or_pred_ones)
        # only compute the mean of the classes with pixels in y_true or y_pred
        per_item_iou = tf.reduce_sum(itemclass_iou, axis=0)/tf.reduce_sum(itemclass_true_or_pred_ones, axis=0)
        per_item_iou = tf.where(tf.math.is_nan(per_item_iou), tf.ones_like(per_item_iou), per_item_iou)

        return tf.reduce_mean(per_item_iou)   

    def show_y_pred(self, y_pred):
        for n in range(len(y_pred)):
            for ax,i in subplots(self.number_of_classes, usizex=4):
                plt.imshow(y_pred[n,:,:,i]>=0.5)
                plt.title(f"item {n}, class {i}, m {np.mean(y_pred[n,:,:,i]>=0.5):.3f}")
                plt.colorbar();

    def show_y_true(self, y_true):
        for ax,i in subplots(len(y_true)):
            plt.imshow(y_true[i], vmin=0, vmax=11, cmap=plt.cm.tab20b, interpolation="none")
            plt.colorbar()
            plt.title(f"item {i}")

