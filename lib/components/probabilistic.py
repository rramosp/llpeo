from .classic import *
from ..utils.autoinit import *
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
import tensorflow_probability as tfp
tfd = tfp.distributions


class GaussianMixtureLayer(tf.keras.layers.Layer):
    
    def __init__(self, number_of_distributions,
                       sigma_value = None,
                       sigma_trainable = True,
                       categorical_weights = 1,
                       categorical_trainable = False,
                       name=None):
        """
        a set of multivariate normals each with each own mean and scalar std (identity*scalar_std covariance matrix)
        """
        super().__init__(name=name)
        self.number_of_distributions = number_of_distributions
        self.sigma_value = sigma_value
        self.sigma_trainable = sigma_trainable
        self.categorical_weights = categorical_weights
        self.categorical_trainable = categorical_trainable
        
    def build(self, input_shape):
        self.dim = input_shape[-1]
        
        self._mu = self.add_weight(
            shape=(self.number_of_distributions, self.dim),
            initializer="random_normal",
            trainable=True,
            name="gmmu"
        )

        self._sigma = self.add_weight(
             shape=(self.number_of_distributions,1),
             initializer="random_normal" if self.sigma_value is None else tf.constant_initializer(self.sigma_value),
             trainable=self.sigma_trainable,
             name="gmsigma"
         )  
                
        self._categorical_weights = self.add_weight(
             shape=(self.number_of_distributions, 1),
             initializer="random_normal" if self.categorical_weights is None else tf.constant_initializer(self.categorical_weights),
             trainable=self.categorical_trainable,
             name="gmweights"
         )  
        
    @tf.function
    def log_prob(self, x):
        log = tf.math.log
        exp = tf.math.exp
        
        dim = self.dim
        s = tf.math.abs(self._sigma)+1e-5
        m = tf.reshape(self._mu,[self.number_of_distributions,1,dim])
        
        # pass categorical weights through a softmax to ensure they add up to 1
        w = tf.math.softmax(self._categorical_weights, axis=0)

        # compute log probs for each distribution
        log_probs_per_distrib = -0.5 * (tf.reduce_sum((x - m)**2/tf.reshape(s,(-1,1,1)), axis=-1) \
                                        + dim*np.log(2*np.pi) + dim*tf.reshape(log(s), (-1,1)))
        
        # aggregate probs according to categorical weights
        log_probs = log(tf.reduce_sum(exp(log_probs_per_distrib) * w, axis=0))
        log_probs_per_distrib = tf.transpose(log_probs_per_distrib, (1,0))
        
        return log_probs, log_probs_per_distrib
    
    def call(self, x):
        return self.log_prob(x)
