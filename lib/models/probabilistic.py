import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
from progressbar import progressbar as pbar
from collections import defaultdict
import pandas as pd
from time import time

from ..components.classic import *
from ..components.probabilistic import *
from ..utils.autoinit import *

psums = lambda x: " ".join([f'{np.sum(np.abs(i)):.4f}' if i is not None else 'None' for i in x])
log = tf.math.log
exp = tf.math.exp


def plot_model(x, y=None, m=None, contours='Pz'):

    assert contours in ['Pz', 'Pc_given_z', 'Pc_given_z/boundary']
    
    if m is not None and 'encoder' in dir(m) and m.encoder is not None:
        x = m.encoder(x).numpy()
    
    sx = np.random.random(size=(10000,x.shape[-1]))
    sx = sx*(x.max(axis=0) - x.min(axis=0) ) + x.min(axis=0)
    if m is not None:
        logPz, Pc_given_z = m(sx)
        levels = 14
        vmin=vmax=None
        if contours=='Pz':
            contour_value = logPz.numpy()
        elif contours == 'Pc_given_z':
            contour_value = Pc_given_z.numpy()
        else:
            contour_value = (Pc_given_z.numpy()>0.5).astype(float)
            vmin=0
            vmax=1
            levels=2
            
        plt.tricontour (sx[:,0], sx[:,1], contour_value, levels=levels, vmin=vmin, vmax=vmax, linewidths=0.5, colors='k')
        plt.tricontourf(sx[:,0], sx[:,1], contour_value, levels=levels, vmin=vmin, vmax=vmax, cmap="viridis")
        plt.colorbar()
        
    if y is None or len(x[y==0])==0 or len(x[y==1])==0:
        plt.scatter(x[:,0], x[:,1], color="red", s=10, alpha=.5)
    else:
        plt.scatter(x[y==0][:,0], x[y==0][:,1], color="red", s=10, alpha=.5)
        plt.scatter(x[y==1][:,0], x[y==1][:,1], color="blue", s=10, alpha=.5)

    if m is not None: 
        c = m.gaussian_mixture_block.get_weights()[0]
        plt.scatter(c[:,0], c[:,1], color='white')   
        
    plt.xlim(x[:,0].min(), x[:,0].max())
    plt.ylim(x[:,1].min(), x[:,1].max())
    plt.title(contours)
        

class GMMPrototypes(tf.keras.Model):

    def __init__(self, 
                 number_of_gaussians, 
                 encoder_layers,
                 gm_sigma_value = None,
                 gm_sigma_trainable = True,
                 gm_categorical_weights = 1.,
                 gm_categorical_trainable = False,
                 Pc_given_rk_trainable = True,
                 ):
        super().__init__()
        autoinit(self)
        
        if encoder_layers is not None:
            self.encoder = DenseBlock(self.encoder_layers)
        else:
            self.encoder = None

        self.gaussian_mixture_block = GaussianMixtureLayer(self.number_of_gaussians, 
                                                           name = 'gm', 
                                                           sigma_value = gm_sigma_value,
                                                           sigma_trainable = gm_sigma_trainable,
                                                           categorical_weights = gm_categorical_weights,
                                                           categorical_trainable = gm_categorical_trainable)

        # Prior on rk
        Prk = tf.ones(number_of_gaussians) / number_of_gaussians
        self.logPrk = tf.math.log(Prk)


    @tf.function
    def call(self, x):

        # pass data through the encoder
        if self.encoder is not None:
            x = self.encoder(x)

        # get probability from prototypes
        logPz, logPz_given_rk = self.gaussian_mixture_block(x)

        # apply Bayes
        logSum_Pz_given_rk = log(1e-7+tf.reduce_sum(exp(logPz_given_rk + self.logPrk), axis=1))
        logPr_given_z = logPz_given_rk + self.logPrk - tf.reshape(logSum_Pz_given_rk, (-1,1))
        
        # compute class probability given each data point
        Pr_given_z = exp(logPr_given_z)
        Pc_given_z = tf.reduce_sum(Pr_given_z * tf.math.sigmoid(self.Pc_given_rk), axis=1)        
        
        return logPz, Pc_given_z


    def build(self, input_shape):

        self.Pc_given_rk = self.add_weight(shape=(self.number_of_gaussians,), 
                                           initializer="random_normal", 
                                           name='pc_given_rk', 
                                           trainable=self.Pc_given_rk_trainable)


        inputs = tf.keras.layers.Input(shape=input_shape[1:])
        if self.encoder_layers is not None:
            x = self.encoder(inputs)
        else:
            x = inputs
        self.gaussian_mixture_block(x)


    @tf.function                        
    def get_loss(self ,xb, yb ):
        logPz, Pc_given_z = self(xb)
        if self.encoder is not None:
            xb = self.encoder(xb)
            x0 = tf.gather(xb, tf.where(yb==0)[:,0])
            x1 = tf.gather(xb, tf.where(yb==1)[:,0])
            x0 = x0/tf.reshape(tf.linalg.norm(x0, axis=1), (-1,1)) 
            x1 = x1/tf.reshape(tf.linalg.norm(x1, axis=1), (-1,1)) 

            # compute similarities between elements intra and inter classes
            sim_inter = (tf.reduce_mean(tf.einsum('ij,kj->ik',x0,x1)) + 1)/2
            sim0 = (tf.reduce_mean(tf.einsum('ij,kj->ik',x0,x0))+ 1)/2
            sim1 = (tf.reduce_mean(tf.einsum('ij,kj->ik',x1,x1))+ 1)/2

            loss_contrastive_z = -0.5*log(sim0+1e-5) - 0.5*log(sim1+1e-5) + log(sim_inter+1e-5)
        else:
            loss_contrastive_z = 0

        loss_Pz = -tf.reduce_mean(logPz)
        loss_Pc_given_z = -tf.reduce_mean( yb*log(Pc_given_z+1e-7) + (1-yb)*log(1-Pc_given_z)+1e-7)

        loss = self.w_Pz*loss_Pz + self.w_Pc_given_z*loss_Pc_given_z + self.w_contrastive_z*loss_contrastive_z

        return {'loss_Pz': loss_Pz, 
                'loss_Pc_given_z': loss_Pc_given_z, 
                'loss_contrastive_z': loss_contrastive_z, 
                'loss_total':loss}
        
    @tf.function                        
    def step(self, xb, yb):
        with tf.GradientTape() as t:
            losses = self.get_loss(xb, yb)
                    
        grads = t.gradient(losses['loss_total'], self.trainable_variables) 
        return grads, losses
        
        
    def fit(self, train_x, train_y, w_Pz=1., w_Pc_given_z=1., w_contrastive_z=1., n_epochs=40, batch_size=20, learning_rate=0.01):

        sumw = w_Pz + w_Pc_given_z + w_contrastive_z

        self.w_Pz = w_Pz / sumw
        self.w_Pc_given_z = w_Pc_given_z / sumw
        self.w_contrastive_z = w_contrastive_z / sumw

        print (f"normalized w_Pz={self.w_Pz:.4f} w_Pc_given_z={self.w_Pc_given_z:.4f} w_contrastive_z={self.w_contrastive_z:.4f}")

        if not 'opt' in dir(self) or self.opt is None:
            print ("creating optimizer object")
            self.opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)

        losses         = defaultdict(lambda: [])
        trainable_vars = defaultdict(lambda: [])
        grads          = defaultdict(lambda: [])
        for epoch in range(n_epochs):
            t0 = time()
            for i in range(len(train_x)//batch_size):
                xb = train_x[i*batch_size:(i+1)*batch_size]
                yb = train_y[i*batch_size:(i+1)*batch_size].reshape(-1,1)
                xb = tf.convert_to_tensor(xb, tf.float32)
                yb = tf.convert_to_tensor(yb, tf.float32)
                            
                _grads, losses_batch = self.step(xb, yb)

                self.opt.apply_gradients(zip(_grads, self.trainable_variables))
                
                if np.isnan(losses_batch['loss_total']):
                    print ("stopping on nan loss", losses)
                    return

            t1 = time()
            for k,v in losses_batch.items():
                losses[k].append(np.mean(v))

            grads['mean'].append({v.name: np.abs(g.numpy()).mean() for g,v in zip(_grads, self.trainable_variables)})
            grads['std'].append({v.name: np.abs(g.numpy()).std() for g,v in zip(_grads, self.trainable_variables)})

            trainable_vars['mean'].append({i.name: i.numpy().mean() for i in self.trainable_variables})
            trainable_vars['std'].append({i.name: i.numpy().std() for i in self.trainable_variables})

            print (f'epoch {epoch+1:4d} {t1-t0:.1f}s loss {losses["loss_total"][-1]:.5f}   wsums {psums(self.trainable_variables)}   grads {psums(_grads)}', end="\r")

        losses = {k:np.r_[v] for k,v in losses.items()}
        self.losses_log = losses
        self.trainable_vars_log = trainable_vars
        self.grads_log = grads

    def sample_loss(self, n_samples, train_x, train_y, lmbda):
        loss_samples = []
        for _ in pbar(range(n_samples)):
            mu_shape = self.gaussian_mixture_block._mu.shape
            smu = np.random.random(size=mu_shape)
            smu = smu*(train_x.max(axis=0) - train_x.min(axis=0) ) + train_x.min(axis=0)
            self.gaussian_mixture_block._mu.assign(smu)
            loss_sample = self.get_loss(train_x, train_y, lmbda=lmbda)['loss_total'].numpy()
            loss_samples.append(loss_sample)
        return np.r_[loss_samples]


    def plot(self, train_x, train_y):
        for ax,i in subplots(4, usizex=6, usizey=4):
            if i==0: self.plot_model_contours( train_x, train_y, contours='Pz')
            if i==1: self.plot_model_contours( train_x, train_y, contours='Pc_given_z')
            if i==2: 
                _, Pc_given_z = self(train_x)
                Pc_given_z = Pc_given_z.numpy()
                acc0 = ((Pc_given_z>0.5)==(train_y==1)).mean()
                acc1 = ((Pc_given_z<0.5)==(train_y==0)).mean()
                self.plot_model_contours( train_x, train_y, contours='Pc_given_z/boundary')
                plt.title(f"Pc_given_z/boundary\naccuracy 0 = {acc0:.3f}   accuracy 1 = {acc1:.3f}")
            if i==3:
                logPz, Pc_given_z = self(train_x)
                plt.hist(logPz[train_y==0], bins=100, alpha=.5, label="y=0", density=True);
                plt.hist(logPz[train_y==1], bins=100, alpha=.5, label="y=1", density=True);
                plt.grid(); plt.legend();    
                plt.title("distribution of Pz");            


    def plot_model_contours(self, x, y=None, contours='Pz'):

        m = self
        assert contours in ['Pz', 'Pc_given_z', 'Pc_given_z/boundary']
        
        sx = np.random.random(size=(10000,x.shape[-1]))
        sx = sx*(x.max(axis=0) - x.min(axis=0) ) + x.min(axis=0)
            
        if m is not None:
            logPz, Pc_given_z = m(sx)
            levels = 14
            vmin=vmax=None
            if contours=='Pz':
                contour_value = logPz.numpy()
            elif contours == 'Pc_given_z':
                contour_value = Pc_given_z.numpy()
            else:
                contour_value = (Pc_given_z.numpy()>0.5).astype(float)
                vmin=0
                vmax=1
                levels=2
                
            if m is not None and 'encoder' in dir(m) and m.encoder is not None:
                sx = m.encoder(sx).numpy()
            plt.tricontour (sx[:,0], sx[:,1], contour_value, levels=levels, vmin=vmin, vmax=vmax, linewidths=0.5, colors='k')
            plt.tricontourf(sx[:,0], sx[:,1], contour_value, levels=levels, vmin=vmin, vmax=vmax, cmap="viridis")
            plt.colorbar()
            
        if m is not None and 'encoder' in dir(m) and m.encoder is not None:
            x = m.encoder(x).numpy()
            
        if y is None or len(x[y==0])==0 or len(x[y==1])==0:
            plt.scatter(x[:,0], x[:,1], color="red", s=10, alpha=.5)
        else:
            plt.scatter(x[y==0][:,0], x[y==0][:,1], color="red", s=10, alpha=.5)
            plt.scatter(x[y==1][:,0], x[y==1][:,1], color="blue", s=10, alpha=.5)

        if m is not None: 
            c = m.gaussian_mixture_block.get_weights()[0]
            plt.scatter(c[:,0], c[:,1], color='white')   
            
        plt.xlim(x[:,0].min(), x[:,0].max())
        plt.ylim(x[:,1].min(), x[:,1].max())
        plt.title(contours)


    def plot_fitlog(self):

        def plot_fitlog_pervar(vlog, title):
            for ax,tvar in subplots(self.trainable_variables):
                vname = tvar.name
                vmean = pd.DataFrame(vlog['mean'])[vname].values
                vstd = pd.DataFrame(vlog['std'])[vname].values
                plt.plot(vmean)
                plt.fill_between(range(len(vmean)), vmean-vstd/2, vmean+vstd/2, alpha=.2)
                plt.grid()
                plt.xlabel("epoch")
                plt.title(f'{title} {vname}')    
        
        plot_fitlog_pervar(self.trainable_vars_log, "trainable\n")
        plot_fitlog_pervar(self.grads_log, "grads\n")
        
        for ax,lname in subplots(list(self.losses_log.keys()), usizex=5):
            plt.plot(self.losses_log[lname])
            plt.title(lname)
            plt.grid()
            plt.xlabel("epoch")

