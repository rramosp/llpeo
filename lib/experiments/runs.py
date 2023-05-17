import tensorflow as tf
tfkl = tf.keras.layers
from keras.utils.layer_utils import count_params
from progressbar import progressbar as pbar
from datetime import datetime
from time import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import wandb
from ..data import dataloaders
from ..utils import autoinit 
from . import metrics
from . import schedulers
from . import plots
from rlxutils import subplots
import segmentation_models as sm
import seaborn as sns
import pandas as pd
import gc
import os
import psutil
from time import sleep
import inspect

# unit test metrics
from ..tests import testmetrics
testmetrics.run()

def get_next_file_path(path, base_name, ext):
    i = 1
    while True:
        file_name = f"{base_name}{i}.{ext}"
        file_path = os.path.join(path, file_name)
        if not os.path.exists(file_path):
            return file_path
        i += 1

class Run:

    def normitem(self, x,l):
        x = x[:,2:-2,2:-2,:]
        l = l[:,2:-2,2:-2]
        return x,l

    def __init__(self,
                 model_class,
                 model_init_args,
                 dataloader_split_method,
                 dataloader_split_args,
                 outdir = '/tmp',
                 learning_rate = 0.01,
                 run_id = None,
                 loss = 'mse',
                 epochs = 10,
                 wandb_project = None,
                 wandb_entity = None,
                 metrics_args = {},
                 class_weights = None,
                 n_batches_online_val = np.inf,
                 n_val_samples = 10,
                 log_imgs = False,
                 log_perclass = False,
                 log_confusion_matrix = False,
                 learning_rate_scheduler = None,
                 learning_rate_scheduler_kwargs = {},
                 measure_mae_on_segmentation = True,
                 early_stopping = False,
                 wandb_tags = []
                ):
        autoinit.autoinit(self)
        self.loss_name = loss
        print (f"using model {model_class.__name__}")

        assert 'partitions_id' in self.dataloader_split_args.keys(), \
               "'dataloader_split_args' must have 'partitions_id'"

        # usually a function name
        if self.learning_rate_scheduler is not None:
            self.learning_rate_scheduler_fn = eval(self.learning_rate_scheduler)
        else:
            self.learning_rate_scheduler_fn = None

        self.partitions_id = dataloader_split_args['partitions_id']
        print (f"using partitions {self.partitions_id}")
        print (f"using loss {self.loss_name}")
        if self.losses_supported()!='all' and not self.loss_name in self.losses_supported():
            raise ValueError(f"loss '{self.loss_name}' not supported in this model, only {self.losses_supported()} are supported")


    def initialize(self):
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        if self.class_weights is not None:
            if 'class_groups' in self.dataloader_split_args.keys() and self.dataloader_split_args['class_groups'] is not None:
                raise ValueError("cannot specify both 'dataloader.class_groups' and 'run.class_weights'")
            
            # normalize class weights
            self.class_weights = {k:v/sum(self.class_weights.values()) for k,v in self.class_weights.items()}
            self.dataloader_split_args['class_groups'] = [i for i in self.class_weights.keys()]

        self.tr, self.ts, self.val = self.dataloader_split_method(**self.dataloader_split_args)

        # if there are no classweights, try first to get default ones from the dataloader
        if self.class_weights is None:        
            self.class_weights = self.tr.get_default_class_weights()
            self.tr.set_class_groups()
            self.ts.set_class_groups()
            self.val.set_class_groups()
            print ("got default class weights from dataloader:", self.class_weights)

        # build the model with the known number of classes
        self.number_of_classes = self.tr.number_of_output_classes

        if not 'number_of_classes' in inspect.signature(self.model_class.__init__).parameters:
            raise ValueError(f"the constructor of {self.model_class.__name__} must accept a parameter 'number_of_classes'")

        self.model_init_args['number_of_classes'] = self.number_of_classes
        self.model = self.model_class(**self.model_init_args)

        # check model produces segmentations if required
        if self.measure_mae_on_segmentation and not self.model.produces_segmentation_probabilities():
            raise ValueError(f"you are requiring 'measure_mae_on_segmentation' but {self.model_class.__name__} does not produce segmentations")

        # feed some data through the model to force creating weight structures
        x, (p,l) = self.tr[0]
        x,l = self.normitem(x,l)
        self.input_shape = x.shape[1:]
        print ("setting input shape to", self.input_shape)
        self.model(x)       

        # if still None, distribute class weights evently across all classes
        if self.class_weights is None:
            n = self.number_of_classes
            self.class_weights = {i: 1/n for i in range(n)}
            print ("seting equal class weights for all classes", self.class_weights)

        self.class_weights_values = list(self.class_weights.values())
               
        # if no zero in class weights set its weight to zero
        if not 0 in self.class_weights.keys() and sum([0 in i for i in self.class_weights.keys() if isinstance(i,tuple)])==0:
            self.class_weights_values = [0] + self.class_weights_values    

        # set various stuff
        self.run_name = f"{self.get_name()}-{self.partitions_id}-{self.loss_name}-{datetime.now().strftime('%Y%m%d[%H%M]')}"
        self.opt = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)
        self.metrics = metrics.ProportionsMetrics(class_weights_values = self.class_weights_values, **self.metrics_args)
        self.trainable_params = count_params(self.model.trainable_weights)
        self.non_trainable_params = count_params(self.model.non_trainable_weights)

        # give model a chance to initalize stuff based on the run
        self.model.init_model_params(self)
        self.init_args['class_weights'] = self.class_weights
        if self.run_id is None:
            self.init_model_params()
            if self.wandb_project is not None:
                wconfig = autoinit.get_autoinit_wandb(self)
                
                wandb.init(project=self.wandb_project, entity=self.wandb_entity, 
                            name=self.run_name, config=wconfig)
                self.run_id = wandb.run.id
            else:
                run_file_path = get_next_file_path(self.outdir, "run","h5")
                self.run_id = os.path.basename(run_file_path)[:-3]
        else:
            run_file_path = os.path.join(self.outdir, self.run_id + ".h5")
            if os.path.exists(run_file_path):
                print ("restoring model weights from file")        
                self.model.load_weights(run_file_path)
            else:
                print ("no model weights file found")
        self.init_args['run_id'] = self.run_id

        self.wandb_tags.append(f"{self.number_of_classes} classes")
        return self

    def get_additional_wandb_config(self):
        r = dict(trainable_params = self.trainable_params,
                 non_trainable_params = self.non_trainable_params,
                 input_shape = self.input_shape)

        r.update(self.model.get_additional_wandb_config())
        return r
    
    def init_model_params(self):
        '''
        Perform a model parameter initialization if needed
        '''
        pass

    def empty_caches(self):
        self.tr.empty_cache()
        self.val.empty_cache()
        self.ts.empty_cache()
        gc.collect()
    
    def get_loss_components(self, p, out):
        return None

    def get_val_sample(self, n=10, shuffle=False):
        
        batch_size_backup = self.val.batch_size
        shuffle_backup = self.val.shuffle
        self.val.rebatch(n, shuffle=shuffle)
        
        
        x, (p, l) = self.val[0]
        x,l = self.normitem(x,l)
        out = self.model(x)
        if self.model.produces_segmentation_probabilities():
            out_segmentation = self.model.predict_segmentation(x).numpy()
        else:
            out_segmentation = None

        if isinstance(out, list):
            out = [ii.numpy() for ii in out]
        else:
            out = out.numpy()
        
        # restore val dataset config
        self.val.rebatch(batch_size_backup, shuffle=shuffle_backup)
        return x, p, l, out, out_segmentation



    def plot_val_sample(self, n=10, return_fig = False, shuffle=False):
        
        shuffle_backup = self.val.shuffle
        self.val.shuffle = shuffle
        val_x, val_p, val_l, val_out, val_out_segmentation = self.get_val_sample(n=n, shuffle=shuffle)
        self.val.shuffle = shuffle_backup
        
        n = len(val_x)

        n_rows = 4 if self.model.produces_segmentation_probabilities() else 3
        for ax, ti in subplots(range(n*n_rows), n_cols=n, usizex=3.5):
            i = ti % n
            row = ti // n
            
            if row==0:
                plots.plot_img(val_x[i], title=self.val.batch_chips_basedirs[i], ylabel='input rgb' if i==0 else None)

            if row==1:
                plots.plot_label(self.number_of_classes, val_l[i], ylabel='labels' if i==0 else None, add_color_bar=True)
                    
            if row==2 and self.model.produces_segmentation_probabilities():
                plots.plot_segmentation_probabilities(self, val_l[i], val_out[i], val_out_segmentation[i], 
                                                     ylabel='thresholded output' if i==0 else None)
                  
            if (row==2 and not self.model.produces_segmentation_probabilities()) or row==3:
                plots.plot_proportions_prediction(self, val_p[i], val_l[i], val_out[i], show_legend= i in [0, n//2, n-1])                
                
        if return_fig:
            fig = plt.gcf()
            plt.close(fig)
            return fig
        else:
            return val_x, val_p, val_l, val_out
    
    def get_name(self):
        r = self.model.get_name()
        if 'tr' in dir(self):
            r = self.tr.basedir.split("/")[-2][:4]+"_"+r
        return r

    @tf.function
    def get_loss(self, out, p, l): 

        if self.loss_name in ['multiclass_proportions_mse', 'mse']:
            return self.metrics.multiclass_proportions_mse(p, out)
        
        if self.loss_name in ['multiclass_proportions_rmse', 'rmse']:
            return self.metrics.multiclass_proportions_rmse(p, out)
        
        if self.loss_name in ['proportions_cross_entropy', 'ppce']:
            return self.metrics.proportions_cross_entropy(p, out)

        if self.loss_name in ['pixel_level_cross_entropy', 'pxce']:
            return self.metrics.pixel_level_categorical_cross_entropy(l, out)
        
        if self.loss_name in ['kldiv']:
            return self.metrics.kldiv(p, out)
        
        if self.loss_name in['ilogkldiv']:
            return self.metrics.ilogkldiv(p, out)

        if self.loss_name in ['multiclass_LSRN_loss', 'lsrn']:
            return self.metrics.multiclass_LSRN_loss(p, out)
        
        if self.loss_name == 'custom':
            return self.model.custom_loss(p, out)
        
        raise ValueError(f"unkown loss '{self.loss_name}'")

    @tf.function
    def predict(self, x):
        out = self.model(x)
        return out

    def losses_supported(self):
        return 'all'

    def get_trainable_variables(self):
        return self.model.trainable_variables

    #@tf.function(input_signature=(tf.TensorSpec(shape=[None,96,96,3], dtype=tf.float32),
    #                              tf.TensorSpec(shape=[None,5], dtype=tf.float32),
    #                              tf.TensorSpec(shape=[None,96,96], dtype=tf.int16)))
    @tf.function
    def aply_model_and_get_loss(self, x, p, l):
        out = self.predict(x)
        loss = self.get_loss(out,p,l)
        return loss, out

    def apply_model_and_compute_gradients(self, x,p,l):
        with tf.GradientTape() as t:
            loss, out = self.aply_model_and_get_loss(x, p, l)
        grads = t.gradient(loss, self.get_trainable_variables())
        self.opt.apply_gradients(zip(grads, self.get_trainable_variables()))     
        return loss, out  

    def fit(self):
        run_file_path = os.path.join(self.outdir, self.run_id + ".h5")            
        tr_loss = 0
        min_val_loss = np.inf
        lr = self.learning_rate
        val_loss_history = []
        self.fit_seconds_history = []
        self.inference_seconds_history = []
        for epoch in range(self.epochs):
            print (f"\nepoch {epoch+1}", flush=True)

            if epoch>0 and self.learning_rate_scheduler_fn is not None:
                # apply learning rate schedule 
                lr = self.learning_rate_scheduler_fn(
                                                     epoch, lr, 
                                                     **self.learning_rate_scheduler_kwargs
                                                    )

                self.opt.learning_rate.assign(lr)

            # the training loop
            losses = []
            fit_seconds = 0
            for x,(p,l) in pbar(self.tr):
                if len(x)==0:
                    continue
                # trim to unet input shape
                x,l = self.normitem(x,l)
                # compute loss
                start_time = time()
                loss, out = self.apply_model_and_compute_gradients(x,p,l)
                fit_seconds += time()-start_time
                losses.append(loss)

            log_dict = {}
            tr_loss = np.mean(losses)
            log_dict['train/loss'] = tr_loss
            log_dict['train/seconds'] = fit_seconds

            self.fit_seconds_history.append(fit_seconds)

            # measure stuff on validation for reporting
            losses, ious, maeps, maeps_perclass, rmaeps, rmaeps_perclass = [], [], [], [], [], []
            losses_components = {}
            max_value = np.min([len(self.val),self.n_batches_online_val ]).astype(int)
            self.pixel_classification_metrics = metrics.PixelClassificationMetrics(number_of_classes=self.number_of_classes)
            self.pixel_classification_metrics.reset_state()
            inference_seconds = 0
            for i, (x, (p,l)) in pbar(enumerate(self.val), max_value=max_value):
                if i>=self.n_batches_online_val:
                    break

                # predict
                x,l = self.normitem(x,l)
                start_time = time()
                out = self.model(x)
                inference_seconds += time()-start_time
                self.model.save_last_input_output_pair(x, out)

                # compute losses
                loss = self.get_loss(out,p,l).numpy()
                loss_components = self.get_loss_components(p, out)
                if loss_components is not None:
                    for k,v in loss_components.items():
                        if not k in losses_components.keys():
                            losses_components[k] = []
                        losses_components[k].append(v)

                losses.append(loss)

                # get segmentation output if model is able to generate it
                if self.model.produces_segmentation_probabilities():
                    out_segmentation = self.model.predict_segmentation(x)

                # measure mae
                if self.measure_mae_on_segmentation and self.model.produces_segmentation_probabilities():
                    maeps.append(self.metrics.multiclass_proportions_mae_on_chip(l, out_segmentation))
                    maeps_perclass.append(self.metrics.multiclass_proportions_mae_on_chip(l, out_segmentation, perclass=True).numpy())
                    rmaeps.append(self.metrics.multiclass_proportions_rmae_on_chip(l, out_segmentation))
                    rmaeps_perclass.append(self.metrics.multiclass_proportions_rmae_on_chip(l, out_segmentation, perclass=True).numpy())
                elif self.model.produces_label_proportions():
                    maeps.append(self.metrics.multiclass_proportions_mae_on_chip(l, out))
                    maeps_perclass.append(self.metrics.multiclass_proportions_mae_on_chip(l, out, perclass=True).numpy())
                    rmaeps.append(self.metrics.multiclass_proportions_rmae_on_chip(l, out))
                    rmaeps_perclass.append(self.metrics.multiclass_proportions_rmae_on_chip(l, out, perclass=True).numpy())

                # measure pixel based stuff
                if self.model.produces_segmentation_probabilities():
                    ious.append(self.metrics.compute_iou(l,out_segmentation))
                    self.pixel_classification_metrics.update_state(l,out_segmentation)

            self.inference_seconds_history.append(inference_seconds)

            # summarize validation stuff
            txt_metrics = ""
            val_loss = np.mean(losses)
            val_loss_components = {k: np.mean(v) for k,v in losses_components.items()}
            if self.model.produces_label_proportions():
                val_mean_mae = np.mean(maeps)
                val_mean_rmae = np.mean(rmaeps)
                txt_metrics += f"mae {val_mean_mae:.5f}"
            if self.model.produces_segmentation_probabilities():
                val_mean_f1 = np.mean(self.pixel_classification_metrics.result('f1', 'micro'))
                val_mean_iou = np.mean(ious)
                txt_metrics += f" f1 {val_mean_f1:.5f} iou {val_mean_iou:.5f}"
                
            # assemble per class metrics
            r = {'loss': np.mean(losses)}
            if self.model.produces_label_proportions():
                r.update({'maeprops_on_chip::global': np.mean(maeps)})
                r.update({f'maeprops_on_chip::class_{k}':v for k,v in zip(range(0, self.number_of_classes), np.r_[maeps_perclass].mean(axis=0))})
                r.update({'rmaeprops_on_chip::global': np.mean(rmaeps)})
                r.update({f'rmaeprops_on_chip::class_{k}':v for k,v in zip(range(0, self.number_of_classes), np.r_[rmaeps_perclass].mean(axis=0))})

            if self.model.produces_segmentation_probabilities(): 
                r['f1::global']  = self.pixel_classification_metrics.result('f1', 'micro').numpy()
                r['iou::global'] = np.mean(ious)
                r.update({f'f1::class_{k}':tf.constant(v).numpy() for k,v in self.pixel_classification_metrics.result('f1', 'per_class').items()})
                r.update({f'iou::class_{k}':tf.constant(v).numpy() for k,v in self.pixel_classification_metrics.result('iou', 'per_class').items()})

            df_perclass = pd.DataFrame([{k:v for k,v in r.items() if 'global' not in k and k!='loss'}], index=["val"]).T

            # save model and log images if better val loss
            if val_loss < min_val_loss:
                print ("new min val loss")
                min_val_loss = val_loss
                print (f"saving model weights into {run_file_path}")
                self.model.save_weights(run_file_path)

                if self.wandb_project is not None:
                    log_dict['val/min_loss'] = val_loss
                    if self.model.produces_segmentation_probabilities():
                        if self.log_imgs:
                            log_dict['val/sample'] = self.plot_val_sample(self.n_val_samples, return_fig=True)
                        if self.log_confusion_matrix:
                            img = metrics.plot_confusion_matrix(self.pixel_classification_metrics.cm)
                            log_dict['val/confusion_matrix'] = wandb.Image(img, caption="confusion matrix")
                        if self.log_perclass:
                            log_dict['val/perclass'] = \
                                wandb.Table(columns = ['metric', 'val'], 
                                            data=[[i,j[0]] for i,j in zip (df_perclass.index, df_perclass.values)])    

            # log to wandb
            if self.wandb_project is not None:
                log_dict["val/loss"] = val_loss
                log_dict["val/seconds"] = inference_seconds

                for k,v in val_loss_components:
                    log_dict[f'val/{k}'] = v

                if self.model.produces_segmentation_probabilities():
                    log_dict["val/f1"] = val_mean_f1
                    log_dict["val/iou"] = val_mean_iou
                if self.model.produces_label_proportions():
                    log_dict["val/maeprops_on_chip"] = val_mean_mae
                    log_dict["val/rmaeprops_on_chip"] = val_mean_rmae
                if self.learning_rate_scheduler_fn is not None:
                    log_dict['train/learning_rate'] = lr
         
                wandb.log(log_dict)
                
            if self.learning_rate_scheduler_fn is not None:
                txt_metrics += f" lr {lr:.7f}"
            # log to screen
            print (f"epoch {epoch+1:3d}, train loss {tr_loss:.5f} fit seconds {fit_seconds:.3f}", flush=True)
            print (f"epoch {epoch+1:3d},   val loss {val_loss:.5f} {txt_metrics} {val_loss_components} inference seconds {inference_seconds:.3f}", flush=True)

            if self.early_stopping and val_loss >= 0.999*np.mean(val_loss_history[-10:]):
                print ("early stopping")
                break

            val_loss_history.append(val_loss)

    def run(self, plot_val_sample=True):
        """
        runs the experiment, calling fit, logging to wandbd, summarizing metrics, etc.
        """
        try:
            print ("\nusing partition", self.partitions_id)
            print ("using loss", self.loss) 
            
            if self.wandb_project is not None:
                wandb.run.tags = wandb.run.tags + tuple(self.wandb_tags)
            print ("-----", psutil.virtual_memory())
            self.save()
            print ("run configuration saved")

            interrupted = False    
            try:
                try:
                    self.fit()
                except RuntimeWarning:
                    raise ValueError('runtime warning')
            except KeyboardInterrupt:
                print (f"-----------------------------------------------------------------------------")
                print (f"keyboard interrupt. saving summary. access this run object in '{self.__class__.__name__}.__saved_run__'")
                print (f"please wait for summary, or ctrl-c again to interrupt")
                print (f"-----------------------------------------------------------------------------")
                self.__class__.__saved_run__ = self
                interrupted = True

            try:
                if self.model.produces_label_proportions() and plot_val_sample:
                    self.plot_val_sample(10); plt.show()
                r = self.summary_result()
                csv_path = os.path.join(self.outdir, self.run_id + '.csv')
                r.to_csv(csv_path)
                if self.wandb_project is not None:
                    config = {}
                    for part in ['train', 'val', 'test']:
                        config[f"mae::{part}"] = r[part]['maeprops_on_chip global']
                        config[f"rmae::{part}"] = r[part]['rmaeprops_on_chip global']
                        if self.model.produces_segmentation_probabilities():
                            config[f"f1::{part}"] = r[part]['f1 global']
                            config[f"iou::{part}"] = r[part]['iou global']


                    config[f"fit_seconds.mean"] = np.mean(self.fit_seconds_history[2:])
                    config[f"fit_seconds.std"] = np.std(self.fit_seconds_history[2:])

                    config[f"inference_seconds.mean"] = np.mean(self.inference_seconds_history[2:])
                    config[f"inference_seconds.std"] = np.std(self.inference_seconds_history[2:])

                    wandb.config.update(config)

                self.empty_caches()
                
                print ("--------")
                print (r[[i=='loss' or 'global' in i for i in r.index]])
                print ("--------")
                print (r[[i!='loss' and 'global' not in i for i in r.index]])
                print ("--------")
                print (f"fit seconds:       mean {np.mean(self.fit_seconds_history):.3f}, std {np.std(self.fit_seconds_history):.5f}")
                print (f"inference seconds: mean {np.mean(self.inference_seconds_history):.3f}, std {np.std(self.inference_seconds_history):.5f}")
                wandb.finish(quiet=True)

            except KeyboardInterrupt:
                interrupted = True

            if interrupted:
                wandb.finish()
                print ("-----------------------------------------------------------------------")
                print ("waiting 10secs. ctrl-c again if you want to completely interrupt")
                print ("-----------------------------------------------------------------------")
                sleep(10)
                print ("done")

        except Exception as e:
            print ("intercepting exception to finish wandb", flush=True)
            wandb.finish()
            raise e


    def summary_dataset(self, dataset_name):
        assert dataset_name in ['train', 'val', 'test']

        if dataset_name == 'train':
            dataset = self.tr
        elif dataset_name == 'val':
            dataset = self.val
        else:
            dataset = self.ts

        losses, maeps, rmaeps, ious = [], [], [], []
        mae_perclass, rmae_perclass = [], []
        self.summary_classification_metrics = metrics.PixelClassificationMetrics(number_of_classes=self.number_of_classes)
        self.summary_classification_metrics.reset_state()
        for x, (p,l) in pbar(dataset):
            if len(x)==0:
                continue            
            x,l = self.normitem(x,l)
            out = self.model(x)
            self.model.save_last_input_output_pair(x, out)
            if self.model.produces_segmentation_probabilities():
                out_segmentation = self.model.predict_segmentation(x)
                iou = self.metrics.compute_iou(l, out_segmentation)
                ious.append(iou)
                self.summary_classification_metrics.update_state(l, out_segmentation)
                
            losses.append(self.get_loss(out,p,l).numpy())
            
            # measure mae
            if self.measure_mae_on_segmentation and self.model.produces_segmentation_probabilities():
                maeps.append(self.metrics.multiclass_proportions_mae_on_chip(l, out_segmentation))
                mae_perclass.append(self.metrics.multiclass_proportions_mae_on_chip(l, out_segmentation, perclass=True).numpy())
                rmaeps.append(self.metrics.multiclass_proportions_rmae_on_chip(l, out_segmentation))
                rmae_perclass.append(self.metrics.multiclass_proportions_rmae_on_chip(l, out_segmentation, perclass=True).numpy())
            elif self.model.produces_label_proportions():
                maeps.append(self.metrics.multiclass_proportions_mae_on_chip(l, out))
                mae_perclass.append(self.metrics.multiclass_proportions_mae_on_chip(l, out, perclass=True).numpy())
                rmaeps.append(self.metrics.multiclass_proportions_rmae_on_chip(l, out))
                mae_perclass.append(self.metrics.multiclass_proportions_rmae_on_chip(l, out, perclass=True).numpy())
                                    
        r = {'loss': np.mean(losses) }
        if self.model.produces_label_proportions():
            r.update({'maeprops_on_chip::global': np.mean(maeps)})
            r.update({f'maeprops_on_chip::class_{k}':v for k,v in zip(range(0, self.number_of_classes), np.r_[mae_perclass].mean(axis=0))})
            r.update({'rmaeprops_on_chip::global': np.mean(rmaeps)})
            r.update({f'rmaeprops_on_chip::class_{k}':v for k,v in zip(range(0, self.number_of_classes), np.r_[rmae_perclass].mean(axis=0))})

        if self.model.produces_segmentation_probabilities(): 
            r['f1::global']  = self.summary_classification_metrics.result('f1', 'micro').numpy()
            r['iou::global'] = np.mean(ious)
            r.update({f'f1::class_{k}':tf.constant(v).numpy() for k,v in self.summary_classification_metrics.result('f1', 'per_class').items()})
            r.update({f'iou::class_{k}':tf.constant(v).numpy() for k,v in self.summary_classification_metrics.result('iou', 'per_class').items()})

        return r
    
    def summary_result(self):
        """
        runs summary_dataset over train, val and test
        returns a dataframe with dataset rows and loss/metric columns
        """
        run_file_path = os.path.join(self.outdir, self.run_id + ".h5")            
        self.model.load_weights(run_file_path)
        r = [self.summary_dataset(i) for i in ['train', 'val', 'test']]
        r = pd.DataFrame(r, index = ['train', 'val', 'test'])

        r = r[['loss'] + sorted([c for c in r.columns if c!='loss'])]
        r.columns = [" ".join(c.split("::")) for c in r.columns]      
        r = r.T
        r.index.name = 'metric'
        return r

    def save(self):
        params_path = os.path.join(self.outdir, self.run_id + ".params")
        autoinit.save_autoinit_spec(self, params_path)

    @classmethod
    def load_from(cls, filepath):
        r = autoinit.load_from_autoinit_spec(filepath)
        return r