import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--conf', required=True, type=str, help='the conf under lib/confs to use')
parser.add_argument('--dataset_folder', required=True, type=str, help='the folder containing a dataset')
parser.add_argument('--dataloader_class', required=True, type=str, help='a class name under dataloaders')
parser.add_argument('--learning_rate', required=False, default=None, type=float, help='the learning rate, a float')
parser.add_argument('--epochs', required=False, default=50, type=int, help='the number of epochs')
parser.add_argument('--loss', required=False, default=None, type=str, help="loss function, 'mse', or 'pxce', etc.")
parser.add_argument('--tags', required=False, default=None, type=str, help="tags for wandb (space separated)")
parser.add_argument('--batch_size', required=False, default=32, type=int, help="the batch size")
parser.add_argument('--early_stopping', action='store_true', help="if specified, training will stop if no progress on val loss")

args = parser.parse_args()

import sys
import os
import tensorflow as tf

sys.path.insert(0, "..")
os.environ['SM_FRAMEWORK']='tf.keras' 
print ("available GPUs", tf.config.list_physical_devices('GPU'))

from lib.experiments import runs
from lib.data import dataloaders
import numpy as np

# load preset models
from lib.confs import kqm as kqmconfs
from lib.confs import classic as classicconfs

conf = args.conf
dataset_folder = args.dataset_folder
dataloader_class = args.dataloader_class
learning_rate = args.learning_rate
loss             = args.loss
tags              = args.tags.split()
epochs           = args.epochs
dataloader_class = eval(f'dataloaders.{args.dataloader_class}')
batch_size       = args.batch_size
early_stopping   = args.early_stopping

wandb_project = 'qm4lp-test-experiments'
wandb_entity  = 'mindlab'

# -----------------------------------
# change these dirs to your settings
# -----------------------------------
outdir = "/opt/data/models"

model = eval(conf)
print (conf, model['model_init_args'])
if not 'metrics_args' in model.keys():
    model['metrics_args'] = {}
    
if loss is None:
    if 'loss' in model.keys():
        loss = model['loss']
        del(model['loss'])
    else:
        loss = 'mse'
else:
    if 'loss' in model.keys():
        del(model['loss'])

    

if learning_rate is None:
    if 'learning_rate' in model.keys():
        learning_rate = model['learning_rate']
        del(model['learning_rate'])
    else:
        learning_rate = 0.0001
else:
    if 'learning_rate' in model.keys():
        del(model['learning_rate'])

tagset = ['set13'] + tags

run = runs.Run(

            **model,
            dataloader_split_method = dataloader_class.split_per_partition,
            dataloader_split_args = dict (
                basedir = dataset_folder,
                partitions_id = 'communes',
                batch_size = batch_size, #32, #'per_partition:max_batch_size=32',
                cache_size = 1000,
                shuffle = True,
                max_chips = None
            ),

            class_weights=None,
            outdir = outdir,

            wandb_project = wandb_project,
            wandb_entity = wandb_entity,
            wandb_tags = [*tagset, conf],
            log_imgs = True,
            log_confusion_matrix = True,        
            loss = loss,
            learning_rate = learning_rate,

            epochs = epochs,
            early_stopping = early_stopping

              )

run.initialize()
run.model.summary()
run.run(plot_val_sample=False)
