import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def scheduler_exp_decay(epoch, lr, start_epoch=0, decay=1, min_lr=0):
    if epoch<start_epoch:
        lr = lr
    else:
        lr = lr * tf.math.exp(-decay*0.01)
    
    return min_lr if lr<min_lr else lr

def scheduler_binary(epoch, lr, period, lr1, lr2):
    if (epoch//period)%2 == 0:
        return lr1
    else:
        return lr2
    
def scheduler_periodic(epoch, lr, period, lrset):
    lr_index = (epoch//period)%len(lrset)
    return lrset[lr_index]

def scheduler_explicit(epoch, lr, schedule):
    """
    for instance schedule =  { 1: 0.0001, 10: 0.0002, 25: 0.00001 }
    will make the lr to be 0.0001 at epoch = 1, will stay like that until
    epoch 10 when it will be set to 0.0002, etc.
    """
    if epoch in schedule.keys():
        return schedule[epoch]
    else:
        return lr

def plot_schedule(init_lr, epochs, learning_rate_scheduler, learning_rate_scheduler_kwargs):
    lr = init_lr
    lrs = [lr]
    learning_rate_scheduler_fn = eval(learning_rate_scheduler)
    for epoch in range(1, epochs):
        lr = learning_rate_scheduler_fn(
                                         epoch, lr, 
                                         **learning_rate_scheduler_kwargs
                                        )
        lrs.append(lr)
        
    plt.plot(lrs)
    plt.grid()
    plt.xlabel("epoch")
    plt.ylabel("learning rate")

