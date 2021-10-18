
"""
Created on Thu Nov 14 21:39:11 2019

@author: Krishnanunni C G
"""
import tensorflow as tf
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import random

###############################################################################
#                               Classification                                # 
###############################################################################
def manifold_classification(batch_data_train, batch_labels_train, NN,manif,label_dimensions):
    sum_val=0
    sum_new=0
    # below statement for CIFAR10
    
    #batch_labels_train=np.squeeze(batch_labels_train,axis=1) 
    for i in range(0,5):
        
        x_train_new = batch_data_train[batch_labels_train == i]
        batch_pred_train,val=NN(x_train_new)


        dimension=np.shape(val)
        

        length=len(x_train_new)
        
                
        
        
        
        new_one=val[1:length]
        
        sum_val = sum_val+manif*length*dimension[1]*tf.math.reduce_mean(tf.keras.losses.mean_squared_error(new_one, val[0:length-1]))
            
    return sum_val

