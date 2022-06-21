
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
def manifold_classification(data_train_inside, labels_train_inside, NN,manif,label_dimensions):
    sum_val=0
    sum_new=0
    # Shuffling data in each epoch
    indices = tf.range(start=0, limit=tf.shape(data_train_inside)[0], dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)

    shuffled_x = tf.gather(data_train_inside, shuffled_indices)
    shuffled_y = tf.gather(labels_train_inside , shuffled_indices)
    
    # Just comparing top 500 points
    
    x=shuffled_x[0:500]
    y=shuffled_y[0:500]
    
    
    #batch_labels_train=np.squeeze(batch_labels_train,axis=1) 
    for i in range(0,1):
        
        #x_train_new = batch_data_train[batch_labels_train == i]
        batch_pred_train,val=NN(x)


        dimension=np.shape(val)
        

        length=len(x)
        
                
        
        
        
        new_one=val[1:length]
        
        sum_val = sum_val+manif*length*dimension[1]*tf.math.reduce_mean(tf.keras.losses.mean_squared_error(new_one, val[0:length-1]))
            
    return sum_val

