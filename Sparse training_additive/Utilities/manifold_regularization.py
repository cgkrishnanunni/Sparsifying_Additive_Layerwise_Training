
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
   
    for i in range(0,10):
        x_train_new = batch_data_train[batch_labels_train == i]
        batch_pred_train,val=NN(x_train_new)
        dimension=np.shape(val)
        

        length=len(x_train_new)
        #for j in range(0,length):
            #for jj in range(j,length):
                #sum=sum+manif*(np.linalg.norm(val[j]-val[jj]))**2
                #sum = sum+manif*tf.keras.losses.mean_squared_error(val[j], val[jj])
                
        #def my_function(number):
        
            #return manif*dimension[1]*tf.keras.losses.mean_squared_error(val[number], val[number+1])
    
        #num_cores = 1 #multiprocessing.cpu_count()

        #result = Parallel(n_jobs=num_cores)(delayed(my_function)(i) for i in range(length-1))
        
        #sum_val=sum_val+sum(result)

                
        #for j in range(0,length):

            #sum=sum+manif*(np.linalg.norm(val[j]-val[jj]))**2
            #if (j+1)<length:
                #sum_val = sum_val+manif*dimension[1]*tf.keras.losses.mean_squared_error(val[j], val[j+1])
                
        #sum=sum/length
                
        #for rr in range(0, dimension[1]):       
       
            #sum_new=sum_new+manif*np.nan_to_num(np.nanstd(val[:,rr], ddof=0))
        
        
        
        new_one=val[1:length]
        
        sum_val = sum_val+manif*length*dimension[1]*tf.math.reduce_mean(tf.keras.losses.mean_squared_error(new_one, val[0:length-1]))
            
    return sum_val

