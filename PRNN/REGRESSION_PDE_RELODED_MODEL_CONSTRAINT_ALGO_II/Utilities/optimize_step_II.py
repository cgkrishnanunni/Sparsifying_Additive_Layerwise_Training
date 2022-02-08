#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 13:35:17 2019

@author: hwan
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import math
from Utilities.Net import Final_Network
from Utilities.additive_output import net_output 
from Utilities.multiplicative_output import net_output_multiply 

import shutil # for deleting directories
import os
import time

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                             Training Properties                             #
###############################################################################
def optimize_step(x_train,y_train,data_test, labels_test,i_val,label_dimensions,hyperp,hyperp_new,run_options,data_input_shape,accuracy,error_L2,gauss_points_new,gauss_solution,gauss_weights_new,data_train_length,Stiffness,model_cons):
    #=== Optimizer ===#
    dd=100
    
    for i in range(0,1):

    #=== Metrice ===#
        mean_loss_train = tf.keras.metrics.Mean()
        mean_loss_val = tf.keras.metrics.Mean()
        mean_loss_test = tf.keras.metrics.Mean()
        mean_accuracy_train = tf.keras.metrics.Mean()
        mean_accuracy_val = tf.keras.metrics.Mean()
        mean_accuracy_test = tf.keras.metrics.Mean()
    
    #=== Initialize Metric Storage Arrays ===#
        storage_array_loss = np.array([])
        storage_array_accuracy = np.array([])
        storage_array_relative_number_zeros = np.array([])
    
    #=== Creating Directory for Trained Neural Network ===#
        model = keras.Sequential()
        model.add(layers.Dense(20, activation='relu', input_shape=(2,)))
        model.add(layers.Dense(20, activation='relu'))
        model.add(layers.Dense(1, activation='linear'))
        model.summary()

        opt = keras.optimizers.Adam(learning_rate=0.001)
    
    
    
        WW=np.ones(len(y_train))
    
        for ii in range(data_train_length,len(y_train)):
            WW[ii]=Stiffness[ii-data_train_length,ii-data_train_length]*Stiffness[ii-data_train_length,ii-data_train_length]*model_cons*len(y_train)
            
     #   if np.absolute(y_train[ii])>2:
      #      WW[ii]=2
    #shape_int=np.shape(y_train)
    #WW=tf.reshape(WW,shape_int)
    

    
    
    
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_squared_error'])
    
    #model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_squared_error'])
        model.fit(x_train,y_train,batch_size=500,epochs=2000,verbose=1,sample_weight=WW)
        #model.fit(x_train,y_train,batch_size=500,epochs=100,verbose=1)
        batch_pred_test = model(data_test)
    
        y_pred_test_add=net_output(hyperp,hyperp_new,data_test, run_options, data_input_shape, label_dimensions,i_val,batch_pred_test)
        
        batch_pred_test=batch_pred_test+y_pred_test_add
            
        mean_accuracy_test(accuracy(batch_pred_test, labels_test,label_dimensions))
        
        L2_error=error_L2(gauss_points_new, gauss_solution, model,label_dimensions,hyperp,hyperp_new,data_input_shape,i_val,run_options,gauss_weights_new)
    
        print('Test Set:  Error: %.5f\n' %(mean_accuracy_test.result()))
        print('Test Set:  RelativeL2: %.5f\n' %( L2_error))
        
        batch_pred_train = model(x_train[0:data_train_length])
        mean_accuracy_train(accuracy(batch_pred_train, y_train[0:data_train_length],label_dimensions))
        
    
        if not os.path.exists("WEIGHTS"):
            os.makedirs("WEIGHTS")
        #if mean_accuracy_test.result()<dd and mean_accuracy_train.result()>1.2:   
        model.save("WEIGHTS"+'/'+"model"+str(i_val))
         #   test_L2=L2_error
          #  train_error=mean_accuracy_train.result()
           # test_normal_error=mean_accuracy_test.result()
    #print('Final Model Saved') 
            #dd=mean_accuracy_test.result()


    #print('Test Set L2: Final Error: %.5f\n' %(test_L2))
    print('Train Set:  Final Error: %.5f\n' %(mean_accuracy_train.result()))
    #print('Test Set normal:  Final Error: %.5f\n' %(test_normal_error))

