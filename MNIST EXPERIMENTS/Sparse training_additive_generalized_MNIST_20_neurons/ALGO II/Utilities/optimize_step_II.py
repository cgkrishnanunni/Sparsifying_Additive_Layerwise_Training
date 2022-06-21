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
def optimize_step(x_train,y_train,data_test, labels_test,i_val,label_dimensions,hyperp,hyperp_new,run_options,data_input_shape,accuracy):
    #=== Optimizer ===#

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
    model.add(layers.Dense(20, activation='elu', input_shape=(784,)))
    model.add(layers.Dense(20, activation='elu'))
    model.add(layers.Dense(10, activation='linear'))
    model.summary()

    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_squared_error'])
    model.fit(x_train,y_train,batch_size=1000,epochs=50,verbose=1)
    
    batch_pred_test = model(data_test)
    
    y_pred_test_add=net_output(hyperp,hyperp_new,data_test, run_options, data_input_shape, label_dimensions,i_val,batch_pred_test)
        
    batch_pred_test=batch_pred_test+y_pred_test_add
            
    mean_accuracy_test(accuracy(batch_pred_test, labels_test))
    

    
    print('Test Set:  Error: %.3f\n' %( mean_accuracy_test.result()))

    
    if not os.path.exists("WEIGHTS"):
        os.makedirs("WEIGHTS")
    model.save("WEIGHTS"+'/'+"model"+str(i_val))
    #print('Final Model Saved') 
        

    

