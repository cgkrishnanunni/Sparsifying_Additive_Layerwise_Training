#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, BatchNormalization
from tensorflow.keras.initializers import RandomNormal, RandomUniform, Constant
import numpy as np
import pandas as pd
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"
from Utilities.Net import Final_Network
from Utilities.Net_new import Final_Network_ALGO_II


# In[ ]:


def create_new(data_train, labels_train,hyperp,hyperp_new, run_options, data_input_shape, label_dimensions,i_val):
    
    y_true = tf.one_hot(tf.cast(labels_train,tf.int64), label_dimensions, dtype=tf.float32)
    
    for j in range(1,i_val+1):

        if j==1:
            data_train=data_train
            new_label=y_true

            
        if j==2:
            
    
            #Network=Final_Network( hyperp,run_options, data_input_shape, label_dimensions) 
            #NNN._set_inputs( data_train)
            #Network.load_weights("WEIGHTS"+'/'+"model_weights"+str(j-1)+'.hdf5').expect_partial()
            #NNN.save("WEIGHTS"+'/'+"model"+str(j-1))
            Network=tf.keras.models.load_model("WEIGHTS"+'/'+"model"+str(j-1))
            #Network.load_weights("WEIGHTS"+'/'+"model_weights"+str(j-1)).expect_partial()
            y_pred=Network(data_train)

            new_label=new_label-y_pred
        
            
        
        if j>2:
            
    

            Network=tf.keras.models.load_model("WEIGHTS"+'/'+"model"+str(j-1))
            #Network.load_weights("WEIGHTS"+'/'+"model_weights"+str(j-1)).expect_partial()
            y_pred=Network(data_train)
            
            new_label=new_label-y_pred

        
        
            
    return data_train,new_label,labels_train
    

