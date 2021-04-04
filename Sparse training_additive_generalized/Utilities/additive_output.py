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


# In[ ]:


def net_output(hyperp,hyperp_new, data, run_options, data_input_shape, label_dimensions,i_val,batch_pred_test):
    
    y_pred_test_add=0*batch_pred_test
    if i_val>1:
        for i_net in range(2,i_val+1):
                
            if i_net==2:    
                Network=Final_Network( hyperp,run_options, data_input_shape, label_dimensions) 
        
                Network.load_weights("WEIGHTS"+'/'+"model_weights"+str(i_net-1)).expect_partial()
    
                y_pred_test_add=y_pred_test_add+Network(data)
        
            if i_net>2:
                
                Network=Final_Network( hyperp_new,run_options, data_input_shape, label_dimensions) 
        
                Network.load_weights("WEIGHTS"+'/'+"model_weights"+str(i_net-1)).expect_partial()
    
                y_pred_test_add=y_pred_test_add+Network(data)
        
    return y_pred_test_add

