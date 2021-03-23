#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, BatchNormalization
from tensorflow.keras.initializers import RandomNormal, RandomUniform, Constant
import numpy as np
import pandas as pd
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

class Final_Network(tf.keras.Model):
    def __init__(self, hyperp, run_options, data_input_shape, output_dimensions):
        super(Final_Network, self).__init__()
###############################################################################
#                  Construct Initial Neural Network Architecture               #
###############################################################################
        #=== Defining Attributes ===#
        self.data_input_shape = data_input_shape
        self.num_hidden_nodes = hyperp.num_hidden_nodes
        self.architecture = [] # storage for layer information, each entry is [filter_size, num_filters]
        self.activation = hyperp.activation
        self.regularization_value=hyperp.regularization

       

        #=== Define Initial Architecture and Create Layer Storage ===#
        self.architecture.append(self.data_input_shape[0]) # input information
        self.architecture.append(self.num_hidden_nodes) # Upsampling data
        self.architecture.append(self.num_hidden_nodes) # First hidden layer
        self.architecture.append(output_dimensions) # classification_layer
        print(self.architecture)


        
        #=== Linear Upsampling Layer to Map to Feature Space ===#
      
        
        self.upsampling_layer = Dense(units = self.architecture[1],
                                      activation = 'linear', use_bias = True, name = 'upsampling_layer')
       
        self.classification_layer = Dense(units = output_dimensions,
                                          activation = 'linear', use_bias = True, name = 'classification_layer')
        
###############################################################################
#                            Network Propagation                              #
############################################################################### 
    def call(self, inputs,hyperp):
        #=== Upsampling ===#
        output = self.upsampling_layer(inputs)  
        for i in range(2,hyperp.max_hidden_layers+1):
            #=== Hidden Layers ===#
            prev_output = output
            dense_layer = Dense(units = self.num_hidden_nodes,
                           activation = self.activation, use_bias = True,name = "W" + str(i))
            output = prev_output + dense_layer(output)  
        
        #=== Classification ===#
        output = self.classification_layer(output)
        return output
    

