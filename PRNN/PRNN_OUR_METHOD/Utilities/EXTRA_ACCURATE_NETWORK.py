#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, BatchNormalization,MaxPool2D, Dropout, AveragePooling2D
from tensorflow.keras.initializers import RandomNormal, RandomUniform, Constant
import numpy as np
import pandas as pd
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

class FCLayerwise_extra_accurate(tf.keras.Model):
    def __init__(self, hyperp, run_options, data_input_shape, output_dimensions,trainable_hidden_layer_index):
        super(FCLayerwise_extra_accurate, self).__init__()
###############################################################################
#                  Construct Initial Neural Network Architecture               #
###############################################################################
###############################################################################
        #=== Defining Attributes ===#
        self.data_input_shape = data_input_shape
        self.num_hidden_nodes = hyperp.num_hidden_nodes
        self.architecture = [] # storage for layer information, each entry is [filter_size, num_filters]
        self.activation = hyperp.activation
        self.hidden_layers_list = [] # This will be a list of Keras layers

        #=== Define Initial Architecture and Create Layer Storage ===#
        self.architecture.append(self.data_input_shape[0]) # input information
        self.architecture.append(self.num_hidden_nodes) # Upsampling data
        self.architecture.append(self.num_hidden_nodes) # First hidden layer
        self.architecture.append(output_dimensions) # classification_layer
        


        self.hidden_layers_list = [] # This will be a list of Keras layers

        #=== Define Initial Architecture and Create Layer Storage ===#
        self.architecture.append([self.data_input_shape[0],num_channels]) # input information
        self.architecture.append([3,self.num_filters]) # Upsampling data
        self.architecture.append([self.kernel_size,self.num_filters]) # First hidden layer
       # self.architecture.append([3, num_channels]) # 3x3 convolutional layer for downsampling features
        self.architecture.append(output_dimensions) # classification_layer
        print(self.architecture)

        
        #=== Linear Upsampling Layer to Map to Feature Space ===#
        l = 1

        self.upsampling_layer = Dense(units = self.architecture[l],
                                      activation = 'linear', use_bias = True,
                                      name = 'upsampling_layer')


        


        
        #=== Define Hidden Layers ===#
        for l in range(2,hyperp.max_hidden_layers):
            
        dense_layer = Dense(units = self.architecture[l],
                           activation = self.activation, use_bias = True,
                           name = "W" + str(l))
        self.hidden_layers_list.append(dense_layer)
        
        #=== Classification Layer ===#
        l = 3

       
        self.classification_layer = Dense(units = output_dimensions,
                                          activation = hyperp.classification_act, use_bias = True,
                                          name = 'classification_layer')
        
        self.upsampling_layer.trainable = False
        for index in range(0, trainable_hidden_layer_index-2):
            self.hidden_layers_list[index].trainable = False
###############################################################################
#                            Network Propagation                              #
############################################################################### 
    def call(self, inputs):
        #=== Upsampling ===#
        #=== Upsampling ===#
        output = self.upsampling_layer(inputs)  
        for hidden_layer in self.hidden_layers_list:
            #=== Hidden Layers ===#
            prev_output = output
            output = prev_output + hidden_layer(output)  
        new_output=output
        #=== Classification ===#
        output = self.classification_layer(output)
        return output,output
