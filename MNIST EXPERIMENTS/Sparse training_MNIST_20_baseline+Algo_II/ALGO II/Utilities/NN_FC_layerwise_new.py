#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 14:29:36 2019

@author: hwan
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, BatchNormalization
from tensorflow.keras.initializers import RandomNormal, RandomUniform, Constant
import numpy as np
import pandas as pd
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

class FCLayerwise_new(tf.keras.Model):
    def __init__(self, hyperp, run_options, data_input_shape, output_dimensions, kernel_regularizer, bias_regularizer):
        super(FCLayerwise_new, self).__init__()
###############################################################################
#                  Construct Initial Neural Network Architecture               #
###############################################################################
        #=== Defining Attributes ===#
        self.data_input_shape = data_input_shape
        self.num_hidden_nodes = hyperp.num_hidden_nodes
        self.architecture = [] # storage for layer information, each entry is [filter_size, num_filters]
        self.activation = hyperp.activation
        self.regularization_value=hyperp.regularization
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.hidden_layers_list = [] # This will be a list of Keras layers

        #=== Define Initial Architecture and Create Layer Storage ===#
        self.architecture.append(self.data_input_shape[0]) # input information
        self.architecture.append(self.num_hidden_nodes) # Upsampling data
        self.architecture.append(self.num_hidden_nodes) # First hidden layer
        self.architecture.append(output_dimensions) # classification_layer
        print(self.architecture)

        #=== Weights and Biases Initializer ===#
        kernel_initializer = RandomNormal(mean=0.0, stddev=0.005)
        bias_initializer = 'zeros'
        
        #=== Linear Upsampling Layer to Map to Feature Space ===#
        l = 1
        self.upsampling_layer = Dense(units = self.architecture[l],
                                      activation = self.activation, use_bias = True,
                                      kernel_initializer = kernel_initializer, bias_initializer = bias_initializer,
                                      name = 'upsampling_layer')
        
        #=== Define Hidden Layers ===#
        l = 2
        dense_layer = Dense(units = self.architecture[l],
                           activation = self.activation, use_bias = True,
                           kernel_initializer = kernel_initializer, bias_initializer = bias_initializer,
                           name = "W" + str(l))
        self.hidden_layers_list.append(dense_layer)
        
        #=== Classification Layer ===#
        l = 3
        
        #new_kernal_regularizer=0.000
        #new_bias_regularizer=0.000
        #kernel_rr = tf.keras.regularizers.l1(new_kernal_regularizer)
        #bias_rr = tf.keras.regularizers.l1(new_bias_regularizer)
       
        self.classification_layer = Dense(units = output_dimensions,
                                          activation = hyperp.classification_act, use_bias = True,
                                          kernel_initializer = kernel_initializer, bias_initializer = bias_initializer,
                                          name = 'classification_layer')
        
###############################################################################
#                            Network Propagation                              #
############################################################################### 
    def call(self, inputs):
        #=== Upsampling ===#
        output = self.upsampling_layer(inputs)  
        for hidden_layer in self.hidden_layers_list:
            #=== Hidden Layers ===#
            prev_output = output
            output =  hidden_layer(output)  
            #output = prev_output + hidden_layer(output)
        new_output=output
        #=== Classification ===#


        output = self.classification_layer(output)
        return output,new_output
    
###############################################################################
#                                 Add Layer                                   #
###############################################################################     
    def add_layer(self, trainable_hidden_layer_index, freeze = True,  add = True):
        kernel_initializer = 'zeros'
        bias_initializer = 'zeros'
        #new_kernal_regularizer=self.regularization_value/(1+np.exp(trainable_hidden_layer_index-2))
        #new_bias_regularizer=self.regularization_value/(1+np.exp(trainable_hidden_layer_index-2))
        #new_kernal_regularizer=self.regularization_value*(1+(trainable_hidden_layer_index-2)/10)
        #new_bias_regularizer=self.regularization_value*(1+(trainable_hidden_layer_index-2)/10)
        #kernel_r = tf.keras.regularizers.l1(new_kernal_regularizer)
        #bias_r = tf.keras.regularizers.l1(new_bias_regularizer)
        
        if add:
            dense_layer = Dense(units = self.num_hidden_nodes,
                                activation = self.activation, use_bias = True,
                                kernel_initializer = kernel_initializer, bias_initializer = bias_initializer,
                                kernel_regularizer = self.kernel_regularizer, bias_regularizer = self.bias_regularizer,
                                name = "W" + str(trainable_hidden_layer_index))
            self.hidden_layers_list.append(dense_layer)
        if freeze:
            self.upsampling_layer.trainable = False
            for index in range(0, trainable_hidden_layer_index-2):
              self.hidden_layers_list[index].trainable = False
        else:
            self.upsampling_layer.trainable = True
            for index in range(0, trainable_hidden_layer_index-2):
              self.hidden_layers_list[index].trainable = True
              
###############################################################################
#                              Sparsify Weights                               #
###############################################################################            
    def sparsify_weights_and_get_relative_number_of_zeros(self, threshold = 1e-6):       
        #=== Classification Layer ===#
        class_weights = self.classification_layer.get_weights()
        sparsified_weights = self.sparsify_weights(class_weights, threshold)
        self.classification_layer.set_weights(sparsified_weights)
        
        #=== Trained Hidden Layer ===#
        trained_weights = self.hidden_layers_list[-1].get_weights()
        sparsified_weights = self.sparsify_weights(trained_weights, threshold)
        self.hidden_layers_list[-1].set_weights(sparsified_weights)
        
        #=== Compute Relative Number of Zeros ===#
        total_number_of_zeros = 0
        total_number_of_elements = 0
        for i in range(0, len(sparsified_weights)):
            total_number_of_zeros += np.count_nonzero(sparsified_weights[i]==0)
            total_number_of_elements += sparsified_weights[i].flatten().shape[0]
        relative_number_zeros = np.float64(total_number_of_zeros/total_number_of_elements)
        
        return relative_number_zeros
    
    def sparsify_weights(self, weights, threshold = 1e-6):
        sparsified_weights = []
        if isinstance(weights, float):
            if abs(weights) > threshold:
                sparsified_weights = weights
            else:
                sparsified_weights = 0
        else:
            for w in weights:
                bool_mask = (abs(w) > threshold).astype(int)
                sparsified_weights.append(w*bool_mask)
            
        return sparsified_weights
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


# In[ ]:




