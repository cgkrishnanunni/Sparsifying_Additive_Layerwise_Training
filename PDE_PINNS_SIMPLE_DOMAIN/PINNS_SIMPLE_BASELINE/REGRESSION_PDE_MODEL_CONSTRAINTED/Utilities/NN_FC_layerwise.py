

import tensorflow as tf
import math
from tensorflow.python.training.tracking.data_structures import NoDependency
from tensorflow.keras.layers import Dense, Conv2D, Flatten, BatchNormalization, Dropout
from tensorflow.keras.initializers import RandomNormal, RandomUniform, Constant
import numpy as np
import pandas as pd
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

class FCLayerwise(tf.keras.Model):
    def __init__(self, hyperp, run_options, data_input_shape, output_dimensions, kernel_regularizer, bias_regularizer):
        super(FCLayerwise, self).__init__()
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
        
        #self.hidden_layers_list = [] #s will be a list of Keras layers
        self.hidden_layers_list = []
        #=== Define Initial Architecture and Create Layer Storage ===#
        self.architecture.append(self.data_input_shape[0]) # input information
        self.architecture.append(self.num_hidden_nodes) # Upsampling data
        self.architecture.append(self.num_hidden_nodes) # First hidden layer
        self.architecture.append(output_dimensions) # classification_layer
        print(self.architecture)

        #=== Weights and Biases Initializer ===#
        kernel_initializer = RandomNormal(mean=0.0, stddev=0.05)
        bias_initializer = 'zeros'
        
        #=== Linear Upsampling Layer to Map to Feature Space ===#
        l = 1
        self.upsampling_layer = Dense(units = self.architecture[l],
                                      activation = 'linear', use_bias = True,
                                      kernel_initializer = kernel_initializer, bias_initializer = bias_initializer,
                                      kernel_regularizer = self.kernel_regularizer, bias_regularizer = self.bias_regularizer,
                                      name = 'upsampling_layer')
        
        
        for l in range(2,hyperp.max_hidden_layers):
            dense_layer = Dense(units = self.num_hidden_nodes,
                           activation = self.activation, use_bias = True, kernel_initializer = kernel_initializer, bias_initializer = bias_initializer, kernel_regularizer = self.kernel_regularizer, bias_regularizer = self.bias_regularizer,
                           name = "W" + str(l))
            self.hidden_layers_list.append(dense_layer)
            
            
      
        l = 3
        

        
        self.classification_layer = Dense(units = output_dimensions,
                                          activation = hyperp.classification_act, use_bias = True,                               
                                         name = 'classification_layer')
        
###############################################################################
#                            Network Propagation                              #
############################################################################### 
    def call(self, inputs):
        #=== Upsampling ===#
        output = self.upsampling_layer(inputs)  
        #output=self.drop(output)
        #output=self.batch_layer(output)
        for hidden_layer in self.hidden_layers_list:
            #=== Hidden Layers ===#
            prev_output = output
            output =  hidden_layer(output)  
            #output=self.batch_layer(output)

        #=== Classification ===#
       # output = self.classification_layer_new(output)
        #output=self.classification_layer_new(output)
        output = self.classification_layer(output)
        return output
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    