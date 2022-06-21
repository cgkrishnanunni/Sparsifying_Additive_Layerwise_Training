#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np
from Utilities.Net import Final_Network

import random


def compute_interior_loss(gauss_points, batch_labels_train,NN,model_constraint,label_dimensions,hyperp,hyperp_new,data_input_shape,i_val,batch_pred_test,run_options,gauss_weights,Coordinates, Stiffness, load,Solution):



            # make prediction
        pred,new = NN(Coordinates)

                
                   
            
            
            
            
            
        #pred=pred+y_pred_train_add
        
        
        
        loss=tf.matmul(Stiffness, pred)

            
            
        loss_final=tf.math.reduce_sum(tf.keras.losses.MSE(load, loss))



        return model_constraint*(loss_final)

