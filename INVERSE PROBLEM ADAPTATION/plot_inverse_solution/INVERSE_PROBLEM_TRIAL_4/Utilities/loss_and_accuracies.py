#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 21:39:11 2019

@author: hwan
"""
import tensorflow as tf
import numpy as np
import math

###############################################################################
#                               Classification                                # 
###############################################################################
def data_loss_classification(y_pred, y_true, label_dimensions,i_val):
    #y_true = tf.one_hot(tf.cast(y_true,tf.int64), label_dimensions, dtype=tf.float32)
    
    if i_val==1:
        #y_true=tf.reshape(y_true,(len(y_pred),1))
        #WW=np.ones(len(y_true))
        #mse = tf.keras.losses.MeanSquaredError()
        #for ii in range(0,len(y_true)):
        #    if np.absolute(y_true[ii]-y_pred[ii])>1:
                #WW[ii]=math.floor(np.absolute(y_true[ii]-y_pred[ii]))
         #       WW[ii]=10
        #return tf.keras.losses.MeanSquaredError(y_true, y_pred,sample_weight=[0.7])
        #return mse(y_true, y_pred, sample_weight=tf.constant(WW))
        #return mse(y_true, y_pred)
        #return tf.keras.losses.KLD(y_true, y_pred)
        #return tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred)
        #return tf.keras.losses.mean_squared_error(y_true, y_pred)
        return tf.math.reduce_mean(tf.keras.losses.MSE(y_true, y_pred))
    if i_val>1:
        return tf.keras.losses.MSE(y_true, y_pred)
        #return tf.keras.losses.mean_absolute_percentage_error(y_true, y_pred)
        #return tf.keras.losses.mean_squared_logarithmic_error(y_true, y_pred)
        





    #return tf.keras.losses.KLD(y_true, y_pred)
    #return tf.nn.softmax_cross_entropy_with_logits(y_true,y_pred)
    #return tf.keras.losses.CategoricalCrossentropy(y_true,y_pred).np()

def accuracy_classification(y_pred,y_true):
    correct = tf.math.in_top_k(tf.cast(tf.squeeze(y_true),tf.int64),tf.cast(y_pred, tf.float32),1)
    return tf.cast(correct, tf.float32)



###############################################################################
#                                 Regression                                  # 
###############################################################################
def data_loss_regression(y_pred, y_true, label_dimensions):
    #return tf.norm(tf.subtract(y_true, y_pred), 2, axis = 1)
    #y_true=tf.reshape(y_true,(len(y_pred),1))
    return tf.math.reduce_mean(tf.keras.losses.MSE(y_true, y_pred))

def relative_error(y_pred, y_true):
    return tf.norm(y_true - y_pred, 2, axis = 1)/tf.norm(y_true, 2, axis = 1)
