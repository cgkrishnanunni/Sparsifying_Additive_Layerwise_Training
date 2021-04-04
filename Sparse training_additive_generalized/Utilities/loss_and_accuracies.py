#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 21:39:11 2019

@author: hwan
"""
import tensorflow as tf
import numpy as np

###############################################################################
#                               Classification                                # 
###############################################################################
def data_loss_classification(y_pred, y_true, label_dimensions,i_val):
    #y_true = tf.one_hot(tf.cast(y_true,tf.int64), label_dimensions, dtype=tf.float32)
    if i_val==1:
        return tf.keras.losses.KLD(y_true, y_pred)
    if i_val>1:
        return tf.keras.losses.MSE(y_true, y_pred)
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
    return tf.norm(tf.subtract(y_true, y_pred), 2, axis = 1)

def relative_error(y_pred, y_true):
    return tf.norm(y_true - y_pred, 2, axis = 1)/tf.norm(y_true, 2, axis = 1)
