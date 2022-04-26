#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np

def compute_interior_loss_weak(batch_data_train, batch_labels_train, NN,model_constraint,label_dimensions,y_pred_train_add):
        '''
        compute the weak form loss - problem specific
        '''
        forcing=tf.constant([200.0])
        # break apart input for tracking (to compute gradient)
        x = tf.convert_to_tensor(batch_data_train[:,0:1])
        y = tf.convert_to_tensor(batch_data_train[:,1:2])

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)

            # restack to feed through model
            batch = tf.stack([x[:,0], y[:,0]], axis=1)

            # make prediction
            pred,new = NN(batch)
            pred=pred+y_pred_train_add

            # compute gradients wrt input
        dx = tape.gradient(pred, x)
        dy = tape.gradient(pred, y)

        grad_u = tf.stack([dx[:,0], dy[:,0]], axis=1)
        del tape
        
        half = tf.constant([0.5], dtype=tf.float32)
        weak_form = half * tf.reduce_sum(tf.square(grad_u), axis=1)
        weak_form = weak_form - forcing * pred


        interior_loss = tf.reduce_mean(weak_form)
        return model_constraint*interior_loss

