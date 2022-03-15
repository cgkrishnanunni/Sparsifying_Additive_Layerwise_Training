#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np
from Utilities.Net import Final_Network
from Utilities.Net_new import Final_Network_ALGO_II

def error_L2(gauss_points, gauss_solution,NN,label_dimensions,hyperp,hyperp_new,data_input_shape,i_val,run_options,gauss_weights):



        # break apart input for tracking (to compute gradient)
        x = tf.convert_to_tensor(gauss_points[:,0:1])
        y = tf.convert_to_tensor(gauss_points[:,1:2])



            # restack to feed through model
        batch = tf.stack([x[:,0], y[:,0]], axis=1)

            # make prediction
        pred = NN(batch)
            
        y_pred_train_add=0*pred
        if i_val>1:
            for i_net in range(2,i_val+1):
                
                if i_net==2:    
                    Network=Final_Network(hyperp,run_options, data_input_shape, label_dimensions) 
                #NNN._set_inputs( data)
            #Network.load_weights("WEIGHTS"+'/'+"model_weights"+str(j-1)+'.hdf5').expect_partial()
                #NNN.save("WEIGHTS"+'/'+"model"+str(i_net-1))
                #Network.load_weights("WEIGHTS"+'/'+"model_weights"+str(i_net-1)+'.hdf5').expect_partial()
                #Network=tf.keras.models.load_model("WEIGHTS"+'/'+"model"+str(i_net-1))
                    Network.load_weights("WEIGHTS"+'/'+"model_weights"+str(i_net-1)).expect_partial()
                
                    y_pred_train_add=Network(batch)
        
                if i_net>2:
                
                    #Network=Final_Network_ALGO_II( hyperp_new,run_options, data_input_shape, label_dimensions) 
                #NNN._set_inputs( data)
            #Network.load_weights("WEIGHTS"+'/'+"model_weights"+str(j-1)+'.hdf5').expect_partial()
                #NNN.save("WEIGHTS"+'/'+"model"+str(i_net-1))
                    #Network.load_weights("WEIGHTS"+'/'+"model_weights"+str(i_net-1)).expect_partial()
                    Network=tf.keras.models.load_model("WEIGHTS"+'/'+"model"+str(i_net-1))
                #Network=tf.keras.models.load_model("WEIGHTS"+'/'+"model"+str(i_net-1))
                #Network.load_weights("WEIGHTS"+'/'+"model_weights"+str(i_net-1)).expect_partial()
                
                    y_pred_train_add=y_pred_train_add+Network(batch)
                
                   
            
            
            
            
            
        pred=pred+y_pred_train_add





        
        shape_solu=np.shape(pred)
        gauss_solution=tf.reshape(gauss_solution, shape_solu)

        gauss_weights = tf.cast(gauss_weights,tf.float32)

        #interior_loss = tf.reduce_mean(tf.square(pde_res))
        
        loss_old_numerator = tf.square(gauss_solution-pred)
        loss_old_denominator = tf.square(gauss_solution)
        
        shape_int=np.shape(loss_old_numerator)
        gauss_weights=tf.reshape(gauss_weights,shape_int)

        numerator_loss = tf.math.reduce_sum(tf.math.multiply(loss_old_numerator, gauss_weights))
        denominator_loss = tf.math.reduce_sum(tf.math.multiply(loss_old_denominator, gauss_weights))
        
        

        return numerator_loss/ denominator_loss

