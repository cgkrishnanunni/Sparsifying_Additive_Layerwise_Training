


import tensorflow as tf
import numpy as np

import random


def compute_interior_loss(gauss_points, batch_labels_train,NN,model_constraint,label_dimensions,hyperp,hyperp_new,data_input_shape,i_val,batch_pred_test,run_options,gauss_weights,Coordinates, Stiffness, load,Solution):



        pred = NN(Coordinates)


                
                   
            
            
            

        
        
        
        loss=tf.matmul(Stiffness, pred)
 
            
            
        loss_final=tf.math.reduce_sum(tf.keras.losses.MSE(load, loss))


        return model_constraint*(loss_final)

