


import tensorflow as tf
import numpy as np


def error_L2(gauss_points, gauss_solution,NN,label_dimensions,hyperp,data_input_shape,i_val,run_options,gauss_weights):



        # break apart input for tracking (to compute gradient)
        x = tf.convert_to_tensor(gauss_points[:,0:1])
        y = tf.convert_to_tensor(gauss_points[:,1:2])



            # restack to feed through model
        batch = tf.stack([x[:,0], y[:,0]], axis=1)

            # make prediction
        pred = NN(batch)
           

        
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

