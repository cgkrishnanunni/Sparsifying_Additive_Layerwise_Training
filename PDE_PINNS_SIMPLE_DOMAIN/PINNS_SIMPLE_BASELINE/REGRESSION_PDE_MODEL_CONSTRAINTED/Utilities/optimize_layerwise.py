


import tensorflow as tf
import numpy as np
import pandas as pd
import math




import shutil 
import os
import time

import pdb 

###############################################################################
#                             Training Properties                             #
###############################################################################
def optimize(hyperp, hyperp_new,run_options, file_paths, NN, data_loss, accuracy, data_and_labels_train, data_and_labels_val, data_and_labels_test, label_dimensions, num_batches_train,data_and_labels_train_new,batch_size,random_seed,num_data_train,i_val,data_input_shape,data_train,labels_train,trainable_hidden_layer_index,compute_interior_loss,error_L2,gauss_solution,gauss_points_new,gauss_weights_new,Coordinates, Stiffness, load,Solution):
    
    

    #=== Optimizer ===#
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    reset_optimizer = tf.group([v.initializer for v in optimizer.variables()])
    
    #=== Metrice ===#
    mean_loss_train = tf.keras.metrics.Mean()
    mean_loss_train_constraint = tf.keras.metrics.Mean()
    mean_loss_val = tf.keras.metrics.Mean()
    mean_loss_test = tf.keras.metrics.Mean()
    mean_accuracy_train = tf.keras.metrics.Mean()
    mean_accuracy_val = tf.keras.metrics.Mean()
    mean_accuracy_test = tf.keras.metrics.Mean()
    
    #=== Initialize Metric Storage Arrays ===#
    storage_array_loss = np.array([])
    storage_array_accuracy = np.array([])
    storage_array_relative_number_zeros = np.array([])
    
    #=== Creating Directory for Trained Neural Network ===#
    if not os.path.exists(file_paths.NN_savefile_directory):
        os.makedirs(file_paths.NN_savefile_directory)
    


###############################################################################
#                             Train Neural Network                            #
############################################################################### 
    loss_validation = 1e5
    #trainable_hidden_layer_index = 2
    relative_number_zeros = 0
    
    model_constraint=hyperp.model_constraint
    max_hidden_layers=trainable_hidden_layer_index+1
    #####################################
    #   Training Current Architecture   #
    #####################################
    while trainable_hidden_layer_index < max_hidden_layers: 
        


        
  

        
        #=== Begin Training ===#
        print('Beginning Training')
        fr=0
        for epoch in range(hyperp_new.num_epochs):
            
            if epoch<=2000:

                lrate = 0.001 
            
            if epoch>2000:

                lrate = 0.0001

            optimizer = tf.keras.optimizers.Adam(learning_rate=lrate)

            
            ff=data_and_labels_train_new.shuffle(num_data_train,seed=random_seed)
            
            data_and_labels_train_new_new = ff.batch(batch_size)
            
            print('================================')
            print('            Epoch %d            ' %(epoch))
            print('================================')
            print(file_paths.filename)
            print('Trainable Hidden Layer Index: %d' %(trainable_hidden_layer_index))
            print('GPU: ' + run_options.which_gpu + '\n')
            print('Optimizing %d batches of size %d:' %(num_batches_train, hyperp_new.batch_size))
            start_time_epoch = time.time()
            for batch_num, (batch_data_train, batch_labels_train,labels) in data_and_labels_train_new_new.enumerate():
                with tf.GradientTape() as tape:
                    start_time_batch = time.time()
                    batch_pred_train = NN(batch_data_train)
                    #=== Display Model Summary ===#
                    if batch_num == 0 and epoch == 0:
                        NN.summary()

                    batch_loss_train = data_loss(batch_pred_train, labels, label_dimensions,i_val)
                    
                    
                    batch_loss_train += compute_interior_loss(gauss_points_new, batch_labels_train, NN,model_constraint,label_dimensions,hyperp,hyperp_new,data_input_shape,i_val,batch_pred_train,run_options,gauss_weights_new,Coordinates, Stiffness, load,Solution)
                    
                    

                    batch_loss_train_new_one=  data_loss(batch_pred_train, labels, label_dimensions,i_val)
                gradients = tape.gradient(batch_loss_train, NN.trainable_variables)
                optimizer.apply_gradients(zip(gradients, NN.trainable_variables))
                elapsed_time_batch = time.time() - start_time_batch
                if batch_num  == 0:
                    print('Time per Batch: %.2f' %(elapsed_time_batch))
                mean_loss_train(data_loss(batch_pred_train, labels, label_dimensions,i_val)) 
                mean_loss_train_constraint(compute_interior_loss(gauss_points_new, batch_labels_train, NN,model_constraint,label_dimensions,hyperp,hyperp_new,data_input_shape,i_val,batch_pred_train,run_options,gauss_weights_new,Coordinates, Stiffness, load,Solution)/model_constraint)
               # mean_accuracy_train(accuracy(batch_pred_train, batch_labels_train))
                
                                        

            
            #=== Computing Testing Metrics ===#
            for batch_data_test, batch_labels_test in data_and_labels_test:
                batch_pred_test = NN(batch_data_test)
            
                    
           
        
                
            
                mean_accuracy_test(accuracy(batch_pred_test, batch_labels_test,label_dimensions))
                

    
           
            
            #=== Update Storage Arrays ===#
            storage_array_loss = np.append(storage_array_loss, mean_loss_train_constraint.result())
            storage_array_accuracy = np.append(storage_array_accuracy, mean_accuracy_test.result())
            

            
            
            
            #=== Display Epoch Iteration Information ===#
            elapsed_time_epoch = time.time() - start_time_epoch
            #print('Time per Epoch: %.2f\n' %(elapsed_time_epoch))
            #print('Training Set: Loss: %.3e, Accuracy: %.3f' %(mean_loss_train.result(), mean_loss_train_constraint.result()))
            #print('Validation Set: Loss: %.3e, Accuracy: %.3f' %(mean_loss_val.result(), mean_accuracy_val.result()))
            #print('Test Set: Loss: %.3e, Accuracy: %.3f\n' %(mean_loss_test.result(), mean_accuracy_test.result()))
            #print('Previous Layer Relative # of 0s: %.7f\n' %(relative_number_zeros))
            
            if mean_accuracy_test.result()<0.02:
                L2_error=error_L2(gauss_points_new, gauss_solution, NN,label_dimensions,hyperp,data_input_shape,i_val,run_options,gauss_weights_new)
        
                print('L2 error: %.3e' %(L2_error))
            
            start_time_epoch = time.time()   


            
            #=== Reset Metrics ===#
            loss_validation = mean_loss_val.result()
            mean_loss_train.reset_states()
            mean_loss_train_constraint.reset_states()
            mean_loss_val.reset_states()
            mean_loss_test.reset_states()
            mean_accuracy_train.reset_states()
            mean_accuracy_val.reset_states()
            
            mean_accuracy_test.reset_states()
                   
        ########################################################
        #   Updating Architecture and Saving Current Metrics   #
        ########################################################  
        print('================================')
        print('     Extending Architecture     ')
        print('================================')          
        #=== Saving Metrics ===#
        metrics_dict = {}
        metrics_dict['loss'] = storage_array_loss
        metrics_dict['accuracy'] = storage_array_accuracy
        df_metrics = pd.DataFrame(metrics_dict)
        df_metrics.to_csv(file_paths.NN_savefile_name + "_metrics_hl" + str(trainable_hidden_layer_index) +str(i_val) + '.csv', index=False)
        
        #=== Sparsify Weights of Trained Layer ===#
        
        L2_error=error_L2(gauss_points_new, gauss_solution, NN,label_dimensions,hyperp,data_input_shape,i_val,run_options,gauss_weights_new)
        
        print('L2 error: %.3e' %(L2_error))
        

       
        #=== Add Layer ===#
        trainable_hidden_layer_index += 1


            
        #=== Preparing for Next Training Cycle ===#
        storage_array_loss = []
        storage_array_accuracy = []
        reset_optimizer   
        

        

    

