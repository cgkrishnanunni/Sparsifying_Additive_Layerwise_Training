#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np
import pandas as pd
from Utilities.Net import Final_Network
from Utilities.additive_output import net_output 

import shutil # for deleting directories
import os
import time

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                             Training Properties                             #
###############################################################################
def optimize_new(hyperp, hyperp_new,run_options, file_paths, NN, data_loss, accuracy, data_and_labels_train, data_and_labels_val, data_and_labels_test, label_dimensions, num_batches_train,data_and_labels_train_new,batch_size,random_seed,num_data_train,i_val,data_input_shape):
    #=== Optimizer ===#
    #optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True)
    optimizer = tf.keras.optimizers.Adam()
    reset_optimizer = tf.group([v.initializer for v in optimizer.variables()])
    
    #=== Metrice ===#
    mean_loss_train = tf.keras.metrics.Mean()
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
    
    #=== Tensorboard ===# Tensorboard: type "tensorboard --logdir=Tensorboard" into terminal and click the link
    if os.path.exists('../Tensorboard/' + file_paths.filename): # Remove existing directory because Tensorboard graphs mess up of you write over it
        shutil.rmtree('../Tensorboard/' + file_paths.filename)  
    summary_writer = tf.summary.create_file_writer('../Tensorboard/' + file_paths.filename)

###############################################################################
#                             Train Neural Network                            #
############################################################################### 
    loss_validation = 1e5
    trainable_hidden_layer_index = 8
    relative_number_zeros = 0
    
    
    #####################################
    #   Training Current Architecture   #
    #####################################
    while trainable_hidden_layer_index < hyperp_new.max_hidden_layers:  
        
        

            
            
            
        
        #=== Initial Loss and Accuracy ===#
        #for batch_num, (batch_data_train, batch_labels_train,labels) in data_and_labels_train.enumerate():
            #batch_pred_train,val = NN(batch_data_train)
            #batch_loss_train = data_loss(batch_pred_train, labels, label_dimensions)
            #batch_loss_train += sum(NN.losses)+ manifold_class(batch_data_train, batch_labels_train, NN,manifold_regul,label_dimensions)
            #mean_loss_train(batch_loss_train) 
            #mean_accuracy_train(accuracy(batch_pred_train, batch_labels_train))

            
        #for batch_data_test, batch_labels_test in data_and_labels_test:
            
            #batch_pred_test,val = NN(batch_data_test)
            
            #y_pred_test_add=net_output(hyperp,batch_data_test, run_options, data_input_shape, label_dimensions,i_val,batch_pred_test)
        
            #batch_pred_test=batch_pred_test+y_pred_test_add
            #batch_pred_test=y_pred_test_add

            #mean_accuracy_test(accuracy(batch_pred_test, batch_labels_test))
            
        #storage_array_loss = np.append(storage_array_loss, mean_loss_train.result())
        #storage_array_accuracy = np.append(storage_array_accuracy, mean_accuracy_test.result())
        #print('Initial Losses:')
        
        #print('Training Set: Loss: %.3e' %(mean_loss_train.result()))
        
        #print('Test Set: Accuracy: %.3f\n' %(mean_accuracy_test.result()))
        
        #=== Begin Training ===#
        print('Beginning Training')
        for epoch in range(hyperp_new.num_epochs):
            
            
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
                    batch_pred_train,val = NN(batch_data_train)
                    #=== Display Model Summary ===#
                    if batch_num == 0 and epoch == 0:
                        NN.summary()
                    batch_loss_train = data_loss(batch_pred_train, labels, label_dimensions,i_val)
                    
                    
                    
                    #batch_loss_train_new_one=batch_loss_train
                    batch_loss_train_new_one= data_loss(batch_pred_train, labels, label_dimensions,i_val)
                gradients = tape.gradient(batch_loss_train, NN.trainable_variables)
                optimizer.apply_gradients(zip(gradients, NN.trainable_variables))
                elapsed_time_batch = time.time() - start_time_batch
                if batch_num  == 0:
                    print('Time per Batch: %.2f' %(elapsed_time_batch))
                mean_loss_train(batch_loss_train_new_one) 
               
               # mean_accuracy_train(accuracy(batch_pred_train, batch_labels_train))
                                        

            
            #=== Computing Testing Metrics ===#
            for batch_data_test, batch_labels_test in data_and_labels_test:
                

                
                batch_pred_test,val = NN(batch_data_test)
                
                
                y_pred_test_add=net_output(hyperp,hyperp_new,batch_data_test, run_options, data_input_shape, label_dimensions,i_val,batch_pred_test)
        
                batch_pred_test=batch_pred_test+y_pred_test_add
                #batch_pred_test=y_pred_test_add
                

                
                mean_accuracy_test(accuracy(batch_pred_test, batch_labels_test))
                
            #for batch_num, (batch_data_train, batch_labels_train,labels) in data_and_labels_train_new_new.enumerate():
                #batch_pred_train,vall = NN(batch_data_train)
                
            
                #y_pred_train_add=net_output(hyperp,hyperp_new,batch_data_train, run_options, data_input_shape, label_dimensions,i_val,batch_pred_train)
        
               # batch_pred_train=batch_pred_train+y_pred_train_add
                #batch_pred_test=y_pred_test_add
            
               # mean_accuracy_train(accuracy(batch_pred_train, batch_labels_train))
    
           
            
            #=== Update Storage Arrays ===#
            storage_array_loss = np.append(storage_array_loss, mean_loss_train.result())
            storage_array_accuracy = np.append(storage_array_accuracy, mean_accuracy_test.result())

            #=== Display Epoch Iteration Information ===#
            elapsed_time_epoch = time.time() - start_time_epoch
            print('Time per Epoch: %.2f\n' %(elapsed_time_epoch))
            print('Training Set: Loss: %.3e, Accuracy: %.3f' %(mean_loss_train.result(), mean_accuracy_train.result()))
            print('Validation Set: Loss: %.3e, Accuracy: %.3f' %(mean_loss_val.result(), mean_accuracy_val.result()))
            print('Test Set: Loss: %.3e, Accuracy: %.3f\n' %(mean_loss_test.result(), mean_accuracy_test.result()))
            print('Previous Layer Relative # of 0s: %.7f\n' %(relative_number_zeros))
            start_time_epoch = time.time()   
            
            #=== Reset Metrics ===#
            loss_validation = mean_loss_val.result()
            mean_loss_train.reset_states()
            mean_loss_val.reset_states()
            mean_loss_test.reset_states()
            mean_accuracy_train.reset_states()
            mean_accuracy_val.reset_states()
            mean_accuracy_test.reset_states()
            
        metrics_dict = {}
        metrics_dict['loss'] = storage_array_loss
        metrics_dict['accuracy'] = storage_array_accuracy
        df_metrics = pd.DataFrame(metrics_dict)
        df_metrics.to_csv(file_paths.NN_savefile_name + "_metrics_hl" + str(trainable_hidden_layer_index) +str(i_val) + '.csv', index=False)
    
                   

        trainable_hidden_layer_index += 1   
        #=== Preparing for Next Training Cycle ===#
        storage_array_loss = []
        storage_array_accuracy = []
        reset_optimizer   
        
    ########################
    #   Save Final Model   #
    ########################            
    #=== Saving Trained Model ===#     
    
    if not os.path.exists("WEIGHTS"):
        os.makedirs("WEIGHTS")
    
    #NN.save("my_model")
    #NN.save_weights("WEIGHTS"+'/'+"model_weights"+str(i_val))
    #print('Final Model Saved') 

