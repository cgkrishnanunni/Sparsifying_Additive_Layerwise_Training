
import tensorflow as tf
import numpy as np
import pandas as pd
import math
from Utilities.Net import Final_Network
from Utilities.additive_output import net_output 
from Utilities.multiplicative_output import net_output_multiply 

import shutil # for deleting directories
import os
import time

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                             Training Properties                             #
###############################################################################
def optimize(hyperp, hyperp_new,run_options, file_paths, NN, data_loss, accuracy, data_and_labels_train, data_and_labels_val, data_and_labels_test, label_dimensions, num_batches_train,data_and_labels_train_new,manifold_class,batch_size,random_seed,num_data_train,i_val,data_input_shape,data_train,labels_train,multiply,trainable_hidden_layer_index, labels_train_manifold, data_train_manifold):
    
    
    rat=0
    #=== Optimizer ===#
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
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
    


###############################################################################
#                             Train Neural Network                            #
############################################################################### 
    loss_validation = 1e5
    #trainable_hidden_layer_index = 2
    relative_number_zeros = 0
    manifold_regul=hyperp_new.manifold
    max_hidden_layers=hyperp.max_hidden_layers
    #max_hidden_layers=trainable_hidden_layer_index+1
    #####################################
    #   Training Current Architecture   #
    #####################################
    while trainable_hidden_layer_index < max_hidden_layers: 
        

        
        if trainable_hidden_layer_index>2:
            NN.add_layer(trainable_hidden_layer_index, freeze=True, add = True)
        #if trainable_hidden_layer_index>2:
            #hyperp_new.num_epochs=500
            #optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
            #hyperp_new.num_epochs=400
        
        if trainable_hidden_layer_index>2:
            manifold_regul=hyperp_new.manifold * math.pow(1.5,(trainable_hidden_layer_index-2))

            
            
            
        

        
        #=== Begin Training ===#
        print('Beginning Training')
        fr=0
        for epoch in range(hyperp_new.num_epochs):

            lrate = 0.001 * math.pow(1,(epoch))    

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
                    batch_pred_train,val = NN(batch_data_train)
                    #=== Display Model Summary ===#
                    if batch_num == 0 and epoch == 0:
                        NN.summary()
                    batch_loss_train = data_loss(batch_pred_train, labels, label_dimensions,i_val)
                    
                    
                    batch_loss_train += sum(NN.losses)+ manifold_class(data_train_manifold, labels_train_manifold, NN,manifold_regul,label_dimensions)
                    #batch_loss_train += sum(NN.losses)
                    #batch_loss_train_new_one=batch_loss_train
                    #batch_loss_train_new_one= data_loss(batch_pred_train, labels, label_dimensions,i_val)
                    batch_loss_train_new_one=  data_loss(batch_pred_train, labels, label_dimensions,i_val)
                gradients = tape.gradient(batch_loss_train, NN.trainable_variables)
                optimizer.apply_gradients(zip(gradients, NN.trainable_variables))
                elapsed_time_batch = time.time() - start_time_batch
                if batch_num  == 0:
                    print('Time per Batch: %.2f' %(elapsed_time_batch))
                mean_loss_train(batch_loss_train_new_one) 
                
               # mean_accuracy_train(accuracy(batch_pred_train, batch_labels_train))
                                        

            
            #=== Computing Testing Metrics ===#
            for batch_data_test,  batch_labels_test in data_and_labels_val:
                batch_pred_val,val = NN(batch_data_test)
                
                if multiply==1:
                    y_pred_test_add=net_output_multiply(hyperp,hyperp_new,batch_data_test, run_options, data_input_shape, label_dimensions,i_val,batch_pred_val)
        
                    batch_pred_val=np.multiply(batch_pred_val,y_pred_test_add)   
                    #batch_pred_test=batch_pred_test
                if multiply==0:
                    
                    y_pred_test_add=net_output(hyperp,hyperp_new,batch_data_test, run_options, data_input_shape, label_dimensions,i_val,batch_pred_val)
        
                    batch_pred_val=batch_pred_val+y_pred_test_add
            
                mean_accuracy_val(accuracy(batch_pred_val, batch_labels_test,label_dimensions))
                
            for batch_data_test, batch_labels_test in data_and_labels_test:
                batch_pred_test,val = NN(batch_data_test)
                
                if multiply==1:
                    y_pred_test_add=net_output_multiply(hyperp,hyperp_new,batch_data_test, run_options, data_input_shape, label_dimensions,i_val,batch_pred_test)
        
                    batch_pred_test=np.multiply(batch_pred_test,y_pred_test_add)   
                    #batch_pred_test=batch_pred_test
                if multiply==0:
                    
                    y_pred_test_add=net_output(hyperp,hyperp_new,batch_data_test, run_options, data_input_shape, label_dimensions,i_val,batch_pred_test)
        
                    batch_pred_test=batch_pred_test+y_pred_test_add
            
                mean_accuracy_test(accuracy(batch_pred_test, batch_labels_test,label_dimensions))   
            #for batch_num, (batch_data_train, batch_labels_train,labels) in data_and_labels_train_new_new.enumerate():
                #batch_pred_train,vall = NN(batch_data_train)
                
            
                #y_pred_train_add=net_output(hyperp,hyperp_new,batch_data_train, run_options, data_input_shape, label_dimensions,i_val,batch_pred_train)
        
               # batch_pred_train=batch_pred_train+y_pred_train_add
                #batch_pred_test=y_pred_test_add
            
               # mean_accuracy_train(accuracy(batch_pred_train, batch_labels_train))
    
           
            
            #=== Update Storage Arrays ===#
            storage_array_loss = np.append(storage_array_loss, mean_loss_train.result())
            storage_array_accuracy = np.append(storage_array_accuracy, mean_accuracy_val.result())

            #=== Display Epoch Iteration Information ===#
            elapsed_time_epoch = time.time() - start_time_epoch
            print('Time per Epoch: %.2f\n' %(elapsed_time_epoch))
            print('Training Set: Loss: %.3e, Accuracy: %.3f' %(mean_loss_train.result(), mean_accuracy_train.result()))
            print('Validation Set: Loss: %.3e, Accuracy: %.3f' %(mean_loss_val.result(), mean_accuracy_val.result()))
            print('Test Set: Loss: %.3e, Accuracy: %.3f\n' %(mean_loss_test.result(), mean_accuracy_test.result()))
            print('Previous Layer Relative # of 0s: %.7f\n' %(relative_number_zeros))
            
            start_time_epoch = time.time()   
            rr=mean_loss_train.result()
            bb=mean_accuracy_val.result()
            fr=bb/rr
            
            if mean_accuracy_test.result()<0.422:
                rat=1
                break
            
            #=== Reset Metrics ===#
            loss_validation = mean_loss_val.result()
            mean_loss_train.reset_states()
            mean_loss_val.reset_states()
            mean_loss_test.reset_states()
            mean_accuracy_train.reset_states()
            mean_accuracy_val.reset_states()
            
            mean_accuracy_test.reset_states()
                   
        ########################################################
        #   Updating Architecture and Saving Current Metrics   #
        ########################################################  
        if rat==1:
            break
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
        if run_options.use_L1 == 1 and i_val==1:
            relative_number_zeros = NN.sparsify_weights_and_get_relative_number_of_zeros(hyperp_new.node_TOL)
            print('Previous Layer Relative # of 0s: %.7f\n' %(relative_number_zeros))
            storage_array_relative_number_zeros = np.append(storage_array_relative_number_zeros, relative_number_zeros)
        
        #=== Saving Relative Number of Zero Elements ===#
            #relative_number_zeros_dict = {}
            #relative_number_zeros_dict['rel_zeros'] = storage_array_relative_number_zeros
            #df_relative_number_zeros = pd.DataFrame(relative_number_zeros_dict)
            #df_relative_number_zeros.to_csv(file_paths.NN_savefile_name + "_relzeros" + '.csv', index=False)
       
        #=== Add Layer ===#
        trainable_hidden_layer_index += 1
        #if trainable_hidden_layer_index < hyperp_new.max_hidden_layers and i_val==1:
         
            #NN.add_layer(trainable_hidden_layer_index, freeze=True, add = True)

            
        #=== Preparing for Next Training Cycle ===#
        storage_array_loss = []
        storage_array_accuracy = []
        reset_optimizer   
        
    ########################
    #   Save Final Model   #
    ########################            
    #=== Saving Trained Model ===#     
    
    #if not os.path.exists("WEIGHTS"):
     #   os.makedirs("WEIGHTS")
    #NN.save_weights("WEIGHTS"+'/'+"model_weights"+str(i_val)+'.h5')
    #NN.save("WEIGHTS"+'/'+"model"+str(i_val))

    #print('Final Model Saved') 
        

    

