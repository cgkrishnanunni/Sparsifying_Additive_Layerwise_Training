

import tensorflow as tf
import numpy as np
from Utilities.Net import Final_Network

import random


def compute_interior_loss(gauss_points, batch_labels_train,NN,model_constraint,label_dimensions,hyperp,hyperp_new,data_input_shape,i_val,batch_pred_test,run_options,gauss_weights,Coordinates, Stiffness, load,Solution):


            # make prediction
        pred,new = NN(Coordinates)
            
        #y_pred_train_add=0*pred
        
        #if i_val>1:
         #   for i_net in range(2,i_val+1):
                
          #      if i_net==2:    
           #         Network=Final_Network( hyperp,run_options, data_input_shape, label_dimensions) 
                #NNN._set_inputs( data)
            #Network.load_weights("WEIGHTS"+'/'+"model_weights"+str(j-1)+'.hdf5').expect_partial()
                #NNN.save("WEIGHTS"+'/'+"model"+str(i_net-1))
                #Network.load_weights("WEIGHTS"+'/'+"model_weights"+str(i_net-1)+'.hdf5').expect_partial()
                #Network=tf.keras.models.load_model("WEIGHTS"+'/'+"model"+str(i_net-1))
            #        Network.load_weights("WEIGHTS"+'/'+"model_weights"+str(i_net-1)).expect_partial()
                
             #       y_pred_train_add=Network(Coordinates)
        
              #  if i_net>2:
                
               #     Network=Final_Network_ALGO_II( hyperp_new,run_options, data_input_shape, label_dimensions) 
                #NNN._set_inputs( data)
                    #Network.load_weights("WEIGHTS"+'/'+"model_weights"+str(i_net-1)+'.hdf5').expect_partial()
                #NNN.save("WEIGHTS"+'/'+"model"+str(i_net-1))
                #    Network.load_weights("WEIGHTS"+'/'+"model_weights"+str(i_net-1)).expect_partial()
                    #Network=tf.keras.models.load_model("WEIGHTS"+'/'+"model"+str(i_net-1))
                #Network.load_weights("WEIGHTS"+'/'+"model_weights"+str(i_net-1)).expect_partial()
                
                 #   y_pred_train_add=y_pred_train_add+Network(Coordinates)
                
                   
            
            
            
            
            
        #pred=pred+y_pred_train_add
        
        
        
        loss=tf.matmul(Stiffness, pred)
        #X=tf.matmul(tf.transpose(pred-Solution),tf.transpose(Stiffness))
        #Y=tf.matmul(X,Stiffness)
        #Z=tf.matmul(Y,pred-Solution)
        #X=tf.matmul(pred-Solution,tf.transpose(pred-Solution))
        #Y=tf.matmul(X,tf.transpose(Stiffness))
        #Z=tf.matmul(Y,Stiffness)
        
        #xss = random.randint(0,len(loss))
        #yss = random.randint(0,len(loss))
        #if xss<=yss:
            
         #   load_new=load[xss:yss]
          #  loss_new=loss[xss:yss]
        #if xss>yss:
         #   load_new=load[yss:xss]
          #  loss_new=loss[yss:xss]
        #loss_interest=0
        #for i in range(0,len(v)):
         #   loss_interest=loss_interest+(Solution[v[i]]-pred[v[i]])**2
            
            
        loss_final=tf.math.reduce_sum(tf.keras.losses.MSE(load, loss))
        #loss_final=tf.math.reduce_max(tf.square(load-loss))
        #loss_final=tf.math.reduce_sum(tf.math.abs(load-loss))
        #+tf.math.reduce_sum(tf.keras.losses.MSE(Solution, pred))
        #loss_new=tf.square(loss)
        #loss_final=tf.math.reduce_sum(loss_new)


        return model_constraint*(loss_final)

