#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, BatchNormalization
from tensorflow.keras.initializers import RandomNormal, RandomUniform, Constant
import numpy as np
import pandas as pd
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"
from Utilities.Net import Final_Network
from Utilities.Net_new import Final_Network_ALGO_II
from sklearn.cluster import KMeans

# In[ ]:


def create_new(data_train, labels_train,hyperp,hyperp_new, run_options, data_input_shape, label_dimensions,i_val,load,Stiffness,Coordinates,Solution):
    
    y_true = labels_train
    
    for j in range(1,i_val+1):

        if j==1:
            data_train=data_train
            new_label=y_true
            Kmean=KMeans(n_clusters=5)
            Kmean.fit(data_train)
            Solution=Solution
            load=load
            labels_train_new=Kmean.labels_
            
            #x = tf.convert_to_tensor(data_train[:,0:1])
            #y = tf.convert_to_tensor(data_train[:,1:2])
            #labels_train_new=np.ones(len(x), dtype='>i4')
            #for i in range(0,len(x)):
             #   if np.abs(x[i]-0)<0.00000001:
                    #labels_train_new[i]=0
              #  if np.abs(x[i]-1)<0.00000001:
               #     labels_train_new[i]=0
               # if np.abs(y[i]-0)<0.00000001:
                #    labels_train_new[i]=0
               # if np.abs(y[i]-1)<0.00000001:
                #    labels_train_new[i]=0
               # if np.abs(x[i])>0.5 and np.abs(y[i]-0.5)<0.00000001:
                #    labels_train_new[i]=0
            
        if j==2:
            
    
            Network=Final_Network( hyperp,run_options, data_input_shape, label_dimensions) 
            #NNN._set_inputs( data_train)
            #Network.load_weights("WEIGHTS"+'/'+"model_weights"+str(j-1)+'.hdf5').expect_partial()
            #NNN.save("WEIGHTS"+'/'+"model"+str(j-1))
            #Network=tf.keras.models.load_model("WEIGHTS"+'/'+"model"+str(j-1))
            Network.load_weights("WEIGHTS"+'/'+"model_weights"+str(j-1)).expect_partial()
            y_pred=Network(data_train)
            y_pred_coordinates=Network(Coordinates)
            
            
            load_pred=tf.matmul(Stiffness, y_pred_coordinates)
            load=load-load_pred
            
            Solution=Solution-y_pred_coordinates
            new_label=tf.reshape(new_label,(len(y_pred),1))
            new_label=new_label-y_pred
        
            labels_train_new=y_true
        
        if j>2:
            
    
            Network=Final_Network_ALGO_II(hyperp_new,run_options, data_input_shape, label_dimensions) 
            #NNN._set_inputs( data_train)
            #Network.load_weights("WEIGHTS"+'/'+"model_weights"+str(j-1)+'.hdf5').expect_partial()
            #NNN.save("WEIGHTS"+'/'+"model"+str(j-1))
            #Network.load_weights("WEIGHTS"+'/'+"model_weights"+str(j-1)+'.h5').expect_partial()
            #Network=tf.keras.models.load_model("WEIGHTS"+'/'+"model"+str(j-1))
            Network.load_weights("WEIGHTS"+'/'+"model_weights"+str(j-1)).expect_partial()
            y_pred=Network(data_train)
            y_pred_coordinates=Network(Coordinates)
            
            load_pred=tf.matmul(Stiffness, y_pred_coordinates)
            load=load-load_pred
            
            Solution=Solution-y_pred_coordinates
            new_label=new_label-y_pred
            labels_train_new=y_true
        
        
            
    return data_train,new_label,labels_train_new,load,Solution
    

