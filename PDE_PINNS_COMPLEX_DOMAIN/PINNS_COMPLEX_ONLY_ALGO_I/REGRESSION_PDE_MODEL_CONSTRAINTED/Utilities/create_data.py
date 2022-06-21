

import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, BatchNormalization
from tensorflow.keras.initializers import RandomNormal, RandomUniform, Constant
import numpy as np
import pandas as pd
import pdb 
from Utilities.Net import Final_Network

from sklearn.cluster import KMeans

# In[ ]:


def create_new(data_train, labels_train,hyperp, run_options, data_input_shape, label_dimensions,i_val,load,Stiffness,Coordinates,Solution):
 



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
            

        
            
    return data_train,new_label,labels_train_new,load,Solution
    

