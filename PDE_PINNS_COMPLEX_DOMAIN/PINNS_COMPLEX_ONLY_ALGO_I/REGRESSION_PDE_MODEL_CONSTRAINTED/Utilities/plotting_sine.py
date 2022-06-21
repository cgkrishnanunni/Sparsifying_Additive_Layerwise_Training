import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, BatchNormalization
from tensorflow.keras.initializers import RandomNormal, RandomUniform, Constant
import numpy as np
import pandas as pd
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"
from Utilities.Net import Final_Network
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.pyplot as plt
import os



def plot_solution(hyperp, data, run_options, data_input_shape, label_dimensions,i_val,X_train, Y_train,labels_test):


    
    hyperp.max_hidden_layers=i_val
    
    x = np.linspace(0, 1, 1000)
    y = np.linspace(0, 1, 1000)
    X, Y = np.meshgrid(x, y)

    d = []
    for i in range(0,1000):
        for j in range(0,1000):
            dd=[X[i,j],Y[i,j]]
            d.append(dd) 

    d=np.array(d)
    data= tf.cast(d,tf.float32)
  
    Network=Final_Network( hyperp,run_options, data_input_shape, label_dimensions) 
        
    Network.load_weights("WEIGHTS"+'/'+"model_weights"+str(1)+str(i_val)).expect_partial()
    
    y_pred_test_add, r=Network(data)
        
         
        
    fig_loss = plt.figure()
    
    
    
    
   

        
      
    Z=tf.reshape(y_pred_test_add,np.shape(X))
#clevels = np.linspace(0.0019333754, 14.727704, 10000)
    levels=np.linspace(-0.5, 9, 1000)
    cs=plt.contourf(X, Y, Z,levels, cmap=cm.jet,vmax=9, vmin=-0.5)     
#plt.contourf(X, Y, Z, 1000, cmap='RdYlBu_r', vmax=14.727704, vmin=0.) 
    plt.colorbar() 

    for c in cs.collections:
        c.set_rasterized(True)
    
    
    
    
    
    
    
    

    plt.show()



    if not os.path.exists("plots"):
        os.makedirs("plots")
        

    fig_loss.savefig("plots"+str(i_val)+'.png')
    
    
