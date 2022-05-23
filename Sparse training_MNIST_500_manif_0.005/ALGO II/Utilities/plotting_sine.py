import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, BatchNormalization
from tensorflow.keras.initializers import RandomNormal, RandomUniform, Constant
import numpy as np
import pandas as pd
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"
from Utilities.Net import Final_Network
from Utilities.Net_new import Final_Network_ALGO_II
import matplotlib.pyplot as plt

# In[ ]:


def plot_sine(hyperp,hyperp_new, data, run_options, data_input_shape, label_dimensions,i_val,X_train, Y_train):

    i_val=i_val+1
    
    
    if i_val>1:
        for i_net in range(2,i_val+1):
                
            if i_net==2:    
                Network=Final_Network( hyperp,run_options, data_input_shape, label_dimensions) 
        
                Network.load_weights("WEIGHTS"+'/'+"model_weights"+str(i_net-1)).expect_partial()
    
                y_pred_test_add=Network(data)
        
            if i_net>2:
                
                #Network=Final_Network_ALGO_II( hyperp_new,run_options, data_input_shape, label_dimensions) 
                Network=tf.keras.models.load_model("WEIGHTS"+'/'+"model"+str(i_net-1))
                #Network.load_weights("WEIGHTS"+'/'+"model_weights"+str(i_net-1)).expect_partial()
    
                y_pred_test_add=y_pred_test_add+Network(data)
        
    fig_loss = plt.figure()
        
    #plt.plot(data, y_pred_test_add, 'ko')
   



        
    x = data[:,0]
    y = data[:,1]
        
    fig, ax = plt.subplots()
    clevels = np.linspace(np.amin(y_pred_test_add), np.amax(y_pred_test_add), 10000)
    #v = np.linspace(0, 8, 10, endpoint=True)
    im = ax.tricontourf(x, y, y_pred_test_add[:,0], clevels, cmap='RdYlBu_r',vmax=8, vmin=0.)
    
    plt.colorbar(im)
    plt.show()



    #plt.title('Predictions After Training Neural network {}'.format(i_val-1))
    #plt.xlabel('x')
    #plt.ylabel('y=sin(x)')
    #plt.legend(['Predicted','True'])

    fig_loss.savefig("plots"+'/'+"AlgoII"+str(i_val-1)+'.png')
    
    
