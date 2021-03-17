#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_fig(hyperp, run_options, file_paths):
    storage_loss_array=[]
    storage_accuracy_array=[]
    max_hidden_layers=hyperp.max_hidden_layers
    no_epoch=hyperp.num_epochs
    
    for i in range(2,max_hidden_layers):
    
        trainable_hidden_layer_index=i
    
    
        name=file_paths.NN_savefile_name + "_metrics_hl" + str(trainable_hidden_layer_index) + '.csv'


        df_metrics =pd.read_csv(name)

        array_metrics = df_metrics.to_numpy()

        storage_loss_array=np.concatenate((storage_loss_array, array_metrics[:,0]), axis=0)
 
        storage_accuracy_array=np.concatenate((storage_accuracy_array, array_metrics[:,1]), axis=0)

#=== Plot and Save Losses===#
    fig_loss = plt.figure()
    x_axis = np.linspace(1, len(storage_loss_array), len(storage_loss_array), endpoint = True)
    plt.plot(x_axis, storage_loss_array)
    plt.title('Loss plot with layer adaptation' )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    

#=== Plot and Save Accuracies===#
    fig_accuracy = plt.figure()
    x_axis = np.linspace(1, len(storage_accuracy_array), len(storage_accuracy_array), endpoint = True)
    plt.plot(x_axis, storage_accuracy_array)
#plt.title('Accuracy for: ' + run_options.filename)
    plt.title('Accuracy plot with layer adaptation ')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

