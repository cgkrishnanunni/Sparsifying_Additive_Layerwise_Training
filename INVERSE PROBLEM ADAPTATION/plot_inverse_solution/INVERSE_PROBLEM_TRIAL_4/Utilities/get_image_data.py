
import tensorflow as tf
from tensorflow.keras import datasets
import sklearn.model_selection as sk
import numpy as np
import pandas as pd
from sklearn import preprocessing

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def load_data(NN_type, dataset, random_seed):
    #=== Load Data ==#
    if dataset == 'MNIST':
        (data_train, labels_train), (data_test, labels_test) = datasets.mnist.load_data()
        data_train = data_train.reshape(data_train.shape[0], 28, 28, 1)
        data_test = data_test.reshape(data_test.shape[0], 28, 28, 1)
        label_dimensions = 10
    if dataset == 'CIFAR10':
        (data_train, labels_train), (data_test, labels_test) = datasets.cifar10.load_data()
        label_dimensions = 10
    if dataset == 'Abalone':
        #target_url = ("http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data")
        #abalone = pd.read_csv(target_url,header=None, prefix="V")
        #abalone.columns = ['Sex', 'Length', 'Diameter', 'Height',
         #          'Whole weight', 'Shucked weight',
          #         'Viscera weight', 'Shell weight', 'Rings']
        #abalone = pd.get_dummies(abalone)
        #y = abalone["Rings"]
        #X = abalone.drop(columns="Rings")
        #X_train, X_test, y_train, y_test=sk.train_test_split(X, y, test_size=0.3)
        #X_train=np.array(X_train)
        #Y_train=np.array(y_train)
        #X_test=np.array(X_test)
        #Y_test=np.array(y_test)
        data_train=np.loadtxt("data_train.data")
        labels_train=np.loadtxt("labels_train.data")
        data_test=np.loadtxt("data_test.data")
        labels_test=np.loadtxt("labels_test.data")
        
        data_val=np.loadtxt("data_val.data")
        labels_val=np.loadtxt("labels_val.data")
        
        labels_train_manifold=np.loadtxt("labels_train_manifold.data")
        data_train_manifold=np.loadtxt("data_train_manifold.data")
        
        label_dimensions = 12
        
    if dataset == 'CIFAR100':
        (data_train, labels_train), (data_test, labels_test) = datasets.cifar100.load_data()
        label_dimensions = 100
        
    
    
    #=== Casting as float32 ===#
    data_train = tf.cast(data_train,tf.float32)
    labels_train = tf.cast(labels_train, tf.float32)

    data_test = tf.cast(data_test, tf.float32)
    labels_test = tf.cast(labels_test, tf.float32)
    
    #=== Normalize Data ===#
    #Normalization

    #data_train=preprocessing.normalize(data_train)
    #data_test=preprocessing.normalize(data_test)
    

    
    #=== Define Outputs ===#
    data_input_shape = data_train.shape[1:]
    if NN_type == 'CNN':
        num_channels = data_train.shape[-1]
    else:
        num_channels = None

    return data_train, labels_train, data_test, labels_test, data_input_shape, num_channels, label_dimensions, labels_train_manifold, data_train_manifold, data_val, labels_val