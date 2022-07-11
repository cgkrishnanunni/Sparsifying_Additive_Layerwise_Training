
import tensorflow as tf


import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def form_train_val_test_batches(data_train, labels_train, data_test, labels_test, batch_size, new_labels, random_seed):
    num_data_train = len(data_train)
    num_data_test = len(data_test)
    

    
    #=== Shuffling Data ===#
    data_and_labels_train_full = tf.data.Dataset.from_tensor_slices((data_train, labels_train,new_labels)).shuffle(num_data_train, seed=random_seed)
    data_and_labels_test = tf.data.Dataset.from_tensor_slices((data_test, labels_test)).batch(batch_size)
    num_batches_test = len(list(data_and_labels_test))

    #=== Partitioning Out Validation Set and Constructing Batches ===#
    current_num_data_train = num_data_train
    num_data_train = int(1 * num_data_train)
    num_data_val = current_num_data_train - num_data_train
    data_and_labels_train_new = data_and_labels_train_full.take(num_data_train)
    data_and_labels_train = data_and_labels_train_new.batch(batch_size)
    data_and_labels_val = data_and_labels_train_full.skip(num_data_train).batch(batch_size)    
    num_batches_train = len(list(data_and_labels_train))
    num_batches_val = len(list(data_and_labels_val))

    return data_and_labels_train, data_and_labels_val, data_and_labels_test, num_data_train, num_data_val, num_data_test, num_batches_train, num_batches_val, num_batches_test,data_and_labels_train_new