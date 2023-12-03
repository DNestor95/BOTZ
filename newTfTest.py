from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from IPython.display import clear_output
from six.moves import urllib

import tensorflow as tf
from tensorflow import feature_column as fc


# Load data
dftrain = pd.read_csv('modded_detection_data.csv')  # training data
dfeval = pd.read_csv('eval.csv')  # testing data

dftrain['BotLabel'] = dftrain['BotLabel'].astype(int)
dfeval['BotLabel'] = dfeval['BotLabel'].astype(int)
# Separate labels
y_train = dftrain.pop('BotLabel')
y_eval = dfeval.pop('BotLabel')

# Categorical columns
CATEGORICAL_COLUMNS = ['Username', 'Hashtags']  # Include 'Hashtags' in categorical columns
# Numeric columns
NUMERIC_COLUMNS = ['FollowerCount', 'RetweetCount', 'MentionCount', 'Verified']

feature_columns = []

# Numeric columns
for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

# Categorical columns
for feature_name in CATEGORICAL_COLUMNS:
    if feature_name == 'Hashtags':
        # Ensure embedding_dimension is defined before using it
        embedding_dimension = min(8, len(dftrain[feature_name].unique()) // 2)
        feature_columns.append(
            tf.feature_column.embedding_column(
                tf.feature_column.categorical_column_with_hash_bucket(feature_name, hash_bucket_size=1000),
                dimension=embedding_dimension))
    else:
        vocabulary = dftrain[feature_name].unique()
        feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))
        # Embedding for categorical columns
        embedding_dimension = min(8, len(vocabulary) // 2)
        feature_columns.append(
            tf.feature_column.embedding_column(
                tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary),
                dimension=embedding_dimension))

# Input function
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=8):
    def input_function():
        data_df['Hashtags'].fillna('', inplace=True)
        data_df.fillna('missing', inplace=True)
        data_df['Hashtags'] = data_df['Hashtags'].astype(str)
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function

#for loop to attempt batch sizes and epoch to find the most effictive testing the batch size aginst each epoch
batch_size = [8, 16, 32, 64, 128, 256, 512, 1024]
epoch = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
#dictionary for the accuracy and the batch size and epoch that it was tested with
batch_acc_dict = {}
epoch_acc_dict = {}
max_batch_acc = 0
max_batch_size = 0
max_epoch_acc = 0
max_epoch = 0


"""
for i in batch_size:
    train_input_fn = make_input_fn(dftrain, y_train, batch_size=i)
    eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

    linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
    linear_est.train(train_input_fn)
    result = linear_est.evaluate(eval_input_fn)

    clear_output()
    
    print(i)
    #add the accuracy and i to the dictiona
    batch_acc_dict[result['accuracy']] = i
    #if the accuracy is higher than the previous highest accuracy, set the new highest accuracy
    if result['accuracy'] > max_batch_acc:
        max_batch_acc = result['accuracy']
        max_batch_size = i
    
    print(result['accuracy'])
    print(result)
    
    """

#function that maakes a results file for the result that is passed in
def makeResultsFile(result, i, j):
    #open the file
    f = open("results.txt", "a")
    #write the accuracy
    f.write("Accuracy: " + str(result['accuracy']))
    
    
    #write the batch size
    f.write(" Batch Size: " + str(j))
    #write the epoch
    f.write(" Epoch: " + str(i))
    f.write("\n")
    #close the file
    f.close()




for i in epoch:
    for j in batch_size:
        train_input_fn = make_input_fn(dftrain, y_train, num_epochs=i, batch_size=j)
        eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

        linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
        linear_est.train(train_input_fn)
        result = linear_est.evaluate(eval_input_fn)

        clear_output()
        print(result['accuracy'])
        print(result)
        print("\n")
        print("Epoch: ")
        print(i)
        print("\n")
        print("Batch Size: ")
        print(j)
        makeResultsFile(result, i, j)

#function that maakes a results file for the result that is passed in

    
    
    
    
    
#This program is going to be able to give us most of the data that we need to be able to tell if what we are wanting to do it going to work or not 






        
        
"""
train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)

clear_output()
print(result['accuracy'])
print(result)
"""
