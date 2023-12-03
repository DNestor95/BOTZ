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
CATEGORICAL_COLUMNS = ['Username', 'Hashtags']
# Numeric columns
NUMERIC_COLUMNS = ['FollowerCount', 'RetweetCount', 'MentionCount', 'Verified']

feature_columns = []

# Numeric columns
for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

# Categorical columns
for feature_name in CATEGORICAL_COLUMNS:
    if feature_name == 'Hashtags':
        embedding_dimension = min(8, len(dftrain[feature_name].unique()) // 2)
        feature_columns.append(
            tf.feature_column.embedding_column(
                tf.feature_column.categorical_column_with_hash_bucket(feature_name, hash_bucket_size=1000),
                dimension=embedding_dimension))
    else:
        vocabulary = dftrain[feature_name].unique()
        feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))
        embedding_dimension = min(8, len(vocabulary) // 2)
        feature_columns.append(
            tf.feature_column.embedding_column(
                tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary),
                dimension=embedding_dimension))

# Input layer
inputs = {colname: tf.keras.layers.Input(name=colname, shape=(), dtype='float32') for colname in dftrain.columns}

# Output layer for binary classification
output_layer = tf.keras.layers.DenseFeatures(feature_columns)(inputs)
output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(output_layer)

# Create the model
model = tf.keras.models.Model(inputs, output_layer)

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

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

# Batch size and epoch loop for experimentation
batch_size = [8, 16, 32, 64, 128, 256, 512, 1024]

# Dictionary for accuracy and the batch size that was tested with
batch_acc_dict = {}
max_batch_acc = 0
max_batch_size = 0

# Training the model
for bs in batch_size:
    train_input_fn = make_input_fn(dftrain, y_train, batch_size=bs)
    eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

    model.fit(train_input_fn(), validation_data=eval_input_fn(), epochs=10, steps_per_epoch=100)
    results = model.evaluate(eval_input_fn())
    batch_acc_dict[bs] = results[1]

    if results[1] > max_batch_acc:
        max_batch_acc = results[1]
        max_batch_size = bs

print("Max Batch Size:", max_batch_size)
print("Max Batch Accuracy:", max_batch_acc)