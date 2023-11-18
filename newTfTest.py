from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
def make_input_fn(data_df, label_df, num_epochs=20, shuffle=True, batch_size=120):
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

train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)

clear_output()
print(result['accuracy'])
