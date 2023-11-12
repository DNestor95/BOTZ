from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf
dftrain = pd.read_csv('bot_detection_data.csv') # training data
dfeval = pd.read_csv() # testing data
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

##table names User ID,Username,Tweet,Retweet Count,Mention Count,Follower Count,Verified,Location,Created At,Hashtags,Bot Label

CATEGORICAL_COLUMNS = ['username', 'location', 'hashtags','Verified']
NUMERIC_COLUMNS = ['retweet_count', 'mention_count', 'follower_count', 'verified', 'Bot_label']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique()  # gets a list of all unique values from given feature column
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

print(feature_columns)