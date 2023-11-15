from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib




import tensorflow as tf
from tensorflow import feature_column as fc

dftrain = pd.read_csv('bot_detection_data.csv') # training data
dfeval = pd.read_csv('eval.csv') # testing data

y_train = dftrain.pop('BotLabel')
y_eval = dfeval.pop('BotLabel')
##User ID,Username,Tweet,RetweetCount,MentionCount,FollowerCount,Verified,Location,CreatedAt,Hashtags,BotLabel
dftrain.head()

##THESE WILL BE ALL THE LABELS TAHT CONTAIN NON NUMBERIC VALUES 
CATEGORICAL_COLUMNS = ['Username', 'Location', 'Hashtags', 'CreatedAt']
##THESE WILL BE ALL THE LABELS TAHT CONTAIN NUMBERIC VALUES

NUMERIC_COLUMNS = ['FollowerCount', 'RetweetCount', 'MentionCount']


feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique()  # gets a list of all unique values from given feature column
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

#possible use a lambda function 

print(feature_columns)
