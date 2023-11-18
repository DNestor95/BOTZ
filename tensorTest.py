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
##UserID,Username,Tweet,RetweetCount,MentionCount,FollowerCount,Verified,Location,CreatedAt,Hashtags,BotLabel


##THESE WILL BE ALL THE LABELS TAHT CONTAIN NON NUMBERIC VALUES 
CATEGORICAL_COLUMNS = ['Username', 'Location', 'Tweet']
##THESE WILL BE ALL THE LABELS TAHT CONTAIN NUMBERIC VALUES

NUMERIC_COLUMNS = ['UserID','FollowerCount', 'RetweetCount', 'MentionCount','Verified', 'Hashtags']


feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique()  # gets a list of all unique values from given feature column
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

#possible use a lambda function 

def make_input_fn(data_df, label_df, num_epochs=20, shuffle=True, batch_size=120):
  def input_function():  # inner function, this will be returned
    data_df.fillna('missing', inplace=True)
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return a function object for use

train_input_fn = make_input_fn(dftrain, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)


linear_est.train(train_input_fn)  #train
result = linear_est.evaluate(eval_input_fn)  # get model metrics/stats by testing on tetsing data

clear_output()  # clears consoke output
print(result['accuracy'])  # the result variable is simply a dict of stats about our model