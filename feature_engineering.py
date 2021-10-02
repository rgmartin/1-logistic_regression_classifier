import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from model_evaluation import cross_val
import time

def re_sample(df_data, target_column_label = 'ClassLabel', random_state=42):
  df_data = shuffle(df_data, random_state = 42)
  X = df_data.drop(columns=target_column_label).copy()
  intercept = np.ones((X.shape[0], 1))
  X = np.concatenate((intercept, X), axis=1)
  y= df_data[target_column_label]
  return X, y;


def logarithm_transformer (data_frame):
  data= data_frame.copy()
  continuous_features = [ feature for feature in data if len(data[feature].unique()) > 25]  
  for feature in continuous_features:
    if 0 not in data[feature].unique():
      data[feature] = np.log(data[feature])
  return data

def standard_scaler(data_frame):
  data= data_frame.copy()
  continuous_features = [ feature for feature in data if len(data[feature].unique()) > 25]
  for feature in continuous_features:
    data[feature] = data[feature] - np.mean(data[feature])
    data[feature] = data[feature] / np.std(data[feature])
  return data

def quadratic_feature_tester(data_frame, features):
  data= data_frame.copy()
  for feature in features:
    data[feature+'_2'] = data[feature] ** 2
  return data

def hola():
	print('hola')