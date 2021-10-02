# imports
import numpy as np
import pandas as pd
import time

def Acc_eval( y_hat, y):
  """
  Computes the accuracy of prediction y_hat with respect to known target y

  Parameters:
  ----------
  y, y_hat: matrixes of shape (n_features) containing known target and prediction

  Returns:
  -------
  accuracy: percentage of accuracy
  """
  return np.count_nonzero([y == y_hat]) / len(y) *100



def cross_val( estimator, X, y, n_folds):
  """
  Implements cross validation to obtain true error of a given model

  Parameters:
  -----------
  estimator: object of an estimator class with fit and predict methods
  X:{array} of shape (n_samples, n_features)
      Training vector, where n_samples is the number of samples and
      n_features is the number of features
  y:{array} of shape (n_samples), target vector relative to X
  n_folds: integer that indicates the number of sets to divide the data to
    make the cross validation

  Returns:
  -------
  accuracy: average accuracy through the n_folds of the cross validation. uses the function
  Acc_eval.
  """
  X_split = np.array_split(X,n_folds)
  y_split = np.array_split(y,n_folds)

  accuracies = []

  start_time = time.time()

  for fold in np.arange(n_folds):
    X_test = np.concatenate([X_split[i] for i in np.arange(n_folds) if i != fold])
    X_val = X_split[fold]
    y_test = np.concatenate([y_split[i] for i in np.arange(n_folds) if i != fold])
    y_val = y_split[fold]

    estimator.fit(X_test, y_test)
    y_hat = estimator.predict(X_val)
    accuracy_fold = Acc_eval(y_hat, y_val)
    accuracies += [accuracy_fold]

  # print("Cross Validation time:" + str(time.time() - start_time) + " seconds")
  return np.mean(accuracies)

def model_test( estimator, X_val, y_val, X, y):
  """
  Implements test to obtain true error of a given model

  Parameters:
  -----------
  estimator: object of an estimator class with fit and predict methods
  X:{array} of shape (n_samples, n_features)
      Testing vector, where n_samples is the number of samples and
      n_features is the number of features
  y:{array} of shape (n_samples), target vector relative to X

  Returns:
  -------
  accuracy: average accuracy through the n_folds of the cross validation. uses the function
  model_test.
  """
  estimator.fit(X_val, y_val)
  y_hat = estimator.predict(X)
  accuracies = Acc_eval(y_hat, y)
  return accuracies