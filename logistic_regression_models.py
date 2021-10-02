# imports
import numpy as np
import pandas as pd
from scipy import special



# Auxiliary functions

def sigmoid(X, w):
  """
  Sigmoid function

  Parameters:
  x: {vector} of shape (n_features)
    Column vector corresponding to n_features of one sample
  w: {vector} of shape (n_features)
    Column vector corresponding to the weights
  """  
  z = np.dot(X, w)
  return special.expit(z)

def cross_entropy_loss(h,y):
  """
  Returns the cross entropy loss

  Parameters
  h = scalar, result of using the sigmoid function: h = sigmoid (w' X)
  y: scalar with the target class
  """
  return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


def gradient_descent(X,h,y):
  return np.dot(X.T, (h-y)) / y.shape[0]
def update_weight_loss(w,learning_rate,gradient):
  return w - learning_rate *gradient

def log_likelihood(x,y, w):
  z=np.dot(x,w)
  ll = np.sum(y*z - np.log(1+np.exp(z)))
  return ll

def gradient_ascent(X, h, y):
    return np.dot(X.T, y - h)
def update_weight_mle(weight, learning_rate, gradient):
    return weight + learning_rate * gradient


# Logistic Regression by minimization of Cross Entropy Loss through gradient descent

class LogisticRegression_gradient_descent:
  """
  This class implements logistic regression.
  Parameters:

  Attributes:

  """
  def __init__(self, learning_rate=0.1, max_iter= 100000, rel_tol = -np.inf, abs_tol = -np.inf, print_time=False):
    self.max_iter = max_iter    
    self.learning_rate = learning_rate
    self.rel_tol = rel_tol
    self.abs_tol = abs_tol
    self.print_time = print_time

  def fit (self, X, y):
    """
    Fit the model according to the given training data

    Parameters
    ----------
    X:{array} of shape (n_samples, n_features)
      Training vector, where n_samples is the number of samples and
      n_features is the number of features
    y:{array} of shape (n_samples), default = None
      Target vector relative to X.

    Returns
    -------
    self
      Fitted estimator    
    """
    if self.print_time:
      start_time = time.time()

    self.w = np.zeros(X.shape[1])       
    for i in range(self.max_iter):      
      self.h= sigmoid(X,self.w)
      self.gradient = gradient_descent(X,self.h,y)

      new_w = update_weight_loss(self.w, self.learning_rate, self.gradient) 


      rel_error = np.linalg.norm(new_w - self.w) /  np.linalg.norm(new_w)
      abs_error = np.linalg.norm(new_w - self.w) 

      if (rel_error < self.rel_tol) or (abs_error < self.abs_tol):        
        break

      self.w = new_w
    
    if self.print_time:
      print("Fitting time:" + str(time.time() - start_time) + " seconds")
  
  def predict(self, X):
    """
    Extimated prediction.

    Parameters
    ---------
    X: array of shape (n_samples, n_features)

    Returns
    -------
    y: array of shape(n_samples)
    Each row has the probability of y being of class 1, therefore we need to further process the results to 
    compare them with 0.5 so as to classify the output as 0 or 1
    """ 
    result  =   sigmoid(X,self.w)
    result = np.where(result<0.5, 0 ,1)
    return result


# Logistic Regression by maximization of Log Likelihood through gradient ascent

class LogisticRegression_maximum_likelihood:
  """
  This class implements logistic regression.
  Parameters:

  Attributes:

  """
  def __init__(self, learning_rate=0.1, max_iter= 100000, rel_tol = -np.inf, abs_tol = -np.inf, print_time = False):
    self.max_iter = max_iter    
    self.learning_rate = learning_rate
    self.rel_tol = rel_tol
    self.abs_tol = abs_tol
    self.print_time = print_time


  def fit (self, X, y):
    """
    Fit the model according to the given training data

    Parameters
    ----------
    X:{array} of shape (n_samples, n_features)
      Training vector, where n_samples is the number of samples and
      n_features is the number of features
    y:{array} of shape (n_samples), default = None
      Target vector relative to X.

    Returns
    -------
    self
      Fitted estimator    
    """

    if self.print_time:
      start_time = time.time()

    self.w = np.zeros(X.shape[1])       
    for i in range(self.max_iter):      
      self.h= sigmoid(X,self.w)
      self.gradient = gradient_ascent(X,self.h,y)

      new_w = update_weight_mle(self.w, self.learning_rate, self.gradient) 


      rel_error = np.linalg.norm(new_w - self.w) /  np.linalg.norm(new_w)
      abs_error = np.linalg.norm(new_w - self.w) 

      if (rel_error < self.rel_tol) or (abs_error < self.abs_tol):        
        break

      self.w = new_w

    if self.print_time:
      print("Fitting time:" + str(time.time() - start_time) + " seconds")
  
  def predict(self, X):
    """
    Extimated prediction.

    Parameters
    ---------
    X: array of shape (n_samples, n_features)

    Returns
    -------
    y: array of shape(n_samples)
    Each row has the probability of y being of class 1, therefore we need to further process the results to 
    compare them with 0.5 so as to classify the output as 0 or 1
    """ 
    result  =   sigmoid(X,self.w)
    result = np.where(result<0.5, 0 ,1)
    return result