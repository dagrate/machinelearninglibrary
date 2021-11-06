#!/usr/bin/env python
# -*- coding: utf-8 -*-
__date__ = '20200513'
__revised__='20211031'
__author__ = 'Jeremy Charlier'
# https://betatim.github.io/posts/stop-ensemble-growth-early/
# https://machinelearningmastery.com/tune-number-size-decision-trees-xgboost-python/
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
#
#
class EarlyStopping:
  def __init__(
      self, estimator,
      xtrain, ytrain,
      xtest, ytest,
      max_n_estimators=20, 
      n_min_iterations=10,
      patience=20, mindelta=0.001):
    self.estimator = estimator
    self.max_n_estimators = max_n_estimators
    self.xtrain = xtrain
    self.ytrain = ytrain
    self.xtest = xtest
    self.ytest = ytest
    self.patience = patience
    self.mindelta = mindelta
    self.n_min_iterations = n_min_iterations
    self.train_score = []
    self.test_score = []
  # end of function __init__
  #
  def __getstate__(self):
    return self.__dict__.copy()
  # end of function __getstate__
  #
  def _make_estimator(self):
    """Make and configure a copy of the estimator attribute.
    Parameters
    ----------
    self : class-like
      class object
    """
    estimator = clone(self.estimator)
    estimator.n_estimators = 1
    return estimator
  # end of function _make_estimator
  #
  def _fit(self):
    """Fit the estimator on xtrain, ytrain.
    Fits up to the max_n_estimators iterations, measures the performance
    on xtest, ytest.
    Parameters
    ----------
    self : class-like
      class object
    """
    est = self._make_estimator()
    scores_ = []
    cntpatience = 0
    for n_est in range(1, self.max_n_estimators+1):
      if "DecisionTree" in str(self.estimator):
        est.max_depth = n_est
      else:
        est.n_estimators = n_est
      # end if
      est.fit(self.xtrain, self.ytrain)
      #
      yscoretrain = est.predict_proba(self.xtrain)
      yscoretest = est.predict_proba(self.xtest)
      ypredtrain = est.predict(self.xtrain)
      ypredtest = est.predict(self.xtest)
      #
      score = 1-roc_auc_score(self.ytest, yscoretest[:,1])
      self.estimator_ = est
      scores_.append(score)
      #
      self.train_score.append(1-roc_auc_score(self.ytrain, yscoretrain[:,1]))
      self.test_score.append(score)
      if len(scores_) > 2:
        if scores_[-1]-scores_[-2]<=self.mindelta:
          cntpatience += 1 
        # end if
      # end of
      if (n_est>self.n_min_iterations and cntpatience>=self.patience):
        return self
      # end if
    # end for
    return self
  # end of function fit
  #
  def findMaxTreesBeforeOverfit(self):
    """Compute the max nbr. of trees before overfitting.
    Parameters
    ----------
    self : class-like
      class object
    """
    self = EarlyStopping._fit(self)
    fct = np.asarray
    diff = np.abs(fct(self.test_score)[:-1]-fct(self.test_score[1:]))
    cdt = diff<self.mindelta
    cdt = np.abs(cdt-1)
    treemax = np.argmin(cdt)
    self.treemax = treemax
    return self
  # end of function findMaxTreesBeforeOverfit
  #
  def validationCurve(self):
    """Display the overfitting curve on train and test set.
    Parameters
    ----------
    self : class-like
      class object
    """
    bestiter = len(self.test_score)
    plt.figure(figsize=(6, 4))
    testline = plt.plot(self.test_score, label='test')
    plt.plot(self.train_score, '--', label='train')
    plt.vlines(self.treemax, 0, .5)
    plt.xlabel('Number of Trees')
    plt.ylabel('1-area under ROC')
    plt.legend(loc='best')
    plt.xlim(0, bestiter)
    plt.tight_layout()
    plt.show()
  # end of function validationCurve
# end of class EarlyStopping
#
if __name__ == '__main__':
  data=load_breast_cancer()
  for item in range(len(data.feature_names)):
    data.feature_names[item]=data.feature_names[item].replace(' ', '_')
  df=pd.DataFrame(data.data, columns=data.feature_names)
  #
  X_train, X_test, y_train, y_test = train_test_split(
      df,
      data.target,
      test_size=0.3,
      shuffle=True
  )
  # overfit pipeline
  estimators=[
    RandomForestClassifier(random_state=0),
    DecisionTreeClassifier(random_state=0)
  ]
  for est in estimators:
    print(est)
    mdl=EarlyStopping(
      est,
      X_train, y_train,
      X_test, y_test, 
      max_n_estimators=20)
    mdl.findMaxTreesBeforeOverfit()
    mdl.validationCurve()
  # end if
