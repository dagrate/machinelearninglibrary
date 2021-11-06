#!/usr/bin/env python
# -*- coding: utf-8 -*-
__date__ = '20200801'
__revised__='20211103'
__author__ = 'Jeremy Charlier'
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
  classification_report, roc_auc_score,
  confusion_matrix, f1_score,
  roc_curve, precision_score, recall_score,
  auc, average_precision_score, 
  precision_recall_curve, accuracy_score)
#
class ModelPipeline():
  def __init__(self, estimator, xtrain, ytrain, xtest, ytest):
    self.estimator=estimator
    self.x_train=xtrain
    self.y_train=ytrain
    self.x_test=xtest
    self.y_test=ytest
    self.ypred=np.zeros((len(self.y_test)))
    self.yscore=np.zeros((len(self.y_test),2))
  # end of function __init__
  #
  def __getstate__(self):
    return self.__dict__.copy()
  # end of function __getstate__
  #
  def brierScore(y_test, yscore):
    """Compute the Brier score (0 = best, 1 = worst). 
    Parameters
    ----------
    y_test : array-like
      true target series
    yscore : array-like
      predicted scores
    Returns
    -------
    bscore : float 
      Brier score
    """
    bscore=(1/len(y_test))
    bscore*=np.sum(np.power(yscore[:,1]-y_test, 2))
    return bscore
  # end of function brierScore
  #
  def dispConfMatrixAsArray(y_test, ypred, disp=True):
    """Display and return the confusion matrix as array.
    Parameters
    ----------
    y_test : array-like
      true target series
    ypred : array-like
      predicted target series
    disp : boolean
      diplay the confusion matrix
    Returns
    -------
    confmatrix : array-like
      pandas dataframe of the confusion matrix
    """
    confmatrix=confusion_matrix(y_test,ypred)
    tn,fp,fn,tp=confmatrix.ravel()
    if disp==True:
      print('\nConfusion Matrix')
      print("%-3s" % 'TN:', "%-5s" % tn,
        "|  %-3s" % 'FP:', "%-5s" % fp)
      print("%-3s" % 'FN:', "%-5s" % fn,
        "|  %-3s" % 'TP:', "%-5s" % tp)
    return confmatrix
  # end of function dispConfMatrixAsArray
  #
  def getClassificationMetricsForPreds(self):
    """Compute metrics for classification models using the predicted class.
    Parameters
    ----------
    self : class-like
      class object
    """
    posLabel = np.unique(self.y_test)
    print("%-40s" % ("Mean Accuracy:"),
      "{:.3f}".format(self.estimator.score(self.x_test, self.y_test))
    )
    for n in posLabel:
      print("%-40s" % ("F1 Score Class " + str(n) + " :"), 
        "{:.3f}".format(
          f1_score(self.y_test,self.ypred,pos_label=n))
      )
      print("%-40s" % ("Recall Score Class "+str(n)+" :"), 
        "{:.3f}".format(
          recall_score(self.y_test,self.ypred,pos_label=n))
      )
    # end for
  # end of function getClassificationMetricsForPreds
  #
  def getClassificationMetricsForScores(self):
    """Compute metrics for classification models using the scores.
    Parameters
    ----------
    self : class-like
      class object
    """
    posLabel = np.unique(self.y_test)
    print("%-40s" % ("ROC AUC Score:"),
      "{:.3f}".format(roc_auc_score(self.y_test, self.yscore[:,1]))
    )
    print("%-40s" % ("Brier Score:"), "{:.3f}".format(
      ModelPipeline.brierScore(self.y_test, self.yscore))
    )
    for n in posLabel:
      print("%-40s" % ("Avrg Precision Score Class "+str(n)+" :"), 
        "{:.3f}".format(
          average_precision_score(self.y_test,self.yscore[:,1],pos_label=n))
      )
    # end for
  # end of function getClassificationMetricsForScores
  #
  def getClassificationMetrics(self):
    """Compute metrics for classification models.
    Parameters
    ----------
    self : class-like
      class object
    """
    print("\nModel Metrics:")
    ModelPipeline.getClassificationMetricsForPreds(self)
    if not "RidgeClassifier" in str(self.estimator):
      ModelPipeline.getClassificationMetricsForScores(self)
    # end if
    _ = ModelPipeline.dispConfMatrixAsArray(self.y_test,self.ypred,disp=True)
  # end of function getClassificationMetrics
  #
  def modelTrain(self):
    """Training pipeline.
    """
    self.estimator=self.estimator.fit(self.x_train,self.y_train)
    return self
  # end of function modelTrain
  #
  def modelPredict(self):
    """Predict pipeline.
    """
    self.ypred=self.estimator.predict(self.x_test)
    if not "RidgeClassifier" in str(self.estimator):
      self.yscore=self.estimator.predict_proba(self.x_test)
    # end if
    ModelPipeline.getClassificationMetrics(self)
    return self
  # end of function modeltrain
# end of class ModelPipeline
#
if __name__ == '__main__':
  # data pipeline
  data=load_breast_cancer()
  for item in range(len(data.feature_names)):
    data.feature_names[item]=data.feature_names[item].replace(' ', '_')
  # end for
  df=pd.DataFrame(data.data, columns=data.feature_names)
  X_train, X_test, y_train, y_test = train_test_split(
    df,
    data.target,
    test_size=0.3,
    shuffle=True
  )
  # model pipeline
  estimators=[
    RidgeClassifier(random_state=0),
    RidgeClassifier(alpha=.5, random_state=0),
    DecisionTreeClassifier(max_depth=2, random_state=0),
    DecisionTreeClassifier(max_depth=5, random_state=0),
    RandomForestClassifier(n_estimators=2, random_state=0),
    RandomForestClassifier(n_estimators=3, random_state=0),
    RandomForestClassifier(n_estimators=7, random_state=0),
  ]
  for estimator in estimators:
    print(estimator)
    mdl=ModelPipeline(
          estimator,
          X_train, y_train,
          X_test, y_test
        ).modelTrain()
    mdl=mdl.modelPredict()
  # end for
# end if