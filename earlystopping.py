#!/usr/bin/env python
# -*- coding: utf-8 -*-
__date__ = '20200513'
__author__ = 'Jeremy Charlier'
# https://betatim.github.io/posts/stop-ensemble-growth-early/
# https://machinelearningmastery.com/tune-number-size-decision-trees-xgboost-python/

import numpy as np
import matplotlib.pyplot as plt


from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.base import clone


class EarlyStopping:
    def __init__(self, estimator,
                 max_n_estimators,
                 xtrain, ytrain,
                 xtest, ytest,
                 n_min_iterations=2, scale=1.02):
        self.estimator = estimator
        self.max_n_estimators = max_n_estimators
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xtest = xtest
        self.ytest = ytest
        self.scale = scale
        self.n_min_iterations = n_min_iterations

        self.test_score = np.empty(max_n_estimators)
        self.train_score = np.empty(max_n_estimators)


    def __getstate__(self):
        return self.__dict__.copy()


    def _make_estimator(self, append=True):
        """
        make and configure a copy of the estimator attribute.
        any estimator that has a `warm_start` option will work.
        """
        estimator = clone(self.estimator)
        estimator.n_estimators = 1
        # estimator.warm_start = True
        return estimator


    def fit(self):
        """
        fit estimator on train set
        fits up to max_n_estimators iterations and measures the performance
        on test set
        """
        est = self._make_estimator()
        scores_ = []

        for n_est in range(1, self.max_n_estimators+1):
            est.n_estimators = n_est
            est.fit(self.xtrain,self.ytrain)

            yscoretrain = est.predict_proba(self.xtrain)
            yscoretest = est.predict_proba(self.xtest)

            score = 1 - roc_auc_score(self.ytest, yscoretest[:,1])
            self.estimator_ = est
            scores_.append(score)

            self.train_score[n_est-1] = 1-roc_auc_score(self.ytrain, yscoretrain[:,1])
            self.test_score[n_est-1] = 1-roc_auc_score(self.ytest, yscoretest[:,1])

            if (n_est > self.n_min_iterations and
                score > self.scale*np.min(scores_)):
                return self

        return self

    def validationCurve(self):
        best_iter = np.argmin(self.test_score) - 1

        plt.figure()
        test_line = plt.plot(self.test_score, label='test')
        colour = test_line[-1].get_color()
        plt.plot(self.train_score, '--', color=colour, label='train')

        plt.xlabel("Number of Trees")
        plt.ylabel("1 - area under ROC")
        plt.legend(loc='best')
        plt.xlim(0, best_iter)
        plt.tight_layout()
        return best_iter


if __name__ == '__main__':
    # Random Forest Feature Importance on Breast Cancer Data
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    ntrees = 100
    clf = RandomForestClassifier(n_estimators=ntrees, random_state=42)
    clf = XGBClassifier(n_estimators=ntrees, random_state=42)

    early = EarlyStopping(
        clf, ntrees,
        X_train, y_train,
        X_test, y_test)

    early.fit()
    maxtree = early.validationCurve()
    print('max trees to avoid overfitting', maxtree)
