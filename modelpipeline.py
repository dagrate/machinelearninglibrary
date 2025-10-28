#!/usr/bin/env python
# -*- coding: utf-8 -*-
__date__ = '20200801'
__revised__ = '20251028'  # Added optimal F1 threshold finding
__author__ = 'Jeremy Charlier'


import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

# Core sklearn imports
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator

# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier

# Metrics
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    accuracy_score,
    brier_score_loss,
    precision_recall_curve,  # <-- Import added
)


class ModelPipeline:
    """
    A reusable pipeline for training, predicting, and evaluating
    binary classification models.

    This class encapsulates the train/test data, the estimator,
    and the corresponding predictions and metrics.
    """

  
    def __init__(
        self,
        estimator: BaseEstimator,
        xtrain: pd.DataFrame,
        ytrain: pd.Series,
        xtest: pd.DataFrame,
        ytest: pd.Series,
    ):
        """
        Initialize the ModelPipeline.

        Parameters
        ----------
        estimator : BaseEstimator
            A scikit-learn compatible classifier instance.
        xtrain : pd.DataFrame
            Training features.
        ytrain : pd.Series
            Training target.
        xtest : pd.DataFrame
            Test/Validation features.
        ytest : pd.Series
            Test/Validation target.
        """
        self.estimator = estimator
        self.x_train = xtrain
        self.y_train = ytrain
        self.x_test = xtest
        self.y_test = ytest

        # Initialize containers for predictions and scores
        self.ypred_train: np.ndarray = None
        self.ypred_test: np.ndarray = None
        self.yscore_train: np.ndarray = None  # Scores for positive class
        self.yscore_test: np.ndarray = None  # Scores for positive class

        # Container for final metrics
        self.metrics: Dict[str, Dict[str, Any]] = {}

        # Optimal threshold
        self.optimal_f1_threshold: Optional[float] = None

  
    def __getstate__(self) -> dict:
        """Return state of instance for pickling."""
        return self.__dict__.copy()

  
    def get_model(self) -> BaseEstimator:
        """
        Returns the trained estimator.

        Returns
        -------
        BaseEstimator
            The fitted scikit-learn model.
        """
        return self.estimator

  
    def modelTrain(self) -> 'ModelPipeline':
        """
        Train the model on the training data.

        Returns
        -------
        ModelPipeline
            The instance itself (self) for method chaining.
        """
        print(f"Training {self.estimator.__class__.__name__}...")
        self.estimator.fit(self.x_train, self.y_train)
        return self

  
    def modelPredict(self) -> 'ModelPipeline':
        """
        Generate predictions for both training and test sets.

        This method populates:
        - self.ypred_train, self.ypred_test (predicted classes)
        - self.yscore_train, self.yscore_test (predicted probabilities/scores
          for the positive class, if available)

        Returns
        -------
        ModelPipeline
            The instance itself (self) for method chaining.
        """
        # Generate class predictions
        self.ypred_train = self.estimator.predict(self.x_train)
        self.ypred_test = self.estimator.predict(self.x_test)

        # Generate probability scores (if possible)
        # We robustly check for `predict_proba`
        if hasattr(self.estimator, 'predict_proba'):
            # Store only the probability of the positive class (class 1)
            self.yscore_train = self.estimator.predict_proba(self.x_train)[:, 1]
            self.yscore_test = self.estimator.predict_proba(self.x_test)[:, 1]
        else:
            # Handle models without `predict_proba` (e.g., RidgeClassifier, SVM)
            # Their scores (from `decision_function`) are not probabilities
            # and cannot be used for probability-based metrics like Brier score.
            print(
                f"Warning: Estimator {self.estimator.__class__.__name__} "
                "does not have 'predict_proba'. "
                "Score-based metrics (ROC AUC, AP, Brier) will be skipped."
            )
            # `yscore` remains None

        return self

  
    def _compute_metrics(
        self, y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray
    ) -> Dict[str, Any]:
        """
        Private helper to compute a dictionary of classification metrics.

        Parameters
        ----------
        y_true : pd.Series
            True target labels.
        y_pred : np.ndarray
            Predicted class labels.
        y_score : np.ndarray
            Predicted probabilities for the positive class (or None).

        Returns
        -------
        Dict[str, Any]
            A dictionary containing all computed metrics.
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(
                y_true, y_pred, zero_division=0
            ),
        }

        # Score-based metrics (only compute if scores are available)
        if y_score is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_score)
            metrics['average_precision'] = average_precision_score(y_true, y_score)
            metrics['brier_score'] = brier_score_loss(y_true, y_score)

        return metrics

  
    def _print_report(self):
        """Private helper to neatly print the metrics for train and test sets."""
        if not self.metrics:
            print("Metrics have not been generated. Call generate_report() first.")
            return

        for dataset_type, metrics in self.metrics.items():
            print(f"\n{'---' * 10}")
            print(f" {dataset_type.UPPER()} SET METRICS (at 0.5 threshold)")
            print(f"{'---' * 10}")

            print(f"{'Accuracy:':<20} {metrics['accuracy']:.4f}")
            print(f"{'F1 Score:':<20} {metrics['f1_score']:.4f}")
            print(f"{'Precision:':<20} {metrics['precision']:.4f}")
            print(f"{'Recall:':<20} {metrics['recall']:.4f}")

            # Print score-based metrics if they exist
            if 'roc_auc' in metrics:
                print(f"{'ROC AUC:':<20} {metrics['roc_auc']:.4f}")
                print(
                    f"{'Avg. Precision (AP):':<20} {metrics['average_precision']:.4f}"
                )
                print(f"{'Brier Score:':<20} {metrics['brier_score']:.4f}")

            print("\nConfusion Matrix:")
            tn, fp, fn, tp = metrics['confusion_matrix'].ravel()
            print(f" {'TN:':<4} {tn:<5} | {'FP:':<4} {fp:<5}")
            print(f" {'FN:':<4} {fn:<5} | {'TP:':<4} {tp:<5}")

            print("\nClassification Report:")
            print(metrics['classification_report'])

  
    def generate_report(
        self, print_report: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Computes all classification metrics for train and test sets.

        This method populates `self.metrics` and can optionally
        print a formatted report.

        Parameters
        ----------
        print_report : bool, optional
            If True (default), prints a formatted report to the console.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            A nested dictionary containing metrics for 'train' and 'test' sets.
        """
        if self.ypred_train is None or self.ypred_test is None:
            raise ValueError(
                "Predictions not found. Call modelPredict() before generate_report()."
            )

        print("Computing metrics...")
        self.metrics['train'] = self._compute_metrics(
            self.y_train, self.ypred_train, self.yscore_train
        )
        self.metrics['test'] = self._compute_metrics(
            self.y_test, self.ypred_test, self.yscore_test
        )

        if print_report:
            self._print_report()

        return self.metrics

  
    def get_feature_importance(
        self, top_n: Optional[int] = None, display: bool = True
    ) -> pd.DataFrame:
        """
        Extracts and displays feature importances from the trained estimator.

        Supports estimators with `.feature_importances_` (e.g., trees, forests)
        and `.coef_` (e.g., linear models).

        Parameters
        ----------
        top_n : int, optional
            If set, returns only the top N most important features.
            If None (default), returns all features.
        display : bool, optional
            If True (default), prints the resulting DataFrame to the console.

        Returns
        -------
        pd.DataFrame
            A DataFrame with features and their importance, sorted
            in descending order of importance. Returns an empty
            DataFrame if the estimator doesn't support feature importance.
        """
        if not hasattr(self.estimator, 'fit'):
            print("Estimator not found or not trained. Please run modelTrain() first.")
            return pd.DataFrame()

        feature_names = self.x_train.columns
        importance_df = pd.DataFrame()

        if hasattr(self.estimator, 'feature_importances_'):
            # --- Handle Tree-based models ---
            importances = self.estimator.feature_importances_
            importance_df = pd.DataFrame(
                {'feature': feature_names, 'importance': importances}
            )
            importance_df = importance_df.sort_values(
                by='importance', ascending=False
            )

        elif hasattr(self.estimator, 'coef_'):
            # --- Handle Linear models ---
            coefs = self.estimator.coef_

            # Squeeze a (1, n_features) array into (n_features,) for binary case
            if coefs.ndim > 1 and coefs.shape[0] == 1:
                importances_raw = coefs[0]
            elif coefs.ndim == 1:
                importances_raw = coefs
            else:
                # Handle multiclass case by averaging abs value across classes
                print(
                    f"Warning: Multiclass coefficients (shape={coefs.shape}). "
                    "Using mean absolute coefficient as importance."
                )
                importances_raw = np.mean(np.abs(coefs), axis=0)

            importance_df = pd.DataFrame(
                {'feature': feature_names, 'coefficient': importances_raw}
            )
            # Use absolute value for ranking importance
            importance_df['importance'] = np.abs(importances_raw)
            importance_df = importance_df.sort_values(
                by='importance', ascending=False
            )

        else:
            print(
                f"Warning: Estimator {self.estimator.__class__.__name__} "
                "does not provide '.feature_importances_' or '.coef_'."
            )
            return pd.DataFrame()

        # Reset index for clean display
        importance_df = importance_df.reset_index(drop=True)

        # Handle top_n
        if top_n:
            importance_df = importance_df.head(top_n)

        if display and not importance_df.empty:
            print(f"\nFeature Importance (Top {top_n or 'All'}):")
            print(importance_df.to_string())

        return importance_df

  
    def find_optimal_f1_threshold(
        self, data_source: str = 'train', display: bool = True
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Finds the optimal probability threshold to maximize the F1 score.

        It is highly recommended to run this on the 'train' set to avoid
        data leakage and get an unbiased estimate on the 'test' set.

        Parameters
        ----------
        data_source : str, optional
            The data set to use for finding the threshold.
            Must be 'train' (default) or 'test'.
        display : bool, optional
            If True (default), prints the findings.

        Returns
        -------
        Tuple[Optional[float], Optional[float]]
            (best_threshold, best_f1_score)
            Returns (None, None) if scores are not available.
        """
        if not hasattr(self.estimator, 'predict_proba'):
            print(
                f"Cannot find optimal threshold: Estimator "
                f"{self.estimator.__class__.__name__} does not "
                "support 'predict_proba'."
            )
            return None, None

        if data_source == 'train':
            y_true = self.y_train
            y_score = self.yscore_train
        elif data_source == 'test':
            y_true = self.y_test
            y_score = self.yscore_test
        else:
            raise ValueError("data_source must be 'train' or 'test'")

        if y_score is None:
            print(
                f"Scores for '{data_source}' set not found. "
                "Run modelPredict() first."
            )
            return None, None

        precision, recall, thresholds = precision_recall_curve(y_true, y_score)

        # Calculate F1 score for each threshold, avoiding division by zero
        # We use [:-1] as 'thresholds' is one element shorter
        numerator = 2 * precision[:-1] * recall[:-1]
        denominator = precision[:-1] + recall[:-1]
        f1_scores = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(denominator),
            where=denominator != 0,
        )

        # Find the best
        best_idx = np.argmax(f1_scores)
        best_f1 = f1_scores[best_idx]
        best_threshold = thresholds[best_idx]

        # Store threshold if found on training data
        if data_source == 'train':
            self.optimal_f1_threshold = best_threshold

        if display:
            print(f"\nOptimal F1 Threshold Analysis (on '{data_source}' set):")
            print(f"  Best F1 Score: {best_f1:.4f}")
            print(f"  At Threshold:  {best_threshold:.4f}")

        return best_threshold, best_f1
# end of class ModelPipeline
#
#
#
if __name__ == '__main__':
    # 1. Data Pipeline
    print("Loading data...")
    data = load_breast_cancer()
    feature_names = [name.replace(' ', '_') for name in data.feature_names]

    df = pd.DataFrame(data.data, columns=feature_names)
    X_train, X_test, y_train, y_test = train_test_split(
        df, data.target, test_size=0.3, shuffle=True, random_state=42
    )

    # 2. Model Pipeline
    estimators: List[BaseEstimator] = [
        RidgeClassifier(random_state=0),
        DecisionTreeClassifier(max_depth=5, random_state=0),
        RandomForestClassifier(n_estimators=50, random_state=0),
    ]

    trained_models = []

    for estimator in estimators:
        print(f"\n{'='*25}")
        print(f" RUNNING PIPELINE FOR: {estimator.__class__.__name__} ")
        print(f"{'='*25}")

        # Instantiate pipeline
        mdl = ModelPipeline(
            estimator, X_train, y_train, X_test, y_test
        )

        # Chain methods: train -> predict
        mdl.modelTrain().modelPredict()
        
        # Generate and print the report (based on default 0.5 threshold)
        metrics_results = mdl.generate_report(print_report=True)

        # Get and display top 10 features
        feature_importance_df = mdl.get_feature_importance(top_n=10, display=True)

        # Get the trained model
        trained_model = mdl.get_model()
        trained_models.append(trained_model)

        # --- DEMONSTRATE OPTIMAL THRESHOLDING (if possible) ---
        if hasattr(trained_model, 'predict_proba'):
            print("\n--- Optimal F1 Threshold Search ---")
            # Find threshold on the TRAIN set to prevent data leakage
            opt_thresh, best_f1 = mdl.find_optimal_f1_threshold(
                data_source='train', display=True
            )
            
            if opt_thresh is not None:
                print(
                    f"\n--- Applying Optimal Threshold ({opt_thresh:.4f}) to TEST set ---"
                )
                # Now, apply this single threshold to the test set
                yscore_test_positive = mdl.yscore_test
                ypred_test_new = (yscore_test_positive >= opt_thresh).astype(int)
                
                # Compute and print new test metrics
                print("New Test Metrics using optimal threshold:")
                print(f"  F1 Score: {f1_score(mdl.y_test, ypred_test_new):.4f}")
                print(f"  Recall:   {recall_score(mdl.y_test, ypred_test_new):.4f}")
                print(f"  Precision:{precision_score(mdl.y_test, ypred_test_new):.4f}")
                
                print("\nNew Test Confusion Matrix:")
                tn, fp, fn, tp = confusion_matrix(
                    mdl.y_test, ypred_test_new
                ).ravel()
                print(f"   {'TN:':<4} {tn:<5} | {'FP:':<4} {fp:<5}")
                print(f"   {'FN:':<4} {fn:<5} | {'TP:':<4} {tp:<5}")
        else:
            print(
                "\nSkipping F1 threshold search: Model does not support 'predict_proba'."
            )
        # -------------------------------------------------------

        print(f"\nPipeline complete for: {trained_model}")
        print(f"Test Set F1 Score (default 0.5 thresh): {metrics_results['test']['f1_score']:.4f}")

    print(f"\nAll models trained. Total models: {len(trained_models)}")
