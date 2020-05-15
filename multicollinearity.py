#!/usr/bin/env python
# -*- coding: utf-8 -*-


__date__ = '20200513'


from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from scipy.cluster import hierarchy

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


# Random Forest Feature Importance on Breast Cancer Data
def featureImportanceForMultiColinearity(clf, X_train, y_train):
    result = permutation_importance(clf, X_train, y_train, n_repeats=10,
                                    random_state=42)
    perm_sorted_idx = result.importances_mean.argsort()

    tree_importance_sorted_idx = np.argsort(clf.feature_importances_)
    tree_indices = np.arange(0, len(clf.feature_importances_)) + 0.5

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    ax1.barh(tree_indices,
            clf.feature_importances_[tree_importance_sorted_idx], height=0.7)
    ax1.set_yticklabels(data.feature_names[tree_importance_sorted_idx])
    ax1.set_yticks(tree_indices)
    ax1.set_ylim((0, len(clf.feature_importances_)))
    ax2.boxplot(result.importances[perm_sorted_idx].T, vert=False,
                labels=data.feature_names[perm_sorted_idx])
    fig.tight_layout()
    plt.show()


def handlingMultiColinearFeatures(X_train, featuresnames, threshold=1):
    '''
    Handling Multicollinear Features
    '''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    corr = spearmanr(X_train).correlation
    corr_linkage = hierarchy.ward(corr)
    dendro = hierarchy.dendrogram(corr_linkage, labels=featuresnames, ax=ax1,
                                  leaf_rotation=90)
    dendro_idx = np.arange(0, len(dendro['ivl']))

    ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
    ax2.set_xticks(dendro_idx)
    ax2.set_yticks(dendro_idx)
    ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
    ax2.set_yticklabels(dendro['ivl'])
    fig.tight_layout()
    plt.show()

    cluster_ids = hierarchy.fcluster(corr_linkage, threshold, criterion='distance')
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)
    selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]

    return selected_features


if __name__ == '__main__':
    # Random Forest Feature Importance on Breast Cancer Data
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    print("Accuracy on test data: {:.2f}".format(clf.score(X_test, y_test)))

    featureImportanceForMultiColinearity(clf, X_train, y_train)
    selected_features = handlingMultiColinearFeatures(X_train, data.feature_names)

    X_train_sel = X_train[:, selected_features]
    X_test_sel = X_test[:, selected_features]

    clf_sel = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_sel.fit(X_train_sel, y_train)
    print("Accuracy on test data with features removed: {:.2f}".format(
          clf_sel.score(X_test_sel, y_test)))

    featureImportanceForMultiColinearity(clf_sel, X_train_sel, y_train)
    _ = handlingMultiColinearFeatures(X_train_sel, data.feature_names)
