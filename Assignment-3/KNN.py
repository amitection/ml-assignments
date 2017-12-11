#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 02:38:29 2017

"""

import evaluation
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

def knn_classifier(X, y):

    # feature Scaling    
    sc_X = StandardScaler()
    X = sc_X.fit_transform(X)

    classifier = KNeighborsClassifier(n_neighbors= 20, metric = 'minkowski', p=2)
    
    # Evaluating the performance using 10 fold cross validation
    evaluationMetric = evaluation.EvaluationMetrics(classifier, X, y, 10, 7, "KNN")
    
    evaluationMetric.cross_validate_for_accuracy()
#    evaluationMetric.cross_validate_precision_score()
#    evaluationMetric.cross_validate_logloss()
#    evaluationMetric.cross_validate_auc_roc()
#    evaluationMetric.cross_validate_recall()
#    evaluationMetric.cross_validate_f1()
    evaluationMetric.time_to_train()
