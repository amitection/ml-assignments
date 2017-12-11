#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 01:28:00 2017

Logistic Regression 
"""

import evaluation
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def logistic_regression_classifier(X, y):

    # feature Scaling    
    sc_X = StandardScaler()
    X = sc_X.fit_transform(X)

    classifier = LogisticRegression(random_state = 0)
    
    # Evaluating the performance using 10 fold cross validation
    evaluationMetric = evaluation.EvaluationMetrics(classifier, X, y, 10, 7, "LogisticRegression")
    
    evaluationMetric.cross_validate_for_accuracy()
    evaluationMetric.cross_validate_precision_score()
    evaluationMetric.cross_validate_logloss()
    evaluationMetric.cross_validate_auc_roc()
    evaluationMetric.cross_validate_recall()
    evaluationMetric.cross_validate_f1()
    evaluationMetric.time_to_train()
