#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 22:24:47 2017

Decision Tree Classifier
"""

import evaluation

from sklearn.tree import DecisionTreeClassifier

def decision_tree_classifier(X, y) :

    classifier = DecisionTreeClassifier(criterion = 'entropy')    
    
    # Evaluating the performance using 10 fold cross validation
    evaluationMetric = evaluation.EvaluationMetrics(classifier, X, y, 10, 7, "DecisionTrees")
    
    evaluationMetric.cross_validate_for_accuracy()
    evaluationMetric.cross_validate_precision_score()
    evaluationMetric.cross_validate_logloss()
    evaluationMetric.cross_validate_auc_roc()
    evaluationMetric.cross_validate_recall()
    evaluationMetric.cross_validate_f1()
    evaluationMetric.time_to_train()
