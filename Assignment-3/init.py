# -*- coding: utf-8 -*-

import config
import dataset
import DecisionTree
import LogisticRegression
import SVM
import KNN

def start_training(dataset, X, y):
    print("Training Dataset: ", dataset)
#    LogisticRegression.logistic_regression_classifier(X, y)
#    DecisionTree.decision_tree_classifier(X, y)
#    KNN.knn_classifier(X,y)
    SVM.svm_classifier(X, y)

X, y = dataset.wine_dataset(config.WINE_RED_DATA_SET)
start_training("Wine RED", X, y)

X, y = dataset.wine_dataset(config.WINE_WHITE_DATA_SET)
start_training("Wine WHITE", X, y)
    
    