"""
Created on Mon Oct 23 22:24:47 2017

Support Vector Machines
"""
import evaluation
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def svm_classifier(X, y):
  
    # feature Scaling    
    sc_X = StandardScaler()
    X = sc_X.fit_transform(X)

    classifier = SVC(random_state = 0)
    #kernel = 'linear', 
    
    # Evaluating the performance using 10 fold cross validation
    evaluationMetric = evaluation.EvaluationMetrics(classifier, X, y, 1, 7, "SVM")
    
    evaluationMetric.cross_validate_for_accuracy()
#    evaluationMetric.cross_validate_precision_score()
#    evaluationMetric.cross_validate_logloss()
#    evaluationMetric.cross_validate_auc_roc()
#    evaluationMetric.cross_validate_recall()
#    evaluationMetric.cross_validate_f1()
#    evaluationMetric.time_to_train()
