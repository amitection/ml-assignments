# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.feature_selection import SelectKBest, RFE
from sklearn.feature_selection import f_classif
from sklearn.svm import SVC
import matplotlib.pyplot as plt

def wine_dataset(path_to_file):
 
    dataset = pd.read_csv(path_to_file, sep=';')
    
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:,-1]
    
    
    # K-Best Feature Selection
#    print("Before feature selection: ",X.shape)
#    X_new = SelectKBest(f_classif, k=10).fit_transform(X, y)
#    print("After feature selection: ",X_new.shape)
    
    # Plot data for visualization
#    for i in range(1,10):
#        plt.scatter(X[y==i]['fixed acidity'], X[y==i]['volatile acidity'], label='Class '+str(i))
#        
#    plt.legend()
#    plt.xlabel('fixed acidity')
#    plt.ylabel('volatile acidity')
#    
#    plt.show()
    
    # Recursive Feature Elimination
    svc = SVC(kernel="linear", C=1)
    rfe = RFE(estimator=svc, n_features_to_select=6, step=1)
    rfe.fit(X, y)
    X_new = rfe.transform(X)
    
    print("Feature ranking: "+str(rfe.ranking_))
    
    return (X_new, y)


