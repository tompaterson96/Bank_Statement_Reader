# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 18:52:40 2019

@author: Tom
"""
# %%
import Bank_Statement_Preprocessor
import sys
import numpy as np
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
#fig, ax = pyplot.subplots()
max_df = 0.45
perc = 0.5
features_train, features_test, labels_train, labels_test = \
    Bank_Statement_Preprocessor.preprocess(max_df = max_df, percentile=perc*100)
print(features_train[:4,:])

# Spot Check Algorithms
models = []
# logistic regression
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
# linear discriminant analysis
models.append(('LDA', LinearDiscriminantAnalysis()))
# k-nearest neighbour
models.append(('KNN', KNeighborsClassifier()))
# classification and regression trees
models.append(('CART', DecisionTreeClassifier()))
# gaussian naive bayes
models.append(('NB', GaussianNB()))
# support vector machine
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    # use statified 10-fold (k=10) cross validation
    kfold = StratifiedKFold(n_splits=2, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, features_train, labels_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Compare Algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.ylabel('Accuracy')
pyplot.ylim((0, 1))
pyplot.show()



# %%
