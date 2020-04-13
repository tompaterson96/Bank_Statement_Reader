# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 18:52:40 2019

@author: Tom
"""
# %%
from importlib import reload
import Bank_Statement_Preprocessor
reload(Bank_Statement_Preprocessor)
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
from sklearn.model_selection import GridSearchCV

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
max_df = 0.45
perc = 50
raw_features_train, raw_features_test, features_train, features_test, \
    labels_train, labels_test = Bank_Statement_Preprocessor.preprocess(max_df = max_df, percentile=perc)
features = np.concatenate((features_train, features_test))
labels = np.concatenate((labels_train, labels_test))


# classification and regression trees
params = {'splitter': ('best', 'random'), 'min_samples_split': [2,3,4], 'min_samples_leaf': [1,2,3]}

dtc = DecisionTreeClassifier()
kFolds = 10
clf = GridSearchCV(dtc, params, cv = kFolds, scoring='accuracy')
clf.fit(features, labels)

names = []
for p in clf.cv_results_['params']:
    names.append('%d%d%s' % \
        (p['min_samples_leaf'], p['min_samples_split'], p['splitter'][0].upper()))

data = np.zeros((kFolds,len(names)))  
for j in range(kFolds):
    s = 'split'+str(j)+'_test_score'
    data[j,:] = clf.cv_results_[s]

# Compare Algorithms
pyplot.boxplot(data, labels=names)
pyplot.title('Grid Search Results')
pyplot.ylabel('Accuracy')
pyplot.ylim((0.6, 1))
pyplot.show()

best_index = np.argmax(np.mean(data, axis=0))
print('Best Parameters:')
print(clf.cv_results_['params'][best_index])
print('mean = ' + str(round(np.mean(data, axis=0)[best_index], 2))\
    + ' +/- ' + str(round(np.std(data, axis=0)[best_index], 2)))

    # print('MSL: %d, MSS: %d, Split: %s = Score: %f, Error: %f' % \
    #     (p['min_samples_leaf'], p['min_samples_split'], p['splitter'], \
    #     clf.cv_results_['mean_test_score'][i], clf.cv_results_['std_test_score'][i]))


# # use statified 10-fold (k=10) cross validation
# kfold = StratifiedKFold(n_splits=2, random_state=1, shuffle=True)
# cv_results = cross_val_score(model, features_train, labels_train, cv=kfold, scoring='accuracy')
# print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# # Compare Algorithms
# pyplot.boxplot(c_vresults, labels=name)
# pyplot.title('Algorithm Comparison')
# pyplot.ylabel('Accuracy')
# pyplot.ylim((0, 1))
# pyplot.show()

# #%%

# for i in range(len(labels_test)):
#     clf = DecisionTreeClassifier()
#     clf.fit(features_train, labels_train)
#     pred = clf.predict(features_test)
#     print(accuracy_score(labels_test, pred))
#     print('Ref: %s - Pred: %s - True: %s' % \
#         (raw_features_test[i], pred[i], labels_test[i]))

# # %%


# %%
