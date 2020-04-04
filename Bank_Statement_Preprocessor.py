import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif

def preprocess(file_path = os.getcwd() + "\\" + "BankStatementDataset.csv", max_df = 0.5, percentile = 10):

    file_path = os.getcwd() + "\\" + "BankStatementDataset.csv"

    dataset = pd.read_csv(file_path)
    array = dataset.values
    #values = array[:,0]
    references = array[:,1]
    #features = array[:,0:2]
    labels = array[:,2]

    ### PROCEEDING WITH ONLY REFERENCES AS FEATURES

    raw_features_train, raw_features_test, labels_train, labels_test = \
        train_test_split(references, labels, test_size=0.1, random_state=1)

    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=max_df,
                                    stop_words='english')
    features_train_transformed = vectorizer.fit_transform(raw_features_train)
    features_test_transformed  = vectorizer.transform(raw_features_test)

    ### feature selection, because text is super high dimensional and 
    ### can be really computationally chewy as a result
    selector = SelectPercentile(f_classif, percentile=percentile)
    selector.fit(features_train_transformed, labels_train)
    features_train_transformed = selector.transform(features_train_transformed).toarray()
    features_test_transformed  = selector.transform(features_test_transformed).toarray()

    return raw_features_train, raw_features_test, features_train_transformed, \
        features_test_transformed, labels_train, labels_test

