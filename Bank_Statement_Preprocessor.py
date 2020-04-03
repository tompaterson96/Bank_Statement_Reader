
# %%
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot
from sklearn.tree import DecisionTreeClassifier

file_path = os.getcwd() + "\\" + "BankStatementDataset.csv"
fig, ax = pyplot.subplots()
max_df = 0.45
perc = 0.5

dataset = pd.read_csv(file_path)
array = dataset.values
#values = array[:,0]
#references = array[:,1]
#features = array[:,0:2]
labels = dataset['Category']
features = dataset.drop(columns = 'Category')
#values = dataset['Value']
#reference = dataset['Reference']

## ADD PIPELINE FUNCTION

text_features = ['Category']
text_transformer = Pipeline(steps = [
    ('tfidf', TfidfVectorizer(sublinear_tf=True, max_df=max_df, stop_words='english')),
    ('selector', SelectPercentile(f_classif, percentile=perc))])

num_features = ['Value']
num_transformer = Pipeline(steps = [('scaler', StandardScaler())])

preprocessor = ColumnTransformer(transformers = [
    ('num', num_transformer, num_features),
    ('txt', text_transformer, text_features)])

clf = Pipeline(steps = [
    ('preprocessing', preprocessor),
    ('classifier', DecisionTreeClassifier())])

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.1, random_state=1)

clf.fit(features_train, labels_train)
print("model score: %.3f" % clf.score(features_test, labels_test))

# feature_union = FeatureUnion(transformer_list = [
#     ('value', array[:,0]),
#     ('references', Pipeline([
#         ('tfidf', TfidfVectorizer(sublinear_tf=True, max_df=max_df, stop_words='english'))
#         ]))
#         ])

# vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=max_df,
#                                 stop_words='english')
# features_train_transformed = vectorizer.fit_transform(features_train[:,1])
# features_test_transformed  = vectorizer.transform(features_test[:,1])

# ### feature selection, because text is super high dimensional and 
# ### can be really computationally chewy as a result
# selector = SelectPercentile(f_classif, percentile=percentile)
# selector.fit(features_train_transformed, labels_train)
# features_train_transformed = selector.transform(features_train_transformed)
# features_test_transformed  = selector.transform(features_test_transformed)

#return features_train_transformed, features_test_transformed, labels_train, labels_test



# %%
