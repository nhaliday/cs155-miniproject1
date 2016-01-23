import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import pca
from sklearn import cross_validation, ensemble, svm, linear_model

data = pd.read_csv('data/training_data.txt', sep='|')

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

N, k = X.shape

clf = Pipeline([('tfidf', TfidfTransformer()), ('svc', svm.LinearSVC())])
print(clf)

scores = cross_validation.cross_val_score(clf, X, y, cv=5)
print("95% confidence interval for score: {} +/- {}".format(scores.mean(), scores.std()*2))
