# -*- coding: utf8 -*-

from sklearn import datasets
from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import numpy as np
iris = datasets.load_iris()
X, y = iris.data, iris.target



X_copy = np.copy(X)
y_copy = np.copy(y)


# pa = PassiveAggressiveClassifier()
# pa.partial_fit(X, y, classes=[0, 1, 2])


#sgd.partial_fit(X, y, classes=[0, 1, 2], sample_weight=np.ones_like(y))





#print('ova fit\n', ova.predict(X))

"""
print('ova partial fit\n', ova2.predict_proba(X))

for _ in range(1000):
    sgd.fit(X, y)

print('sgd fit\n', sgd.predict(X))


ova = OneVsRestClassifier(sgd)
for _ in range(0):
    ova.fit(X, y)
"""

n_iter = 100

sgd = SGDClassifier(loss='log', alpha=0.01)
ova_fit_first = OneVsRestClassifier(sgd)
ova2 = OneVsRestClassifier(sgd)

sgd_partial = SGDClassifier(loss='log', alpha=0.01)

ova_fit_first.fit(X, y)
for __ in range(n_iter):
    perm = np.random.permutation(y.shape[0])
    X_copy = X_copy[perm, :]
    y_copy = y_copy[perm]
    ova_fit_first.partial_fit(X_copy, y_copy)
    ova2.partial_fit(X_copy, y_copy, classes=[0, 1, 2])
    sgd_partial.partial_fit(X_copy, y_copy, classes=[0, 1, 2])

# for _ in range(n_iter):
#     perm = np.random.permutation(y.shape[0])
#     X_copy = X_copy[perm, :]
#     y_copy = y_copy[perm]
#
#
print('ova fit first\n', ova_fit_first.predict(X))

print('ova partial fit\n', ova2.predict(X))

print('sgd partial fit\n', sgd_partial.predict(X))

print('truth\n', y)



