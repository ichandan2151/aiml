# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 09:58:14 2023

@author: chand
"""

from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn.model_selection import train_test_split as split
dataset = load_iris()
X = dataset.data
Y = dataset.target
X_train, X_test, Y_train, Y_test = split(X,Y, test_size=0.2)
gnb = GaussianNB()
classifier= gnb.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)
print('Accuracy Matrics', metrics.classification_report(Y_test,y_pred))
print('Accuracy of the classifier is', metrics.accuracy_score(Y_test, y_pred))
print('confiusion matrix')
print(metrics.confusion_matrix(Y_test, y_pred))