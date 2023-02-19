import math
import os

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pandas as pd
import os
"""
iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(X_train, X_test, y_train, y_test)
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
"""


f = open(os.getcwd()+"/T2DV2/mainColumn.csv", errors='ignore')
dataSubcol = pd.read_csv(f)
tableName = list(set(dataSubcol.iloc[:,-1]))
tableIndex = dataSubcol.iloc[:,-1]
train = math.floor(len(tableName)*0.2)
train_tables, test_tables = tableName[0:train], tableName[train+1:len(tableName)-1]
train_data = dataSubcol.loc[dataSubcol.iloc[:, dataSubcol.shape[1] - 1].isin(train_tables)]
test_data = dataSubcol.loc[dataSubcol.iloc[:, dataSubcol.shape[1] - 1].isin(test_tables)]
train_features, train_subject = train_data.iloc[:, :-2].values, train_data.iloc[:, -2].values
test_features, test_subject = test_data.iloc[:, :-2].values, test_data.iloc[:, -2].values
# print(train_features, test_features, train_subject, test_subject)
clf = svm.SVC(kernel='linear')#
clf.fit(train_features, train_subject)
y_pred = clf.predict(test_features)
accuracy = accuracy_score(test_subject, y_pred)
print("Accuracy:", accuracy)
# 随机选取test train table
#a = [2, 'c']
#ith_column = 2
#selected_cols = df.loc[df.iloc[:, ith_column].isin(a)]
