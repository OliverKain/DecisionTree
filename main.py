# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree, preprocessing
import pydot
import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

dataset_url = "data/vnindex-affection.csv"
le = preprocessing.LabelEncoder()
balance_data = pd.read_csv(dataset_url, sep=',', header=None)
balance_data = balance_data.apply(le.fit_transform)

print("Dataset Types:\n" + str(balance_data.dtypes))
print("Dataset Shape:\n" + str(balance_data.shape))

X = balance_data.values[:, 0:18]
Y = balance_data.values[:, 18]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=100)

clf_gini = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=5,
                                  max_features=None, max_leaf_nodes=None, min_samples_leaf=5,
                                  min_samples_split=2, min_weight_fraction_leaf=0.0,
                                  presort=False, random_state=100, splitter='best')
clf_gini.fit(X_train, y_train)
y_pred = clf_gini.predict(X_test)
print("Accuracy is " + str(accuracy_score(y_test,y_pred)*100))
dotfile = open("./tree.dot", 'w')
tree.export_graphviz(clf_gini, out_file=dotfile,
    feature_names=['Khoi ngoai mua rong (ty dong)','Tang truong GDP','Huy dong','Cho vay','Lam phat',
    'Lai suat thuc','Lai suat FED','Inflation of USA','Lai suat thuc USA',
    'FDI dang ky (ty USD)','FDI thuc hien (ty USD)','Can can thuong mai (ty USD)','Ty gia USD/VND',
    'Tang truong tin dung','Huy dong von','No cong/ GDP','No chinh phu/ GDP','No nuoc ngoai (ty USD)'],
    class_names=['GIAM', 'TANG'], filled=True, rounded=True, special_characters=True)
dotfile.close()

(graph,) = pydot.graph_from_dot_file('./tree.dot')
graph.write_png('./tree.png')
