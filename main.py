import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree, preprocessing
import pydot

dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
le = preprocessing.LabelEncoder()
balance_data = pd.read_csv(dataset_url, sep=',', header=None)
balance_data = balance_data.apply(le.fit_transform)

print("Dataset Length:: " + str(len(balance_data)))
print("Dataset Shape:: " + str(balance_data.shape))

X = balance_data.values[:, 0:14]
Y = balance_data.values[:, 14]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)

clf_gini = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=10,
                                  max_features=None, max_leaf_nodes=None, min_samples_leaf=5,
                                  min_samples_split=2, min_weight_fraction_leaf=0.0,
                                  presort=False, random_state=100, splitter='best')
clf_gini.fit(X_train, y_train)
y_pred = clf_gini.predict(X_test)
print("Accuracy is " + str(accuracy_score(y_test,y_pred)*100))
# dotfile = open("./tree.dot", 'w')
# tree.export_graphviz(clf_gini, out_file=dotfile, feature_names=['age','workclass','fnlwgt','education','education-num',
# 'marital-status','occupation','relationship','race','sex',
# 'capital-gain','capital-loss','hours-per-week','native-country'])
# dotfile.close()
#
# (graph,) = pydot.graph_from_dot_file('./tree.dot')
# graph.write_png('./tree.png')
