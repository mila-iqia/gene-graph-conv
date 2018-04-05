import sys
import sklearn, sklearn.model_selection, sklearn.metrics, sklearn.linear_model, sklearn.neural_network, sklearn.tree
import numpy as np
import matplotlib, matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

def lr(df, labels, trials, samples, train_ratio=0.5):
    scores = []
    for i in range(trials):
        X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(df[:samples], labels[:samples], stratify=labels[:samples], train_size=train_ratio, random_state=i)

        model = sklearn.linear_model.LogisticRegression()
        model = model.fit(X_train, y_train)

        score = sklearn.metrics.roc_auc_score(y_test, model.predict(X_test))
        scores.append(score)
    return np.round(np.mean(scores), 2),  np.round(np.std(scores),2)


def mlp(df, labels, trials, samples, train_ratio=0.5):
    # Try with only 1 layer, and with regularization
    scores = []
    for i in range(trials):
        X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(df[:samples], labels[:samples], stratify=labels[:samples], train_size=train_ratio, random_state=i)

        model = sklearn.neural_network.MLPClassifier()
        model = model.fit(X_train, y_train)

        score = sklearn.metrics.roc_auc_score(y_test, model.predict(X_test))
        scores.append(score)
    return np.round(np.mean(scores), 2),  np.round(np.std(scores),2)

def decision_tree(df, labels, trials, samples, train_ratio=0.5):
    scores = []
    for i in range(trials):
        X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(df[:samples], labels[:samples], stratify=labels[:samples], train_size=train_ratio, random_state=i)

        model = sklearn.tree.DecisionTreeClassifier()
        model = model.fit(X_train, y_train)

        score = sklearn.metrics.roc_auc_score(y_test, model.predict(X_test))
        scores.append(score)
    return np.round(np.mean(scores), 2),  np.round(np.std(scores),2)
