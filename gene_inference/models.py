import sklearn, sklearn.model_selection, sklearn.metrics, sklearn.linear_model, sklearn.neural_network, sklearn.tree
import numpy as np


def lr(dataset, trials, train_size, test_size, penalty=False, **kwargs):
    scores = []

    for i in range(trials):
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(dataset.df, dataset.labels, stratify=dataset.labels, train_size=train_size, test_size=test_size, random_state=i)

        model = sklearn.linear_model.LogisticRegression()
        if penalty:
            model = sklearn.linear_model.LogisticRegression(penalty='l1', tol=0.001)
        model = model.fit(X_train, y_train)

        score = sklearn.metrics.roc_auc_score(y_test, model.predict(X_test))
        scores.append(score)
    return np.round(np.mean(scores), 2),  np.round(np.std(scores), 2)


def mlp(dataset, trials, train_size, test_size, penalty=False, **kwargs):
    # Try with only 1 layer, and with regularization
    scores = []

    for i in range(trials):
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(dataset.df, dataset.labels, stratify=dataset.labels, train_size=train_size, test_size=test_size, random_state=i)

        model = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(32,3), learning_rate_init=0.001, early_stopping=False,  max_iter=1000)
        model = model.fit(X_train, y_train)

        score = sklearn.metrics.roc_auc_score(y_test, model.predict(X_test))
        scores.append(score)
    return np.round(np.mean(scores), 2),  np.round(np.std(scores), 2)


def decision_tree(dataset, trials, train_size, test_size, penalty=False, **kwargs):
    scores = []

    for i in range(trials):
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(dataset.df, dataset.labels, stratify=dataset.labels, train_size=train_size, test_size=test_size, random_state=i)
        model = sklearn.tree.DecisionTreeClassifier()
        model = model.fit(X_train, y_train)

        score = sklearn.metrics.roc_auc_score(y_test, model.predict(X_test))
        scores.append(score)
    return np.round(np.mean(scores), 2),  np.round(np.std(scores), 2)
