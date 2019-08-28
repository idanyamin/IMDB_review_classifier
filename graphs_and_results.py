"""
author: Idan Yamin
"""
import numpy as np
import pandas as pd
import sklearn.model_selection
import sklearn.linear_model
import sklearn.model_selection as model_selection
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier


NUM_FEATURES = 17500
TRAIN_PERCENTAGE = 0.8
LAMBDA_STEP_SIZE = 40
LAMBDA_STEP_PATH = 10000
LASSO_STEP_PATH = 0.005
OPTIMAL_LEAFS_MUM = 1024
OPTIMAL_DEPTH = 170
OPTIMAL_NUMBER_OF_TREES = 120
OPTIMAL_NUM_OF_FEATURES = 1600

CLASSIFIER = None


def zero_one_loss(classifier, x_train, x_test, y_train, y_test):
    """

    :param classifier: sklearn classifier
    :param x_train: training data
    :param x_test: test data
    :param y_train: training labels
    :param y_test: testing labels
    :return: the zero one loss of the classifier in training data
    """
    vec = CountVectorizer()
    x_train_copy = x_train.copy()
    x_test_copy = x_test.copy()
    x_train_copy = vec.fit_transform(x_train_copy)
    x_test_copy = vec.transform(x_test_copy)

    my_classifier = classifier(solver='sag', max_iter=50).fit(x_train_copy, y_train)
    return np.mean(np.abs(my_classifier.predict(x_test_copy)-y_test))


def feature_selection_graph(classifier, x_train, x_test, y_train, y_test, cross_validation=False):
    """

    :param classifier: sklearn classifier
    :param x_train: training data
    :param x_test: test data
    :param y_train: training labels
    :param y_test: testing labels
    :param cross_validation: if true use cross validation otherwise use test set.
    :return: Draw a graph of Mean 0-1 Loss as a function of the number of features.
    """
    # init the axis
    x_axis = []
    train_y_axis = []
    test_y_axis = []

    # train the classifier using increasing number of most commonly used words.
    for i in range(100, 45000, 7500):
        x_train_copy = x_train.copy()
        y_train_copy = y_train.copy()
        x_test_copy = x_test.copy()
        y_test_copy = y_test.copy()
        x_axis.append(i)
        vec = CountVectorizer(max_features=i)
        x_train_copy = vec.fit_transform(x_train_copy)
        x_test_copy = vec.transform(x_test_copy)
        my_classifier = classifier(solver='lbfgs').fit(x_train_copy, y_train_copy)
        if cross_validation:
            y_value = 1 - np.mean(sklearn.model_selection.cross_val_score(classifier(solver='lbfgs'), x_train_copy, y_train_copy, cv=10))
            test_y_axis.append(y_value)
        else:
            test_y_axis.append(np.mean(np.abs(my_classifier.predict(x_test_copy)-y_test_copy)))

        train_y_axis.append((np.mean(np.abs(my_classifier.predict(x_train_copy)-y_train_copy))))

    # plot the graph
    plt.plot(x_axis, test_y_axis)
    plt.plot(x_axis, train_y_axis)
    plt.xlabel('Number of Features')
    plt.ylabel('Mean 0-1 Loss Error')
    plt.title('Mean 0-1 Loss as a function of the number of features')
    print('x axis: ' + str(x_axis))
    print('y-test axis: ' + str(test_y_axis))
    print('y-train axis: ' + str(train_y_axis))
    plt.legend(('test', 'train'))
    plt.show()


def linear_classifier(x_train, x_test, y_train, y_test):
    """
    :param x_train: training data
    :param x_test: testing data
    :param y_train: training label
    :param y_test: testing label
    :return: mean 0-1 loss
    """
    print('It might take a couple of minutes')
    x_test_copy, x_train_copy = bag_of_words(x_test, x_train, NUM_FEATURES)
    my_classifier = sklearn.linear_model.LinearRegression().fit(x_train_copy, y_train)
    print('Linear classifier mean 0-1 loss: ' + str(np.mean(np.abs(my_classifier.predict(x_test_copy) - y_test))))


def bag_of_words(x_test, x_train, num_of_features):
    """
    :param x_test: testing set
    :param x_train: training set
    :param num_of_features: the number of most commonly features
    :return: a bag of words representation of x test and x train.
    """
    vec = CountVectorizer(max_features=num_of_features)
    x_train_copy = x_train.copy()
    x_test_copy = x_test.copy()
    x_train_copy = vec.fit_transform(x_train_copy)
    x_test_copy = vec.transform(x_test_copy)
    return x_test_copy, x_train_copy


def plot_ridge_error_as_function_of_lambda(x_train, x_test, y_train, y_test):
    """
    :param x_train: training data
    :param x_test: testing data
    :param y_train: training labels
    :param y_test: testing labels
    plot a graph of mean 0-1 Loss as a function of lambda
    """
    x_axis = []
    y_train_axis = []
    y_test_axis = []
    x_test_copy, x_train_copy = bag_of_words(x_test, x_train, NUM_FEATURES)
    # calculate x axis and y axis
    for i in range(0, 11):
        x_axis.append(i * LAMBDA_STEP_SIZE)
        my_classifier = sklearn.linear_model.RidgeClassifier(alpha=i * LAMBDA_STEP_SIZE).fit\
            (x_train_copy, y_train)
        y_test_axis.append(np.mean(np.abs(my_classifier.predict(x_test_copy) - y_test)))
        y_train_axis.append(np.mean(np.abs(my_classifier.predict(x_train_copy) - y_train)))

    # plot the graph
    plt.plot(x_axis, y_test_axis)
    plt.plot(x_axis, y_train_axis)
    plt.xlabel('Value of lambda')
    plt.ylabel('Mean 0-1 Loss')
    plt.title('Mean 0-1 Loss as a function of lambda')
    print('x-axis: ' + str(x_axis))
    print('y-train-axis: ' + str(y_train_axis))
    print('y-test-axis: ' + str(y_test_axis))
    plt.legend(('test', 'train'))
    plt.show()


def plot_ridge_regularization_path(x_train, y_train):
    """
    :param x_train: training data
    :param y_train: testing data
    plot regularization path of ridge regression using only 10 most commonly used words.
    """
    # create bag of words
    vec = CountVectorizer(max_features=10)
    x_train_copy = x_train.copy()
    x_train_copy = vec.fit_transform(x_train_copy)
    # init axis
    x_axis = []
    y_wight = []
    # calculate x axis a nd y axis
    for i in range(0, 300):
        x_axis.append(i * LAMBDA_STEP_PATH)
        my_classifier = sklearn.linear_model.Ridge(alpha=i * LAMBDA_STEP_PATH, fit_intercept=False).fit(x_train_copy, y_train)
        y_wight.append(my_classifier.coef_)

    # plot the graph
    plt.plot(x_axis, y_wight)
    plt.xlabel('Value of lambda')
    plt.ylabel('Weights')
    plt.title('Weights as a function of lambda')
    print('x-axis: ' + str(x_axis))
    plt.show()


def boosting_optimal_tree(x_train, x_test, y_train, y_test):
    """
    :param x_train: training data
    :param x_test: testing data
    :param y_train: training labels
    :param y_test: testing labels
    print test and train score of the optimal classifier.
    """
    x_test_copy, x_train_copy = bag_of_words(x_test, x_train, NUM_FEATURES)
    adb = AdaBoostClassifier(base_estimator=RandomForestClassifier(bootstrap=True, n_estimators=OPTIMAL_NUMBER_OF_TREES,
                                            max_leaf_nodes=OPTIMAL_LEAFS_MUM, max_depth=OPTIMAL_DEPTH,
                                            random_state=1, max_features='log2', n_jobs=4),
                             n_estimators=50, learning_rate=1)
    adb = adb.fit(x_train_copy, y_train)
    print('test score: ' + str(adb.score(x_test_copy, y_test)))
    print('train score: ' + str(adb.score(x_train_copy, y_train)))


def random_forest_error_as_a_function_of_trees_num(x_train, x_test, y_train, y_test, iterations, step_size, max_features):
    """
    :param x_train: training data
    :param x_test: testing data
    :param y_train: training labels
    :param y_test: testing labels
    :param iterations: number of points to plot
    :param step_size: distance between points
    :param max_features: 'sqrt' or 'log2'
    plot mean 0-1 error as a function of the number of trees.
    """
    x_test_copy, x_train_copy = bag_of_words(x_test, x_train, NUM_FEATURES)
    # init axis
    x_axis = []
    y_axis_test = []
    y_axis_train = []
    # calculate x-axis and y-axis
    for i in range(0, iterations):
        x = int(max(i * step_size, 1))
        x_axis.append(x)
        classifier = RandomForestClassifier(bootstrap=True, n_estimators=x,
                                            max_leaf_nodes=OPTIMAL_LEAFS_MUM, max_depth=OPTIMAL_DEPTH,
                                            random_state=1, max_features=max_features, n_jobs=4)
        classifier = classifier.fit(x_train_copy, y_train)
        y_axis_test.append(1 - classifier.score(x_test_copy, y_test))
        y_axis_train.append(1 - classifier.score(x_train_copy, y_train))

    # plot the graph
    plt.plot(x_axis, y_axis_test)
    plt.plot(x_axis, y_axis_train)
    print('x_axis: ' + str(x_axis))
    print('y_test_axis: ' + str(y_axis_test))
    print('y_train_axis: ' + str(y_axis_train))
    plt.title('mean 0-1 error as a function of the number of trees')
    plt.xlabel('Trees number')
    plt.ylabel('Mean 0-1 error')
    plt.legend(('test', 'train'))
    plt.show()


def random_forest_error_as_a_function_of_features_num(x_train, x_test, y_train, y_test, iterations, step_size, max_features):
    """
    :param x_train: x training data
    :param x_test: x testing data
    :param y_train: training labels
    :param y_test: testing labels
    :param iterations: number of points to plot
    :param step_size: distance between points
    :param max_features: 'sqrt' or 'log2'
    plot mean 0-1 error as a function number of features
    """
    # init x-axis
    x_axis = []
    y_axis_test = []
    y_axis_train = []
    start = 4
    # calculate axis
    for i in range(start, iterations + start):
        x = int(max(i * step_size, 2))
        x_test_copy, x_train_copy = bag_of_words(x_test, x_train, x)
        x_axis.append(x)
        classifier = RandomForestClassifier(bootstrap=True, n_estimators=1,
                                            max_leaf_nodes=OPTIMAL_LEAFS_MUM, max_depth=OPTIMAL_DEPTH,
                                            random_state=1, max_features=max_features, n_jobs=4)
        classifier = classifier.fit(x_train_copy, y_train)
        y_axis_test.append(1 - classifier.score(x_test_copy, y_test))
        y_axis_train.append(1 - classifier.score(x_train_copy, y_train))

    # plot graph
    plt.plot(x_axis, y_axis_test)
    plt.plot(x_axis, y_axis_train)
    print('x_axis: ' + str(x_axis))
    print('y_test_axis: ' + str(y_axis_test))
    print('y_train_axis: ' + str(y_axis_train))
    plt.title('mean 0-1 error as a function number of features')
    plt.xlabel('Features number')
    plt.ylabel('Mean 0-1 error')
    plt.legend(('test', 'train'))
    plt.show()


def random_forest_error_as_a_function_of_depth(x_train, x_test, y_train, y_test, iterations, step_size, max_features):
    """
    :param x_train: training data
    :param x_test: testing data
    :param y_train: train labels
    :param y_test: test labels
    :param iterations: number of points
    :param step_size: distance between points
    :param max_features: 'sqrt' or 'log2'
    plot mean 0-1 error as a function tree's depth
    """
    x_test_copy, x_train_copy = bag_of_words(x_test, x_train, NUM_FEATURES)
    # init axis
    x_axis = []
    y_axis_test = []
    y_axis_train = []
    # calculate axis
    for i in range(0, iterations):
        x = int(max(i * step_size, 2))
        x_axis.append(x)
        classifier = RandomForestClassifier(bootstrap=True, n_estimators=1, max_leaf_nodes=OPTIMAL_LEAFS_MUM, max_depth=x, random_state=1, max_features=max_features, n_jobs=4)
        classifier = classifier.fit(x_train_copy, y_train)
        y_axis_test.append(1 - classifier.score(x_test_copy, y_test))
        y_axis_train.append(1 - classifier.score(x_train_copy, y_train))

    # plot graph
    plt.plot(x_axis, y_axis_test)
    plt.plot(x_axis, y_axis_train)
    print('x_axis: ' + str(x_axis))
    print('y_test_axis: ' + str(y_axis_test))
    print('y_train_axis: ' + str(y_axis_train))
    plt.title('mean 0-1 error as a function tree\'s depth')
    plt.xlabel('Depth')
    plt.ylabel('Mean 0-1 error')
    plt.legend(('test', 'train'))
    plt.show()


def random_forest_error_as_a_function_of_leafs(x_train, x_test, y_train, y_test, iterations, step_size, max_features):
    """

    :param x_train: training data
    :param x_test: testing data
    :param y_train: training labels
    :param y_test: testing labels
    :param iterations: number of points
    :param step_size: distance between points
    :param max_features: 'sqrt' or 'log2'
    plot mean 0-1 error as a function of the number of leafs
    """
    x_test_copy, x_train_copy = bag_of_words(x_test, x_train, NUM_FEATURES)
    # init axis
    x_axis = []
    y_axis_test = []
    y_axis_train = []
    # calculate axis
    for i in range(0, iterations):
        x = int(max(i * step_size, 2))
        x_axis.append(x)
        classifier = RandomForestClassifier(bootstrap=True, n_estimators=1, max_leaf_nodes=x, random_state=1, max_features=max_features, n_jobs=4)
        classifier = classifier.fit(x_train_copy, y_train)
        y_axis_test.append(1 - classifier.score(x_test_copy, y_test))
        y_axis_train.append(1 - classifier.score(x_train_copy, y_train))

    # plot graph
    plt.plot(x_axis, y_axis_test)
    plt.plot(x_axis, y_axis_train)
    print('x_axis: ' + str(x_axis))
    print('y_test_axis: ' + str(y_axis_test))
    print('y_train_axis: ' + str(y_axis_train))
    plt.title('mean 0-1 error as a function of the number of leaves')
    plt.xlabel('Number of leaves')
    plt.ylabel('Mean 0-1 error')
    plt.legend(('test', 'train'))
    plt.show()
    print('score: ' + str(classifier.score(x_test_copy, y_test)))


def adaboost(x_train, x_test, y_train, y_test):
    """
    :param x_train: training data
    :param x_test: testing data
    :param y_train: train labels
    :param y_test: test labels
    print mean 0-1 loss of boosted decision tree classifier.
    """
    print('It might take a while')
    x_test_copy, x_train_copy = bag_of_words(x_test, x_train, NUM_FEATURES)
    adb = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_leaf_nodes=10),
                             n_estimators=150, learning_rate=0.5)
    adb.fit(x_train_copy, y_train)
    print('mean 0-1 loss: ' + str(1 - adb.score(x_test_copy, y_test)))


def decision_stump_loss(x_train, x_test, y_train, y_test):
    """
    :param x_train: training data
    :param x_test: testing data
    :param y_train: training labels
    :param y_test: testing labels
    :return: 0-1 loss of decision tree classifier.
    """
    x_test_copy, x_train_copy = bag_of_words(x_test, x_train, NUM_FEATURES)
    classifier = DecisionTreeClassifier(max_leaf_nodes=10, random_state=1).fit(x_train_copy, y_train)
    return np.mean(np.abs(classifier.predict(x_test_copy)-y_test))


def plot_lasso_regularization_path(x_train, y_train):
    """
    :param x_train: training data
    :param y_train: training labels
    Weights as a function of lambda using only 10 features. (regularization path).
    """
    vec = CountVectorizer(max_features=10)
    x_train_copy = x_train.copy()
    x_train_copy = vec.fit_transform(x_train_copy)
    x_axis = []
    y_wight = []
    for i in range(0, 300):
        x_axis.append(i * LASSO_STEP_PATH)
        my_classifier = sklearn.linear_model.Lasso(alpha=i * LASSO_STEP_PATH,
                                                   fit_intercept=False).fit(x_train_copy, y_train)
        y_wight.append(my_classifier.coef_)

    plt.plot(x_axis, y_wight)
    plt.xlabel('Value of lambda')
    plt.ylabel('Weights')
    plt.title('Weights as a function of lambda')
    print('x-axis: ' + str(x_axis))
    plt.show()


def get_data(test_size=0.3):
    """
    opening IMDB_Dataset.csv and divides the data into 4 ndarrays:
    x train, y train, x test, y test.
    :param test_size: test size percentage (the default is 0.3).
    :return: x_train, x_test, y_train, y_test
    """
    data = pd.read_csv("IMDB_Dataset.csv")
    reviews = data["review"]
    labels = data["sentiment"]

    labels = np.unique(labels, return_inverse=True)[1]
    x_train, x_test, y_train, y_test = model_selection.train_test_split(reviews, labels,
                                                                        test_size=test_size,
                                                                        random_state=1)
    return x_train, x_test, y_train, y_test


def describe(x_train):
    print('training data description:\n'+str(pd.DataFrame.describe(x_train)) + '\n')
    vec = CountVectorizer(max_features=NUM_FEATURES)
    x_train_copy = x_train.copy()
    x_train_copy = vec.fit_transform(x_train_copy)
    print('Bag of words:')
    data = pd.DataFrame(x_train_copy.todense(), columns=vec.get_feature_names())
    print(data)


