"""
author: Idan Yamin
"""
import graphs_and_results
import sklearn
import sklearn.linear_model


def ridge_menu(x_train, x_test, y_train, y_test):
    """
    Opens the ridge menu
    :param x_train: train data
    :param x_test: test data
    :param y_train: train labels
    :param y_test: test labels
    """
    print('You choose Ridge Regression')
    while True:
        user_input = input('1.plot graph of 0-1 mean loss as a function of lambda\n'
                           '2.plot regularization path\n')
        try:
            val = int(user_input)
            if not (1 <= val <= 2):
                raise ValueError
            print('It may take a while')
            if val == 1:
                graphs_and_results.plot_ridge_error_as_function_of_lambda(x_train, x_test, y_train, y_test)
                break

            if val == 2:
                graphs_and_results.plot_ridge_regularization_path(x_train, y_train)
                break

        except ValueError:
            print('Enter a number between 1 and 3')


def menu():
    """
    This functions manges the main menu.
    """
    x_train, x_test, y_train, y_test = graphs_and_results.get_data()
    while True:
        user_input = input('1.Base line classifier (Logistic Regression)'
                           '\n2.Linear classifier mean 0-1 loss (an example of bad classifier)\n'
                           '3.Ridge regression\n'
                           '4.Lasso regression, plot regularization path\n'
                           '5.Adaboost\n'
                           '6.Random Forest\n'
                           '7.Describe the data\n')
        try:
            val = int(user_input)
            if not (1 <= val <= 7):
                raise ValueError
            if val == 1:
                classifier = sklearn.linear_model.LogisticRegression
                logistic_regression_menu('You choose logistic regression', x_train, x_test, y_train, y_test, classifier)
                break
            if val == 2:
                graphs_and_results.linear_classifier(x_train, x_test, y_train, y_test)
                break
            if val == 3:
                ridge_menu(x_train, x_test, y_train, y_test)
                break
            if val == 4:
                graphs_and_results.plot_lasso_regularization_path(x_train, y_train)
                break
            if val == 5:
                adaboost_menu(x_train, x_test, y_train, y_test)
                break
            if val == 6:
                random_forest_menu(x_train, x_test, y_train, y_test)
                break
            if val == 7:
                graphs_and_results.describe(x_train)
                break

        except ValueError:
            print('Enter a number between 1 and 7')
            continue


def random_forest_menu(x_train, x_test, y_train, y_test):
    """
    opens the random forest menu
    :param x_train: train data
    :param x_test: test data
    :param y_train: train labels
    :param y_test: test labels
    :return:
    """
    while True:
        user_input = input('1.Mean error as a function of leaves graph (max_features=sqrt)\n'
                           '2.Mean error as a function of leaves graph (max_features=log2)\n'
                           '3.Mean error as a function of tree\'s depth graph\n'
                           '4.Mean error as a function of the number of features\n'
                           '5.Mean error as a function of the number of tress\n'
                           '6.Mean error of boosted optimal tree\n')
        try:
            val = int(user_input)
            if not (1 <= val <= 6):
                raise ValueError
            if val == 1:
                graphs_and_results.random_forest_error_as_a_function_of_leafs(x_train, x_test, y_train, y_test, 45, 100, 'sqrt')
                break
            if val == 2:
                graphs_and_results.random_forest_error_as_a_function_of_leafs(x_train, x_test, y_train, y_test, 600, 25, 'log2')
                break
            if val == 3:
                graphs_and_results.random_forest_error_as_a_function_of_depth(x_train, x_test, y_train, y_test, 200, 2,
                                                           'log2')
                break
            if val == 4:
                graphs_and_results.random_forest_error_as_a_function_of_features_num(x_train, x_test, y_train, y_test, 25, 1000, 'log2')
                break

            if val == 5:
                graphs_and_results.random_forest_error_as_a_function_of_trees_num(x_train, x_test, y_train, y_test, 10, 40, 'log2')
                break

            if val == 6:
                graphs_and_results.boosting_optimal_tree(x_train, x_test, y_train, y_test)
                break

        except ValueError:
            print('Enter a number between 1 and 6')
            continue


def adaboost_menu(x_train, x_test, y_train, y_test):
    """
    Opens Adaboost menu
    :param x_train: train data
    :param x_test: test data
    :param y_train: train labels
    :param y_test: test labels
    """
    print('AdaBoost')
    while True:
        user_input = input('1.Decision stump classifier\n'
                           '2.boost decision stump\n')
        try:
            val = int(user_input)
            if not (1 <= val <= 2):
                raise ValueError
            if val == 1:
                print('mean 0-1 loss: ' +
                      str(graphs_and_results.decision_stump_loss(x_train, x_test, y_train, y_test)))
            if val == 2:
                graphs_and_results.adaboost(x_train, x_test, y_train, y_test)

        except ValueError:
            print('Enter a number between 1 and 10')


def logistic_regression_menu(text, x_train, x_test, y_train, y_test, classifier):
    """
    Opens logistic regression menu
    :param text: text to print
    :param x_train: train data
    :param x_test: test data
    :param y_train: train labels
    :param y_test: test labels
    :param classifier: classifier
    """
    print(text)
    while True:
        user_input = input('Choose one of the options below\n'
                           '1.plot 0-1 loss as a function of the number of features (a couple of minutes)\n'
                           '2.Get mean 0-1 loss on all the words in the bag of words\n')
        try:
            val = int(user_input)
            if not (1 <= val <= 2):
                raise ValueError
            if val == 1:
                user_input = input('Choose one of the options below\n'
                                   '1.use test set for validation\n'
                                   '2.use cross validation\n')
                val = int(user_input)
                if not (1 <= val <= 2):
                    raise ValueError

                if val == 1:
                    graphs_and_results.feature_selection_graph(classifier, x_train, x_test, y_train, y_test)
                    break
                else:
                    graphs_and_results.feature_selection_graph(classifier, x_train, x_test, y_train, y_test, True)
                    break
            if val == 2:
                print('mean 0-1 loss: ' + str(graphs_and_results.zero_one_loss(classifier, x_train, x_test, y_train, y_test)))

        except ValueError:
            print('Enter a number between 1 and 2')
            continue