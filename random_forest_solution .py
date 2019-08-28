import numpy as np
import pandas as pd
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import pickle


NUM_FEATURES = 17500
TRAIN_PERCENTAGE = 0.8
LAMBDA_STEP_SIZE = 40
LAMBDA_STEP_PATH = 10000
LASSO_STEP_PATH = 0.005
OPTIMAL_LEAFS_MUM = 1024
OPTIMAL_DEPTH = 170
OPTIMAL_NUMBER_OF_TREES = 120
OPTIMAL_NUM_OF_FEATURES = 1600


def get_data(path):
    """
    get data, training data is in the file 'IMDB_Dataset.csv'
    testing data is in the file 'DataVault.csv'
    :return:  x_train, x_test, y_train, y_test
    """
    train_data = pd.read_csv("IMDB_Dataset.csv")
    test_data = pd.read_csv(path)
    x_train = train_data["review"]
    y_train = train_data["sentiment"]
    x_test = test_data["review"]
    y_test = test_data["sentiment"]
    y_train = np.unique(y_train, return_inverse=True)[1]
    y_test = np.unique(y_test, return_inverse=True)[1]

    return x_train, x_test, y_train, y_test


def boosting_optimal_tree():
    """
    train optimal classifier.
    the trained classifier will be saved in 'adabost_50_final.pickle' file.
    It's a sklearn.ensemble.AdaBoostClassifier object.
    :param x_train: train data
    :param y_train: train labels
    """
    x_train, x_test, y_train, y_test = get_data('DataVault.csv')
    vec = CountVectorizer(max_features=OPTIMAL_NUM_OF_FEATURES)
    x_train_copy = x_train.copy()
    x_train_copy = vec.fit_transform(x_train_copy)
    adb = AdaBoostClassifier(base_estimator=RandomForestClassifier(bootstrap=True, n_estimators=OPTIMAL_NUMBER_OF_TREES,
                                            max_leaf_nodes=OPTIMAL_LEAFS_MUM, max_depth=OPTIMAL_DEPTH,
                                            random_state=1, max_features='log2', n_jobs=4),
                             n_estimators=50, learning_rate=1)

    adb = adb.fit(x_train_copy, y_train)
    pickle_out = open('adabost_50_final.pickle', 'wb')
    pickle.dump(adb, pickle_out)
    pickle_out.close()
    print('Training is done.')


def print_score(x_test, y_test):
    """
    :param x_test: test data
    :param y_test: test labels
    prints test score of x_test and y_test. make sure
    'bag_of_words_opt.pickle' and 'adabost_50_final.pickle' is in the same
    file as 'optimal_classifier'
    """
    pickle_in = open('bag_of_words_opt.pickle', 'rb')
    vec = pickle.load(pickle_in)
    pickle_in.close()
    pickle_in = open('adabost_50_final.pickle', 'rb')
    adb = pickle.load(pickle_in)
    pickle_in.close()
    x_test_copy = vec.transform(x_test)
    print('test score: ' + str(adb.score(x_test_copy, y_test) * 100) + '% success rate')


def vault_score():
    """
    print score on vault
    """
    x_train, x_test, y_train, y_test = get_data('DataVault.csv')
    print_score(x_test, y_test)


def user_score():
    """
    print score on user's review
    :return:
    """
    x_train, x_test, y_train, y_test = get_data(sys.argv[1])
    print_score(x_test, y_test)


def menu():
    """
    opens main menu
    """
    while True:
        user_input = input('1.Get score on vault\n'
                           '2.Get score on your reviews\n')
        try:
            val = int(user_input)
            if not (1 <= val <= 3):
                raise ValueError

            if val == 1:
                vault_score()
                break

            if val == 2:
                user_score()
                break


        except ValueError:
            print('Insert a number between 1 and 2\n')


def main():
    menu()


if __name__ == '__main__':
    main()