from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm

import pandas as pd
import numpy as np

from model import validation
from model import utils


class Training(object):
    """
        Base class to handle training
    """

    def __init__(self, data_path):
        """
            init method

            :param data_path: training file path
        """
        
        self.data_path = data_path

        self.df = pd.read_csv(data_path, delimiter = ',')
        self.classifiers = []

    def start_process(self):
        """
            start pretraining, with normalization 
        """

        n_columns = len(self.df.columns)

        #filter data to that will be used by the model
        X = self.df.iloc[:, 1: n_columns - 1] 
        y = self.df['Type']

        #binarize the labels 
        #Quality -> 0 
        #Spam -> 1
        lb = LabelBinarizer()
        y = lb.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 13)

        print('Len train: {} - {:.2f}%'.format(len(y_train), len(y_train)/ len(y)*100))
        print('Len test:  {} - {:.2f}%'.format(len(y_test), len(y_test)/ len(y)*100))

        unique_train, counts_train = np.unique(y_train, return_counts=True)
        unique_test, counts_test  = np.unique(y_test, return_counts=True)

        train_count = dict(zip(unique_train, counts_train))
        test_count = dict(zip(unique_test, counts_test))

        print('Train count: ', train_count)
        print('Test count: ', test_count)

        X_train = utils.normalize(X_train)
        X_test = utils.normalize(X_test)

        self._start_training(X_train, y_train, X_test, y_test)

    def _start_training(self, X_train, y_train, X_test, y_test):
        """
            Start training and validation

            :param X_train: train X values
            :param y_train: train y values
            :param X_test: test X values
            :param y_test: test y values
        """
    
        np.random.seed(10)

        self.classifiers = {'RandomForest': RandomForestClassifier(), 
                    'KNN': KNeighborsClassifier(),
                    'MLP' : MLPClassifier(), 
                    'SVC' : svm.SVC()}

        y_pred = {'RandomForest': [],
                'KNN': [],
                'MLP' : [],
                'SVC' : []}

        for name, clf in self.classifiers.items():
            clf.fit(X_train, y_train)
            y_pred[name] = clf.predict(X_test)

        validation.Validation(self.classifiers, y_test, y_pred)
