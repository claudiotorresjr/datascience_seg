from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from model import utils


class Training(object):
    """
        Base class to handle training
    """
    def __init__(self, data_path):
        self.data_path = data_path

        self.df = pd.read_csv(data_path, delimiter = ',')

    def start_training(self):
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