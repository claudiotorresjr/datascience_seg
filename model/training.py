from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, classification_report,  recall_score, accuracy_score, precision_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import copy

import pandas as pd
import numpy as np

from model import charts
from model import utils


class Training(object):
    """
        Base class to handle training
    """

    def __init__(self, data_path, avaliation):
        """
            init method

            :param data_path: training file path
            :param avaliation: flag to show or not the kfold avaliation
        """
        
        self.data_path = data_path
        self.avaliation = avaliation

        self.df = pd.read_csv(data_path, delimiter = ',')
        self.classifiers = []

        np.random.seed(10)

    def start_process(self):
        """
            start pretraining, with normalization 
        """

        n_columns = len(self.df.columns)

        #filter data to that will be used by the model
        X = self.df.iloc[:, 1: n_columns - 1] 
        y = self.df['Type']

        #convert to numpy array 
        X = X.to_numpy()
        y = y.to_numpy()

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

        #### O grid search já foi executado e os parâmetros salvos. Não há a necessidade de rodar novamente pois é demorado.
        # classifiers = {
        #     RandomForestClassifier(n_jobs=-1): {
        #         'min_samples_split': [3, 5, 10], 
        #         'n_estimators' : [100, 300],
        #         'max_depth': [3, 5, 15, 25],
        #         'max_features': [3, 6, 11]
        #     },
        #     KNeighborsClassifier(): {
        #         'n_neighbors': [3, 5, 7, 10], 
        #         'algorithm' : ['auto', 'ball_tree', 'kd_tree'],
        #         'metric': ['euclidean', 'manhattan', 'minkowski']
        #     },
        #     KNeighborsClassifier(): {
        #         'hidden_layer_sizes' : [80, 100, 130],
        #         'learning_rate_init' : [0.001, 0.0001],
        #         'max_iter' : [100, 150, 200]
        #     }
        # }

        #for clf, param_grid in classifiers.items():
        #    _best_clf, classifiers[clf] = self._apply_grid_search(X_train, y_train, X_test, y_test, clf, param_grid)
        classifiers = {
            "RandomForest": "Curva Roc - Random Forest",
            "MLP": "Curva Roc - MLP",
            "KNN": "Curva Roc - KNN"
        }
    
        print("Starting K-fold training...")
        roc_curves = self._start_kfold(classifiers, X, y)

        for clf_name, sup_title in classifiers.items():
            charts.Charts.plot_roc_curve_folds(roc_curves, 5, sup_title, clf_name)

    def _percent(self, a, b):
        """
            Calculate the relation of two values
            :param a: Numerator
            :param b: Denominator
        """

        return (a / b) * 100

    def _calculate_metrics(self, y_true, y_pred):
        """
            Calculate the classifier metrics

            :param y_true: the real label
            :param y_pred: the predict label
        """
    
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)

        metrics = {'Acuracia': acc, 'Precisao': precision, 'Recall': recall}
    
        return metrics

    def _start_kfold(self,classifiers_params, X, y):
        """
            Show the kfold avaliation

            :param classifiers_params: all classifiers bes params
            :param X: X data
            :param y: y data (label)

            :return: roc curves
        """

        K_FOLDS = 5
 
        #prepare classifiers
        self.classifiers = {
            'RandomForest': RandomForestClassifier(n_estimators=100, max_depth = 15, max_features = 3 , min_samples_split = 3),
            'KNN': KNeighborsClassifier(n_neighbors = 10, algorithm='auto', metric = 'manhattan'),
            'MLP' : MLPClassifier(hidden_layer_sizes = 100, learning_rate_init = 0.001, max_iter= 200, early_stopping=True)
        }

        metrics = {
            'RandomForest': {'train': {}, 'test': {}},
            'KNN' : {'train': {}, 'test': {}},
            'MLP' : {'train': {}, 'test': {}}
        }

        roc_curve_metrics = {
            'RandomForest' : {'FP': 0.0, 'TP': 0.0, 'Threshold': 0.0},
            'KNN' : {'FP': 0.0, 'TP': 0.0, 'Threshold': 0.0},
            'MLP' : {'FP': 0.0, 'TP': 0.0, 'Threshold': 0.0}
        }

        #store the classifiers dict for every fold
        fold_classifiers = []
        fold_metrics = []
        roc_curves = [] 

        skfold = StratifiedKFold(n_splits = K_FOLDS)
        skfold.get_n_splits(X, y)
        fold_count = 0

        for train_index, test_index in skfold.split(X, y):
            print('------------------------------------------')
            print('Fold {} out of 5'.format(fold_count + 1))
            len_train, len_test = len(train_index), len(test_index)
            total = len_train + len_test

            print('Train: {} {:.2f}% Test: {} {:.2f}%'.format(len_train, self._percent(len_train, total), len_test, self._percent(len_test, total)))
        
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]


            for clf_name, classifier in self.classifiers.items():
                print('Training {}...'.format(clf_name))
                classifier.fit(X_train, y_train.ravel())

                y_pred_train = classifier.predict(X_train)
                y_pred_test = classifier.predict(X_test)

                metrics[clf_name]['train'] = self._calculate_metrics(y_train, y_pred_train)
                metrics[clf_name]['test'] = self._calculate_metrics(y_test, y_pred_test)

                #calculate roc_curves
                y_pred_test_prob = classifier.predict_proba(X_test)[:, 1]

                fpr, tpr, thr = roc_curve(y_test.ravel(), y_pred_test_prob)

                roc_curve_metrics[clf_name]['FP'] = copy.copy(fpr)
                roc_curve_metrics[clf_name]['TP'] = copy.copy(tpr)
                roc_curve_metrics[clf_name]['Threshold'] = copy.copy(thr)

        
            fold_classifiers.append(self.classifiers)
            fold_metrics.append(metrics)
            roc_curves.append(roc_curve_metrics)

            fold_count += 1

        if self.avaliation:
            self._kfold_avaliation(K_FOLDS, fold_metrics, fold_classifiers)

        return roc_curves

    def _apply_grid_search(X_train, y_train, X_test, y_test, clf, param_grid, name):
  
        scorers = {
        'precision_score': make_scorer(precision_score),
        'recall_score': make_scorer(recall_score),
        'accuracy_score': make_scorer(accuracy_score)
        }

        grid_search = GridSearchCV(clf, param_grid, scoring=scorers, refit='precision_score', return_train_score= True, n_jobs=-1)
        grid_search.fit(X_train, y_train.ravel())

        y_pred = grid_search.predict(X_test)

        print(f"Best params for {name}:")
        print(grid_search.best_params_)
        print("Confusion matrix:")
        print(confusion_matrix(y_pred, y_test))

        return grid_search, grid_search.best_params_

    def _kfold_avaliation(self, K_FOLDS, fold_metrics, fold_classifiers):
        """
            Show the kfold avaliation

            :param K_FOLDS: total of kfold
            :param fold_metrics: kfold metrics calculated
            :param fold_classifiers: list of all classifiers
        """
    
        for fold in range(K_FOLDS):
            print('\n=========== Fold {} ===============\n'.format(fold))

            for clf_name in fold_classifiers[fold].keys():

                print('\n-- {} --'.format(clf_name))

                acc_train  = fold_metrics[fold][clf_name]['train']['Acuracia']
                prec_train = fold_metrics[fold][clf_name]['train']['Precisao']
                rec_train  = fold_metrics[fold][clf_name]['train']['Recall']

                acc_test  = fold_metrics[fold][clf_name]['test']['Acuracia'] 
                prec_test = fold_metrics[fold][clf_name]['test']['Precisao'] 
                rec_test  = fold_metrics[fold][clf_name]['test']['Recall'] 
            
                print('          Treino - Teste')
                print('Acuracia:  {:.2f}  -  {:.2f}'.format(acc_train, acc_test))
                print('Precisao:  {:.2f}  -  {:.2f}'.format(prec_train, prec_test))
                print('Recall:    {:.2f}  -  {:.2f}'.format(rec_train, rec_test))