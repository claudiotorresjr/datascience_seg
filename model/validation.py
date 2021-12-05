from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

class Validation(object):
    """
        Base class to handle validation
    """

    def __init__(self, classifiers, y_test, y_pred):
        """
            Init validation

            :param classifiers: list of classifiers
            :param y_test: test y data
            :param y_pred: pred y data
        """

        print("\nStarting validation process")
        for name, _clf in classifiers.items():
            print(name)
            print(classification_report(y_test, y_pred[name]))
            print(confusion_matrix(y_test, y_pred[name]))   