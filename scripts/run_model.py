#!/usr/bin/python3
import argparse
import os
import pandas as pd

from model import preprocessing
from model import training
from model import predict
from model import utils

def get_arguments():
    """
        Get command line arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p',
        help='If present, the model will make the preprocessing.',
        action='store_true'
    )
    parser.add_argument(
        "-f",
        action='store',
        dest="predict_file",
        help="Tweets dataset to get prediction",
        required="True"
    )
    parser.add_argument(
        "-c",
        action='store',
        dest="classifier_name",
        help="Classifier to use in prediciton (RandomForest, KNN, MLP, SVC)",
        required="True"
    )


    options = parser.parse_args()
    return options

def training_and_validation_process(data_path):
    """
        Start the training and validation process

        :param data_path: train file path

        :return: classifiers list
    """

    print("\nStarting training process")
    train = training.Training(data_path)

    train.start_process()

    return train.classifiers

def do_preprocessing():
    """
        Start the preprocessing
    """

    print("Starting preprocessing")
    pre = preprocessing.Preprocessing()
    
    pre.start_process()


def main():
    """
        Main method
    """

    arg = get_arguments()

    classifiers = ["RandomForest", "KNN", "MLP", "SVC"]
    if arg.classifier_name not in classifiers:
        print("Classificador nao encontrado")
        exit()

    if not os.path.isfile(arg.predict_file):
        print("Arquivo para predicao nao encontrado")
        exit()

    if arg.p:
        do_preprocessing()

    classifiers = training_and_validation_process(preprocessing.Preprocessing.CLEAN_TRAIN_PATH)

    df = pd.read_csv(preprocessing.Preprocessing.CLEAN_TRAIN_PATH, delimiter = ',')
    utils.generate_charts_and_infos(df, preprocessing.Preprocessing.CLEAN_TRAIN_PATH, "model/charts/train")

    predict.Predict(arg.predict_file, classifiers, arg.classifier_name)

if __name__ == "__main__":
    main()