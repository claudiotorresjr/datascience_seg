import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd

from model import utils

class Predict(object):
    """
        Base class to handle prediction
    """

    def __init__(self, data_path, classifiers, classifier_name):
        """
            Start prediciton process

            :param data_path: prediction file path
            :param classifiers: list of classifiers
            :param classifier_name: classifier name
        """
    
        print("\nStarting prediction process")
        type_dict = {
            'Tweet': str,
            'following':int,
            'followers':int,
            'actions': int,
            'is_retweet' : bool,
            'tweet_url' : str
        }
        predict_df = pd.read_csv(
            data_path,
            quotechar="'", dtype = type_dict, delimiter = ',', warn_bad_lines=False, error_bad_lines=False)

        predict_df = utils.create_features_columns(predict_df)

        predict_df_copy = predict_df.copy()
        predict_df_2 = predict_df_copy.iloc[:, 1:]

        predict_df_2 = utils.normalize(predict_df_2)

        result = classifiers[classifier_name].predict(predict_df_2)

        predict_df_copy["Type"] = pd.Series(result)
        predict_df_copy["Type"].replace({0:"Quality", 1:"Spam"}, inplace=True)

        chart_dir = data_path.split("/")[-1].split(".")[0]
        utils.generate_charts_and_infos(predict_df_copy, data_path, f"model/charts/{classifier_name}_{chart_dir}")