#!/usr/bin/python3
import argparse
import pandas as pd
import numpy as np

from model import preprocessing
from model import training
from model import charts
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

    options = parser.parse_args()
    return options

def generate_charts_and_infos(data_path):
    ch = charts.Charts(data_path)
    
    print("Generating charts...")
    ####### pie chart of all and unique tweets (spam and non-spam)
    output = "train/all_and_unique_spam"
    labels = 'Spam', 'Non-spam'
    description = ["Relação entre todos os tweets", "Relação entre tweets únicos (sem duplicados)"]
    df_unique = ch.df.drop_duplicates(subset="Tweet")
    ch.pie_chart_tweets(ch.df, df_unique, "Type", "Spam", "Quality", labels, description, output)

    ####### pie chart of spam and non-spam (retweeted and not)
    output = "train/retweet_and_not_spam"
    labels = 'retweet', 'No-retweet'
    description = ["Relação entre spam", "Relação entre não spam"]
    ch.pie_chart_tweets(ch.df[ch.df["Type"] == "Spam"], ch.df[ch.df["Type"] == "Quality"], "is_retweet", 1, 0, labels, description, output)

    ####### bar chart for relations between features (ex. hashtags) between spam and non-spam
    limit = np.arange(1, 6, 1)
    output = "train/hashtags_relation"
    ch.get_col_counts_plot(ch.df, limit, 'hashtags', "Número de hashtags", output)
    output = "train/mentions_relation"
    ch.get_col_counts_plot(ch.df, limit, 'mentions', "Número de menções", output)
    output = "train/urls_relation"
    ch.get_col_counts_plot(ch.df, limit, 'URLs', "Número de URLs", output)
    print("Charts generated")

    print(f"\n{data_path.split('/')[-1]} summary information")
    print(f"{utils.create_info_table(ch.df)}\n")

    print("\nGenerating top ranking...")
    for element in [["Hashtags", utils.find_hashtag], ["URLs", utils.find_url]]:
        print(f"\n{data_path.split('/')[-1]} {element[0]} rank for Spam and Non-spam")
        top_elements_df = []
        for tp in ["Spam", "Quality"]:
            ####### get all 'Spam' tweets and 'find_*' in it
            all_elements = utils.get_all_infos(ch.df, tp, element[1], element[0])
            ####### print a top 5 hashtags from Spam
            top_elements_df.append(utils.get_ranking(all_elements, 5, [element[0], "Quantidade"], tp))
        
        top = pd.concat([d.reset_index(drop=True) for d in [top_elements_df[0], top_elements_df[1]]], axis=1)
        print(top)

    print("-"*20)

def start_training_process(data_path):
   train = training.Training(data_path)

   train.start_training()

def do_preprocessing():
    print("Starting preprocessing")
    pre = preprocessing.Preprocessing()
    
    pre.start_process()


def main():
    arg = get_arguments()

    #all related with train
    if arg.p:
        do_preprocessing()

    generate_charts_and_infos(preprocessing.Preprocessing.CLEAN_TRAIN_PATH)

    start_training_process(preprocessing.Preprocessing.CLEAN_TRAIN_PATH)

if __name__ == "__main__":
    main()