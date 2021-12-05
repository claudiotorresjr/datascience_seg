import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from model import utils
from model import charts

class Charts(object):
    def __init__(self, data_path):
        self.data_path = data_path

        self.df = pd.read_csv(data_path, delimiter = ',')

    def pie_chart_tweets(self, df1, df2, column, comp1, comp2, labels, description, output):
        spam = len(df1[df1[column] == comp1])
        non_spam = len(df1[df1[column] == comp2])
        all_sizes = [spam/len(df1), non_spam/len(df1)]
        
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
        axes[0].set_title(description[0])
        axes[0].pie(all_sizes, labels=labels, autopct='%1.2f%%', startangle=90)
        axes[0].axis('equal')
        
        spam = len(df2[df2[column] == comp1])
        non_spam = len(df2[df2[column] == comp2])
        unique_sizes = [spam/len(df2), non_spam/len(df2)]

        axes[1].set_title(description[1])
        axes[1].pie(unique_sizes, labels=labels, autopct='%1.2f%%', startangle=90)
        axes[1].axis('equal')
        
        #plt.show()
        plt.savefig(f"model/charts/{output}.png")

    def count_col(self, df, num, limit):
        tweets = {
            "Spam": [],
            "Quality": []
        }
        for tp, l in tweets.items():
            all_tweets = df[df['Type'] == tp]
            for i in limit:
                count_values = len(all_tweets[all_tweets[num] == i])
                l.append(count_values)
        
        return tweets

    def get_col_counts_plot(self, df, limit, num, description, output):
        list_test = []
        list2_test = []

        all_tweets = self.count_col(df, num, limit)
        unique_tweets = self.count_col(df.drop_duplicates(subset="Tweet"), num, limit)
        
        width = 0.3
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
        
        #For all tweets
        #print(f"ALL spam tweets: {all_tweets['Spam']}")
        #print(f"ALL Quality tweets: {all_tweets['Quality']}")
        axes[0].bar(limit, all_tweets["Spam"], width, label = "spam")
        axes[0].bar(limit + width, all_tweets["Quality"], width, label = "non-spam")

        axes[0].set_title("Relação entre todos os tweets")
        axes[0].set_yscale('log')
        axes[0].legend()
        axes[0].set_xlabel(description)
        axes[0].set_ylabel('Quantidade de tweets')
        axes[0].set_xticks(limit)
        
        #For unique tweets
        #print(f"UNIQUE spam tweets: {unique_tweets['Spam']}")
        #print(f"UNIQUE Quality tweets: {unique_tweets['Quality']}")
        axes[1].bar(limit, unique_tweets["Spam"], width, label = "spam")
        axes[1].bar(limit + width, unique_tweets["Quality"], width, label = "non-spam")
        
        axes[1].set_title("Relação entre tweets únicos (sem duplicados)")
        axes[1].set_yscale('log')
        axes[1].legend()
        axes[1].set_xlabel(description)
        axes[1].set_ylabel('Quantidade de tweets')
        axes[1].set_xticks(limit)

        #plt.show()
        plt.savefig(f"model/charts/{output}.png")