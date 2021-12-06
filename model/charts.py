import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from model import utils
from model import charts

class Charts(object):
    def pie_chart_tweets(self, df1, df2, column, comp1, comp2, labels, description, output):
        """
            Create a pie chart

            :param df1: first dataframe
            :param df2: second dataframe to compare with the first
            :param column: column to be compared
            :param comp1: element value to compare (ex. Spam)
            :param comp2: element value to compare (ex. Quality)
            :param labels: chart label
            :param description: chart description
            :param output: chart output
        """

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
        plt.savefig(f"{output}.png")

    def count_col(self, df, col, limit):
        """
            count number of occurences of col using limit as trashold

            :param df: dataframe
            :param col: column name
            :param limit: treshold list
            :param description: chart description
            :param output: chart output
        """

        tweets = {
            "Spam": [],
            "Quality": []
        }
        for tp, l in tweets.items():
            all_tweets = df[df['Type'] == tp]
            for i in limit:
                count_values = len(all_tweets[all_tweets[col] == i])
                l.append(count_values)
        
        return tweets

    def get_col_counts_plot(self, df, limit, col, description, output):
        """
            create a bar plot

            :param df: dataframe
            :param limit: treshold list
            :param col: column name
        """
        list_test = []
        list2_test = []

        all_tweets = self.count_col(df, col, limit)
        unique_tweets = self.count_col(df.drop_duplicates(subset="Tweet"), col, limit)
        
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
        plt.savefig(f"{output}.png")

    @staticmethod
    def plot_roc_curve_folds(roc_dict, n_folds, sup_title, clf_name):
        """
            Plot the roc curves for a given classifier

            :param roc_dict: dict with roc curves of classifiers
            :param n_folds: total of folds
            :param sup_title: chart title
            :param clf_name: classsifier name
        """

        fig, axs = plt.subplots(nrows=1, ncols=n_folds, sharey=True, sharex=True, figsize=(15,5))

        for fold in range(5):
            axs[fold].plot(roc_dict[fold][clf_name]['FP'], roc_dict[fold][clf_name]['TP'])
            axs[fold].set_title('Fold ' + str(fold))
            axs[fold].set_xlabel('FP', fontsize=12)
            axs[fold].set_ylabel('VP', fontsize=12)

            axs[fold].grid(linewidth=0.25)

            #remove bounding box around the graphs 
            axs[fold].spines['left'].set_visible(False)
            axs[fold].spines['top'].set_visible(False)
            axs[fold].spines['right'].set_visible(False)  

        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        fig.suptitle(sup_title)
    
        fig.savefig(f"model/charts/ROC/{clf_name}_curves.png")