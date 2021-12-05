import re
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from model import charts

def find_url(string):
    """
        Find all URLs in string

        :param string: string to search for urls

        :return: all urls found
    """

    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.findall(regex, string)      

    return url 

def has_url(string):
    """
        Verify if string has url

        :param string: string to find urls

        :return: boolean if found or not
    """

    return len(find_url(string)) > 0
   
def count_url(string):
    """
        Count the number of urls in string

        :param string: string to count urls

        :return: number of urls
    """

    return len(find_url(string))

def count_words(string):
    """
        Count the number of words in string

        :param string: string to count words

        :return: number of words
    """

    words = string.split(' ')  

    return len(words)

def find_hashtag(string):
    """
        Find all hashtags in string

        :param string: string to search for hashtags

        :return: all hashtags
    """

    regex = r"\B(\#[a-zA-Z]+\b)"
    hashtags = re.findall(regex, string)
  
    return hashtags

def count_hashtags(string):
    """
        Count the number of hashtags in string

        :param string: string to count hashtags

        :return: number of hashtags
    """

    return len(find_hashtag(string))

def hashtag_per_word_ratio(string):
    """
        Calculate the ratio between hashtags and words 

        :param string: string to count hashtags and words

        :return: hashtag per word ratio
    """

    hashtags_num = count_hashtags(string)
    words_num = count_words(string)

    return hashtags_num / words_num

def URL_per_word_ratio(string):
    """
        Calculate the ratio between urls and words 

        :param string: string to count urls and words

        :return: url per word ratio
    """

    urls_num = count_url(string)
    word_num = count_words(string)

    return urls_num / word_num

def count_numbers(string):
    """
        Count the number of numbers in string

        :param string: string to count numbers

        :return: number of numbers
    """

    #can't find decimal numbers 
    regex = "[0-9]+"
    numbers = re.findall(regex, string)
  
    return len(numbers)
  
def count_mentions(string):
    """
        Count the number of mentions in string

        :param string: string to count mentions

        :return: number of mentions
    """

    regex = r"(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9_]+)"
    mentions = re.findall(regex, string)

    return len(mentions)

def generate_charts_and_infos(df, data_path, output_dir):
    """
        Generate all charts and calculate infos about the data

        :param df: dataframe
        :param data_path: data path to get tha file name
        :param output_dir: chart output
    """

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        print(f"created folder: {output_dir}")

    ch = charts.Charts()
    
    print("Generating charts...")
    ####### pie chart of all and unique tweets (spam and non-spam)
    output = f"{output_dir}/all_and_unique_spam"
    labels = 'Spam', 'Non-spam'
    description = ["Relação entre todos os tweets", "Relação entre tweets únicos (sem duplicados)"]
    df_unique = df.drop_duplicates(subset="Tweet")
    ch.pie_chart_tweets(df, df_unique, "Type", "Spam", "Quality", labels, description, output)

    ####### pie chart of spam and non-spam (retweeted and not)
    output = f"{output_dir}/retweet_and_not_spam"
    labels = 'retweet', 'No-retweet'
    description = ["Relação entre spam", "Relação entre não spam"]
    ch.pie_chart_tweets(df[df["Type"] == "Spam"], df[df["Type"] == "Quality"], "is_retweet", 1, 0, labels, description, output)

    ####### bar chart for relations between features (ex. hashtags) between spam and non-spam
    limit = np.arange(1, 6, 1)
    output = f"{output_dir}/hashtags_relation"
    ch.get_col_counts_plot(df, limit, 'hashtags', "Número de hashtags", output)
    output = f"{output_dir}/mentions_relation"
    ch.get_col_counts_plot(df, limit, 'mentions', "Número de menções", output)
    output = f"{output_dir}/urls_relation"
    ch.get_col_counts_plot(df, limit, 'URLs', "Número de URLs", output)
    print("Charts generated")

    print(f"\n{data_path.split('/')[-1]} summary information")
    print(f"{create_info_table(df)}\n")

    print("\nGenerating top ranking...")
    for element in [["Hashtags", find_hashtag], ["URLs", find_url]]:
        print(f"\n{data_path.split('/')[-1]} {element[0]} rank for Spam and Non-spam")
        top_elements_df = []
        for tp in ["Spam", "Quality"]:
            ####### get all 'Spam' tweets and 'find_*' in it
            all_elements = get_all_infos(df, tp, element[1], element[0])
            ####### print a top 5 hashtags from Spam
            top_elements_df.append(get_ranking(all_elements, 5, [element[0], "Quantidade"]))
        
        top = pd.concat([d.reset_index(drop=True) for d in [top_elements_df[0], top_elements_df[1]]], axis=1)
        print(top)

    print("-"*20)

def create_features_columns(df, train=False):
    """
        Calculate all df features

        :param df: dataframe
        :param train: if set, will append the Type column

        :return: dataframe
    """

    model_features = {
        "words": count_words,
        "hashtags": count_hashtags,
        "hashtag_ratio": hashtag_per_word_ratio,
        "URLs": count_url,
        "URL_ratio": URL_per_word_ratio,
        "numbers": count_numbers,
        "mentions": count_mentions
    }
    new_columns = ['Tweet', 'following', 'followers', 'actions', 'is_retweet']

    for feature, func in model_features.items():
        df[feature] = df['Tweet'].apply(func)
        new_columns.append(feature)
    
    if train:
        new_columns.append('Type')
    
    df = df[new_columns]

    return df

def normalize(to_normalize):
    """
        Normalize data

        :param to_normalize: data to be normalized

        :return: normalized data
    """

    min_max_scaler = MinMaxScaler()
    to_normalize = min_max_scaler.fit_transform(to_normalize)

    return to_normalize

def get_all_infos(df, tp, func, is_url):
    """
        Get all elements from df 'Type' field that match the func method
        (ex. Get all URLs from df 'Type' field that match the find_url method)

        :param df: dataframe to search
        :param tp: type of tweet (spam or non-spam)
        :param func: method to get the elements from df
        :param is_url: if is "URLs",search for the real url (some tt urls is like t.co/xxx)

        :return: all elements found
    """

    new_df = df[df["Type"] == tp]
    
    all_elements = {}
    for tweet in new_df["Tweet"]:
        elements = []
        try:
            elements = [''.join(i) for i in func(tweet)]
        except Exception as ex:
            pass
        else:
            for u in elements:
                #se for url, acha a url normal (a maioria do twitter vem na forma t.co)
                #if is_url == "URLs":
                #    u = os.popen(
                #        f"curl -s -o /dev/null --head -w '%{{url_effective}}\n' -L {u}").read()
                if u in all_elements:
                    all_elements[u] += 1
                else:
                    all_elements[u] = 1

    return all_elements

def get_ranking(rank_dict, rank, columns):
    """
        Return a dataframe with the top 'rank' values in rank_dict

        :param rank_dict: dict with all elements and it count
        :param rank: top 'rank' values
        :param columns: two columns list: element name and int count (URLs and count)
        :param tp: type of tweet (spam or non-spam)

        :return: dataframe with ranking
    """

    rank_dict = dict(sorted(rank_dict.items(), key=lambda x: x[1], reverse=True))
    
    keys = []
    values = []
    for i, v in enumerate(rank_dict):
        if i == 5:
            break

        value = rank_dict[v]
        if columns[0] == "URLs":
            #se for url, acha a url normal (a maioria do twitter vem na forma t.co)
            v = os.popen(
                f"curl -s -o /dev/null --head -w '%{{url_effective}}\n' -L {v}").read()
        keys.append(v.split('\n')[0])
        values.append(value)
    
    return pd.DataFrame(list(zip(keys, values)), index=np.arange(1, rank+1, 1), columns=columns)

def get_data_info(data_df, tweet_type):
    """
        Get dataframe data about spam and non-spam tweets

        :param data_df: dataframe
        :param tweet_type: Spam or Quality

        :return: list with all data calculated
    """

    #todos os tweets do tipo 'tweet_type'
    only_tweet_type = data_df[data_df['Type'] == tweet_type]
    
    #tweets que possuem urls
    tweet_url = only_tweet_type[only_tweet_type['URLs'] > 0]
    #tweets que possuem mais de 1 urls
    more_one_url = only_tweet_type[only_tweet_type['URLs'] > 1]
    
    #quantidade de tweets do tipo 'tweet_type'
    tweet_count = len(only_tweet_type)

    #quantidade de tweets com url
    tweet_url_count = len(tweet_url)
    #quantidade de tweets com mais de uma url
    tweet_more_one_url_count = len(more_one_url)
    
    #tweets unicos (sem repetição)
    unique_tweet = only_tweet_type['Tweet'].unique()
    #quantidade de tweets unicos (sem repetição)
    unique_tweet_count = len(unique_tweet)
    
    return [
        f"{tweet_count} ({(tweet_count/len(data_df))*100:.2f}%)",
        unique_tweet_count,
        f"{(unique_tweet_count/tweet_count)*100:.2f}%",
        tweet_url_count,
        f"{(tweet_url_count/tweet_count)*100:.2f}%",
        tweet_more_one_url_count,
        f"{(tweet_more_one_url_count/tweet_url_count)*100:.2f}%"
    ]

def create_info_table(df):
    """
        Create the table (dataframe) with the data info list

        :param df: dataframe

        :return: dataframe with all info
    """

    spam = get_data_info(df, "Spam")
    non_spam = get_data_info(df, "Quality")

    index = [
        "Tweets identificados",
        "Tweets únicos (sem repetição)",
        "Tweets únicos/identificados",
        "Total de urls",
        "urls/identificados",
        "mais de 1 url", 
        "mais de 1 url/total de urls"
    ]
    return pd.DataFrame(list(zip(spam, non_spam)), index=index, columns=["spam", "non-spam"])