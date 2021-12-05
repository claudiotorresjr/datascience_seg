import tweepy
import argparse

API_KEY = ""
API_SECRET = ""
BEARER_TOKEN = ""

BRAZIL_WOE_ID = 23424768

auth = tweepy.OAuthHandler(API_KEY, API_SECRET)
api = tweepy.API(auth)

def get_arguments():
    """
        Get command line arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t',
        action='store',
        dest="tag",
        help='Tag to search in tweeter',
        required="True"
    )
    parser.add_argument(
        '-n',
        action='store',
        dest="num_tweets",
        help='Number of tweets',
        required="True"
    )
    parser.add_argument(
        "-o",
        action='store',
        dest="output_path",
        help="Output file path",
        required="True"
    )

    options = parser.parse_args()
    return options

def get_all_urls(tweet, full_text, urls_to_remove):
    """
        Get all urls from tweet

        :param full_text: tweet text
        :param urls_to_remove: urls to not get
    """

    for entitie in ["urls", "media"]:
        try:
            for ent in tweet.entities[entitie]:
                #se a url ou media esta no texto do tweet e a expanded_url é de um status, entao remove do tweet
                if ent['url'] in full_text and "https://twitter.com/" in ent['expanded_url']:
                    #print(f"({entitie}) - Tem link do retweet: {ent['url']}. Será removido.")
                    if ent['url'] not in urls_to_remove:
                        urls_to_remove.append(ent['url'])
        except Exception as ex:
            #nao existe key entities
            pass

def save_tweets_dataset(all_tweets_info, urls_to_remove, output_path, remove_urls=False):
    """
        Save all tweets info to a file

        :param all_tweets_info: all tweets info
        :param urls_to_remove: urls to not save
        :param output_path: tweets info file output
        :param remove_urls: if set, will not save urls in urls_to_remove
    """

    with open(output_path, "w") as save_file:
        save_file.write(f"id,Tweet,following,followers,actions,is_retweet,tweet_url\n")
        for i, tt in enumerate(all_tweets_info):
            if remove_urls:
                for url in urls_to_remove:
                    if url in tt['full_text']:
                        tt['full_text'] = tt['full_text'].replace(url, "")

            save_file.write(f"{i},'{tt['full_text']}',{tt['following']},{tt['followers']},{tt['action']},")
            save_file.write(f"{tt['isretweeted']},{tt['tweeter_url']}\n")
    
def search_for_tag(tag, num_tweets, output_path):
    """
        Search for a tag/text in twitter

        :param num_tweets: number of tweets to search
        :param output_path: tweets info file output
    """

    all_tweets_info = []
    urls_to_remove = []
    for tweet in tweepy.Cursor(api.search_tweets, tag, lang='pt', count=100, tweet_mode='extended').items(num_tweets):
        if 'retweeted_status' in tweet._json:
            full_text = tweet._json['retweeted_status']['full_text'].replace('\n', '')
            isretweeted = True
        else:
            full_text = tweet.full_text.replace('\n', '')
            isretweeted = False

        actions = tweet.retweet_count + tweet.favorite_count
        
        get_all_urls(tweet, full_text, urls_to_remove)

        all_tweets_info.append({
            "full_text": full_text,
            "following": tweet.user.friends_count,
            "followers": tweet.user.followers_count,
            "action": actions,
            "isretweeted": isretweeted,
            "tweeter_url": f"https://twitter.com/{tweet.user.screen_name}/status/{tweet.id}"
        })

    save_tweets_dataset(all_tweets_info, urls_to_remove, output_path)

def main():
    """
        Main method
    """

    arg = get_arguments()
    tag = arg.tag
    num_tweets = int(arg.num_tweets)
    output_path = arg.output_path
    
    search_for_tag(tag, num_tweets, output_path)

if __name__ == "__main__":
    main()

# #todos os metodos do user
# for a in dir(tweet.user):
#     print(a)

# print("_______________________"*3)

#todos os metodos do tweet
# for a in dir(tweet):
#     print(a)