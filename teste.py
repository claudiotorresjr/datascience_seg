import tweepy

API_KEY = "zlBrn2JIxuL5vp8KfKquoHovZ"
API_SECRET = "U8kJRmA80zDUUUCrrpMcCpUVmxnL6Vr88kKaiwIUVKuBV8h0jI"
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAISpUgEAAAAANxIPMALzrWI351MQA1aXDuZLxR8%3DmVGoZekK4OUGFuOorQKRGmATUTg2tI1ow2QR9s04kKsAXxdVSB"

BRAZIL_WOE_ID = 23424768

auth = tweepy.OAuthHandler(API_KEY, API_SECRET)
api = tweepy.API(auth)

#fetching the trends
trends = api.get_place_trends(id=BRAZIL_WOE_ID)

for value in trends:
    for trend in value['trends']:
        #pega só os trends que são #
        #if trend['name'].startswith('#'):
            print(trend)

tweets = api.search_tweets( q='#AlcolumbreDesempregadoEm2023', count=1)

list_tweets = [tweet for tweet in tweets]
for tweet in list_tweets:
    print(f"user.screen_name: {tweet.user.screen_name}")
    print(f"user.created_at: {tweet.user.created_at}")
    print(f"user.description: {tweet.user.description}")
    print(f"user.location: {tweet.user.location}")
    print(f"user.friends_count: {tweet.user.friends_count}")
    print(f"user.followers_count: {tweet.user.followers_count}")
    print(f"user.statuses_count (numer of tweets): {tweet.user.statuses_count}")
    print(f"tweet.retweet_count: {tweet.retweet_count}")
    print("_______________________"*3)

#todos os metodos do user
for a in dir(tweet.user):
    print(a)

print("_______________________"*3)

#todos os metodos do tweet
for a in dir(tweet):
    print(a)