# Meltem Keniş
# August 2019 - "Python and Machine Learning" project
# Sentiment Analysis of Global Warming Using Twitter

from warnings import filterwarnings
import pandas as pd
import tweepy, codecs
import nltk
from nltk.corpus import stopwords
from textblob import Word
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob

filterwarnings('ignore')

## Data retrieval from Twitter:

# Key and tokens which are necessary to enter Twitter API:

consumer_key = 'XXX'
consumer_secret = 'XXX'
access_token = 'XXX'
access_token_secret = 'XXX'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)


# We specify the features of the data that we will extract from Twitter:
tweets = api.search(
            q = "#globalwarming",
            lang = "en",
            result_type = "mix",
            count = 1000,
        )

#  We specify the features of the tweets with globalwarming hashtag:

def hashtag_df(tweets):
    id_list = [tweet.id for tweet in tweets]
    df = pd.DataFrame(id_list, columns = ["id"])
    
    df["text"] = [tweet.text for tweet in tweets]
    df["created_at"] = [tweet.created_at for tweet in tweets]
    df["retweeted"] = [tweet.retweeted for tweet in tweets]
    df["retweet_count"] = [tweet.retweet_count for tweet in tweets]
    df["user_screen_name"] = [tweet.author.screen_name for tweet in tweets]
    df["user_followers_count"] = [tweet.author.followers_count for tweet in tweets]
    df["user_location"] = [tweet.author.location for tweet in tweets]
    df["Hashtags"] = [tweet.entities.get('hashtags') for tweet in tweets]
    
    return df


# Converting tweets with globalwarming hashtag to a dataframe:
df = hashtag_df(tweets)

# Saving dataframe as csv file.
df.to_csv("data_twitter.csv")

## Text Mining:

# Case conversion:
df['text'] = df['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))

# Removing numbers, punctuations and 'rt' expressions:
df['text'] = df['text'].str.replace('[^\w\s]','')
df['text'] = df['text'].str.replace('rt','')
df['text'] = df['text'].str.replace('\d','')

# Removing stopwords (I, me, myself, he, she, they, our, mine, you, yours):
nltk.download('stopwords')
sw = stopwords.words('english')
df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in sw))

get_ipython().system('pip install textblob')

# Lemmi
nltk.download('wordnet')
df['text'] = df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()])) 


## Creating Word Cloud:

get_ipython().system('pip install WordCloud')

text = " ".join(i for i in df.text)

wordcloud = WordCloud(background_color = "white").generate(text)
plt.imshow(wordcloud, interpolation = "bilinear")
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()


## Sentiment Analysis:

def calculate_sentiment_score(df):

    text = df["text"]

    for i in range(0,len(text)):
        textB = TextBlob(text[i])
        sentiment_score = textB.sentiment.polarity
        df.set_value(i, 'sentiment_score', sentiment_score)
        
        if sentiment_score < 0.00:
            emotion_class = 'Negative'
            df.set_value(i, 'emotion_class', emotion_class)

        elif sentiment_score > 0.00:
            emotion_class = 'Pozitive'
            df.set_value(i, 'emotion_class', emotion_class)

        else:
            emotion_class = 'Neutral'
            df.set_value(i, 'emotion_class', emotion_class)
            
    return df

df = hashtag_df(tweets)
sgw = calculate_sentiment_score(df)

sgw.to_csv("sentiment_global_warming.csv")
df.groupby("emotion_class").count()["id"]
emotion_freq = df.groupby("emotion_class").count()["id"]
emotion_freq.plot.bar(x = "emotion_class",y = "id");
