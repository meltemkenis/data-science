# Sentiment Analysis of Global Warming Using Twitter" Project

## Introduction

This project aims to perform sentiment analysis on tweets related to global warming using Twitter data. It retrieves tweets with the hashtag "#globalwarming" and analyzes their sentiment to determine whether they are positive, negative, or neutral.

## Requirements

To run the code in this project, you will need the following libraries installed:

- pandas
- tweepy
- nltk
- textblob
- numpy
- matplotlib
- wordcloud

## Data Retrieval from Twitter

The code connects to the Twitter API using the provided consumer key, consumer secret, access token, and access token secret. It searches for tweets with the hashtag "#globalwarming" in English language and retrieves up to 1000 results.

## Text Mining

After retrieving the tweets, the code performs text preprocessing to prepare the text for sentiment analysis. The text is converted to lowercase, numbers, punctuations, and 'rt' (re-tweet) expressions are removed, and stopwords (common words like "I," "me," "myself," "he," "she," etc.) are removed using NLTK's stopwords.

## Creating Word Cloud

The code generates a word cloud visualization based on the processed text. The word cloud displays the most frequent words in the tweets related to global warming.

## Sentiment Analysis

Sentiment analysis is performed on each tweet using TextBlob. The sentiment score of each tweet is calculated as polarity, indicating whether it is positive, negative, or neutral.

## Files

After running this code with the correct credentials, please expect to see following files: 

- data_twitter.csv: CSV file containing the extracted Twitter data (tweet ID, text, created_at, retweeted, retweet_count, user_screen_name, user_followers_count, user_location, hashtags).

- sentiment_global_warming.csv: CSV file containing the sentiment analysis results (tweet text, sentiment score, sentiment class - positive, negative, or neutral).

## Running the Code

Before running the code, make sure to replace the 'XXX' placeholders in the code with your actual Twitter API key, consumer secret, access token, and access token secret.

You can run the code in your Python environment to retrieve the data, perform text mining, create the word cloud, and conduct sentiment analysis on the tweets.

Please note that you may need to install the required libraries using pip install before running the code.
