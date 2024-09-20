import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

mcd_data = pd.read_csv("data/McDonald_s_Reviews.csv", encoding="latin-1")

# Data exploration
# print(mcd_data.columns)
# print(mcd_data.head(10))
# print(mcd_data.sample)
# print(mcd_data.info)

# Sentiment score calculation
sia = SentimentIntensityAnalyzer()

# Performing sentiment analysis on each review
sentiments = []
for review in mcd_data['review']:
    sentiment = sia.polarity_scores(review)
    sentiments.append(sentiment)

# print(sentiments)

# Sentiment classification
sentiment_labels = []
for sentiment in sentiments:
    compound_score = sentiment['compound']
    if compound_score >= 0.05:
        sentiment_labels.append("Positive")
    elif compound_score <= 0.05:
        sentiment_labels.append("Negative")
    else:
        sentiment_labels.append("Neutral")

# print(sentiment_labels)

# adding the sentiment labels to the dataframe
mcd_data['sentiment'] = sentiment_labels

# print(mcd_data[['review', 'sentiment']])
