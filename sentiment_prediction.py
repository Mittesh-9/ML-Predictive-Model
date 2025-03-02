import pandas as pd
# adding 'nltk.download('vader_lexicon'after import nltk to ensure VADER lexicon is available for SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
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

# Dataset splitting
X = mcd_data['review']
y = mcd_data['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Vectorization >> to convert the textual data into a numerical representation suitable for ML algorithms. This process involved transforming the reviews into a format that captures their features and patterns effectively.
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model training (Support Vector Classifier)
model = SVC()
model.fit(X_train_tfidf, y_train)

y_prediction = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_prediction)
# print("Accuracy:", accuracy)
# print("Classification report:")
# print(classification_report(y_test, y_prediction))

# sentiment prediction function >>
def predict_sentiment(review):
    review_tfidf = vectorizer.transform([review])
    sentiment = model.predict(review_tfidf)
    return sentiment[0]


# sample testing
#new_review = "This restaurant has excellent service and delicious food."
#predicted_sentiment = predict_sentiment(new_review)
#print("Predicted sentiment:", predicted_sentiment)

#new_review2 = "This restaurant sucks."
#predicted_sentiment = predict_sentiment(new_review2)
#print("Predicted sentiment:", predicted_sentiment)

#new_review3 = "This is dull"
#predicted_sentiment = predict_sentiment(new_review4)
#print("Predicted sentiment:", predicted_sentiment)

