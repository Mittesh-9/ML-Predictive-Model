# ML-Predictive-Model

This repository contains a Machine Learning project focused on **Sentiment Analysis** using restaurant reviews. The goal is to predict whether a review is **Positive**, **Negative**, or **Neutral** based on its content.

## Features

- **Sentiment Analysis** on restaurant reviews
- Flask web application for real-time sentiment prediction
- Uses **Natural Language Processing (NLP)** techniques for text vectorization
- Model trained using **Support Vector Machine (SVM)**

## Getting Started

### Prerequisites

Make sure you have Python 3.x installed on your system. Install the required Python libraries using the `requirements.txt` file:

pip install -r requirements.txt

### Required Libraries

- Flask
- pandas
- scikit-learn
- nltk

## Dataset

The project uses the `McDonald_s_Reviews.csv` dataset located in the `data/` folder. This dataset contains customer reviews from McDonald's, which are used to train the sentiment analysis model.

## How It Works

1. **Data Preparation:**
   - The reviews are loaded from `McDonald_s_Reviews.csv`.
   - Sentiment scores are calculated using NLTK's `SentimentIntensityAnalyzer`.
   - Reviews are classified as Positive, Negative, or Neutral based on the compound score.

2. **Model Training:**
   - The text data is transformed into numerical format using `TfidfVectorizer`.
   - The Support Vector Classifier (SVC) is trained on the vectorized reviews.

3. **Prediction:**
   - When a user submits a review via the web interface, the model predicts the sentiment based on the trained classifier.

## Flask Web App

The project includes a web application built using Flask. The app allows users to input a restaurant review and receive a sentiment prediction in real-time.

### Home Page:
- Enter your review in the text box and click "Submit".
- The predicted sentiment will be displayed on the same page.

![Screenshot](path/to/screenshot.png)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
