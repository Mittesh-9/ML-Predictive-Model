import os
from flask import Flask, render_template, request
from sentiment_prediction import predict_sentiment

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        sentiment = predict_sentiment(review)
        return render_template('index.html', review=review, sentiment=sentiment)

if __name__ == '__main__':
    # Get the port from environment variable and default to 5000 for local development
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
    print(f"Running on port: {port}")

