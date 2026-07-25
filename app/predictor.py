import joblib

# Load the saved model
model = joblib.load("models/fake_news_model.pkl")

# Load the saved TF-IDF vectorizer
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


def predict_news(news_text):
    """
    Predict whether a news article is Fake or Real and return
    the prediction along with its confidence score.
    """

    # Convert the text into TF-IDF features
    text_vector = vectorizer.transform([news_text])

    # Predict the class
    prediction = model.predict(text_vector)[0]

    # Get prediction probabilities
    probabilities = model.predict_proba(text_vector)[0]

    # Confidence of the predicted class
    confidence = probabilities[prediction] * 100

    # Convert numeric prediction to readable text
    if prediction == 0:
        result = "Fake News"
    else:
        result = "Real News"

    return {
        "prediction": result,
        "confidence": round(confidence, 2)
    }