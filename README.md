# 📰 Fake News Detection System

A machine learning-powered Fake News Detection System built with **Python**, **Scikit-learn**, and **FastAPI**. The project classifies news articles as **Fake** or **Real** using a **TF-IDF Vectorizer** and a **Logistic Regression** model, and exposes the trained model through a REST API.

---

## 🚀 Features

* Preprocesses and combines fake and real news datasets
* Converts text into numerical features using TF-IDF
* Trains a Logistic Regression classifier
* Achieves approximately **98.6% test accuracy**
* Saves the trained model and vectorizer using Joblib
* Provides predictions through a FastAPI REST API
* Returns prediction confidence scores
* Includes input validation and error handling
* Interactive API documentation with Swagger UI

---

## 🛠️ Tech Stack

* Python 3.13
* Pandas
* NumPy
* Scikit-learn
* Joblib
* FastAPI
* Uvicorn

---

## 📂 Project Structure

```text
FakeNewsDetector/
│
├── app/
│   ├── main.py
│   └── predictor.py
│
├── data/
│   ├── Fake.csv
│   ├── True.csv
│   └── archive.zip
│
├── models/
│   ├── fake_news_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── training/
│   └── train.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone <repository-url>
cd FakeNewsDetector
```

Create and activate a virtual environment:

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the API

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

Open the API documentation:

* Swagger UI: `http://127.0.0.1:8000/docs`
* ReDoc: `http://127.0.0.1:8000/redoc`

---

## 📡 API Endpoints

### GET /

Returns a welcome message.

### POST /predict

Predicts whether a news article is fake or real.

### Example Request

```json
{
  "text": "The White House announced new sanctions targeting several foreign companies accused of violating international trade regulations."
}
```

### Example Response

```json
{
  "prediction": "Real News",
  "confidence": "92.41%"
}
```

---

## 📊 Model Performance

* Algorithm: Logistic Regression
* Feature Extraction: TF-IDF Vectorizer
* Train-Test Split: 80/20
* Test Accuracy: **98.59%**

Classification metrics:

* Precision: ~0.99
* Recall: ~0.99
* F1-score: ~0.99

---

## ⚠️ Limitations

* The model was trained primarily on political and world news articles.
* Predictions on very short or out-of-domain text may be less reliable.
* The model recognises patterns in text rather than verifying factual correctness.
* Confidence scores indicate the model's certainty, not whether a claim is objectively true.

---

## 💡 Future Improvements

Possible future enhancements include:

* Docker containerisation
* React or Streamlit frontend
* Cloud deployment (Render or Hugging Face Spaces)
* Support for larger and more diverse datasets
* Deep learning models such as BERT or RoBERTa

---

## 📜 License

This project is intended for educational and learning purposes.


## 📸 Screenshots

### Home Endpoint

![Home Page](screenshots/home.png)

### Successful Prediction

![Prediction](screenshots/real_prediction.png)

### Empty Input Validation

![Empty Input](screenshots/empty_input.png)

### Minimum Length Validation

![Short Input](screenshots/short_input.png)
