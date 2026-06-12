# Email/SMS Spam Classifier

A machine learning web app that classifies SMS/Email messages as **Spam** or **Not Spam (Ham)** using Natural Language Processing (NLP) and a Multinomial Naive Bayes model.

## Features

- Real-time spam detection via a Streamlit web interface
- Text preprocessing (lowercasing, tokenization, stopword removal, stemming)
- TF-IDF vectorization with 3000 features
- Multinomial Naive Bayes classifier (97% accuracy)

## Project Structure

```
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
├── .gitignore
├── README.md
├── config/
│   ├── nltk.txt           # NLTK data requirements
│   ├── Procfile.txt       # Heroku deployment config
│   └── setup.sh           # Deployment setup script
├── data/
│   └── spam.csv           # Training dataset
├── models/
│   ├── model.pkl          # Trained Naive Bayes model
│   └── vectorizer.pkl     # TF-IDF vectorizer
├── notebooks/
│   └── sms-spam-detection.ipynb  # EDA & model training
└── scripts/
    └── train_model.py     # Script to retrain the model
```

## Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Email-Spam-Classifier.git
   cd Email-Spam-Classifier
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data**
   ```python
   python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt_tab')"
   ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

## Usage

1. Open the app in your browser (http://localhost:8501)
2. Type or paste a message in the text area
3. Click **Predict**
4. The app will classify it as **Spam** or **Not Spam**

## Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 97% |
| Precision | 100% |
| Best Model | Multinomial Naive Bayes |

## Deployment

The app is configured for deployment on **Heroku** (see `config/` files).

## Dataset

The model is trained on the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) — 5,574 SMS messages labeled as spam or ham.
