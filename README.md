# TweetSentimentClassifier

# TweetSentimentClassifier

A Natural Language Processing (NLP) project that performs sentiment analysis on Twitter data using machine learning and lexicon-based methods.

## 🔍 Project Overview

This project classifies tweets as **Positive**, **Negative**, or **Neutral** using:
- Machine learning with **Naive Bayes**
- Lexicon-based approaches with **VADER** and **TextBlob**

## 📁 Dataset

- `train.csv` — Labeled tweet dataset for model training.
- `twitter.csv` — Unlabeled tweets used for real-time sentiment prediction.

  
- The dataset should have the following columns:  
- `id`: unique identifier for each tweet  
- `label`: target class (e.g., 0 or 1)  
- `tweet`: original tweet text  
- `clean_tweet`: cleaned/preprocessed tweet text  

## ⚙️ Features

- Tweet preprocessing: handle removal, punctuation cleanup, short word filtering.
- Text normalization with stemming and tokenization.
- Visualization with WordCloud and Hashtag Frequency plots.
- Sentiment classification using:
  - **Multinomial Naive Bayes (BoW features)**
  - **VADER SentimentIntensityAnalyzer**
  - **TextBlob Polarity Analysis**
- Evaluation metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix.

## 🧪 Tech Stack

- Python
- Pandas, NumPy
- Matplotlib, Seaborn, WordCloud
- NLTK, TextBlob
- Scikit-learn (CountVectorizer, Naive Bayes)
- VADER Lexicon (via NLTK)

## 🚀 How to Run

1. Clone the repository.
2. Place `train.csv` and `twitter.csv` in the project directory.
3. Run the notebook or Python file:
   ```bash
   jupyter notebook tweet_sentiment_classifier.ipynb
