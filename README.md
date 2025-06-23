# 📰 Fake News Detection using LSTM & Random Forest

This project aims to **detect fake news articles** using two powerful machine learning models: **LSTM (Long Short-Term Memory)** and **Random Forest**. The pipeline is built using **NLP techniques**, including text preprocessing, tokenization, and vectorization, and is powered by Python libraries like `nltk` and `sklearn`.

---

## 🚀 Key Features

- ✅ Detects whether a news article is **fake or real**
- 🔁 Trained and evaluated using both:
  - Deep learning model: **LSTM**
  - Traditional ML model: **Random Forest**
- 🧠 NLP pipeline:
  - Data cleaning and filtering with **nltk**
  - Tokenization and vectorization of text
- 📊 Evaluation metrics: Accuracy, Confusion Matrix, etc.
- 💻 **Streamlit Web App** for real-time interaction
---

## 🛠️ Tech Stack

| Purpose                 | Libraries / Tools           |
|------------------------|-----------------------------|
| Language               | Python                      |
| Deep Learning          | TensorFlow / Keras (LSTM)   |
| Machine Learning       | scikit-learn (Random Forest)|
| NLP                    | nltk, re                    |
| Data Processing        | pandas, numpy               |
| Vectorization          | CountVectorizer / TF-IDF    |
| Model Evaluation       | sklearn.metrics             |

---

## 🧪 Workflow Overview

1. **Load & inspect the dataset**
2. **Text preprocessing**:
   - Lowercasing
   - Removing punctuation, numbers
   - Removing stopwords
   - Lemmatization (via nltk)
3. **Tokenization and Vectorization**:
   - Using `CountVectorizer` or `TF-IDF`
4. **Model Training**:
   - Train **LSTM** model on tokenized sequences
   - Train **Random Forest** model on vectorized text
5. **Model Evaluation**:
   - Accuracy
   - Confusion Matrix
   - F1-score

---



