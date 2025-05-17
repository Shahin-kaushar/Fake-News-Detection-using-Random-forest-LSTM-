# streamlit app
from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
import numpy as np

import streamlit as st
import joblib

# Load the trained  model
rf_model = joblib.load("fake_news_rf.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
lstm_model = load_model("fake_news_lstm.keras")
tokenizer = joblib.load("tokenizer.pkl")


# Streamlit UI
st.title("📰 Fake News Detection using LSTM  & Random Forest")
st.write("Enter a news article to check if it's Fake or Real.")

# User Input
user_input = st.text_area("Paste your news article here...")
model_choice = st.selectbox("Choose Model", ["Random Forest", "LSTM"])


import re
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def text_cleaning(text):
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove single characters
    text = re.sub(r'\s+[a-z]\s+', ' ', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove special characters, digits, and punctuation
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Remove multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stopwords
    text = " ".join([word for word in text.split() if word not in stop_words])
    
    return text

def predict_news(text, model_type="RF"):
    if model_type == "RF":
        text_tfidf = vectorizer.transform([text])
        prediction = rf_model.predict(text_tfidf)
    else:
        text_seq = tokenizer.texts_to_sequences([text])
        text_pad = pad_sequences(text_seq, maxlen=200)
        prediction = lstm_model.predict(text_pad)
        prediction = np.round(prediction).astype(int)
    
    return "FAKE" if prediction[0] == 0 else "REAL"



if st.button("Check News"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        # Preprocess input text
        cleaned_text = text_cleaning(user_input)
        
        model_type = "RF" if model_choice == "Random Forest" else "LSTM"
        result = predict_news(user_input, model_type)
        st.subheader(f"Prediction: {result}")

