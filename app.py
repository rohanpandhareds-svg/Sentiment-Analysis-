"""Streamlit front end for the Flipkart review sentiment model.

Run:  streamlit run app.py
Needs sentiment_model.h5 and tokenizer.pkl (created by sentiment_analysis.ipynb).
"""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import re
import joblib
import nltk
import streamlit as st
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 200


@st.cache_resource
def load_assets():
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    return load_model("sentiment_model.h5"), joblib.load("tokenizer.pkl")


model, tokenizer = load_assets()
STOPWORDS = set(stopwords.words("english"))
_ws_tokenizer = nltk.tokenize.WhitespaceTokenizer()
_lemmatizer = nltk.stem.WordNetLemmatizer()


def preprocess(text: str) -> str:
    # Must match the preprocessing used in training
    words = [w for w in str(text).split() if w not in STOPWORDS]
    words = [_lemmatizer.lemmatize(w) for w in _ws_tokenizer.tokenize(" ".join(words))]
    text = " ".join(words).lower()
    return re.sub("[^a-zA-Z]", " ", text)


st.set_page_config(page_title="Review Sentiment", page_icon="🛒")
st.title("🛒 Product Review Sentiment")
st.write("Type a product review and the model predicts whether the customer is satisfied.")

review = st.text_area("Review", placeholder="e.g. Great quality, delivered on time")

if st.button("Analyse") and review.strip():
    seq = pad_sequences(tokenizer.texts_to_sequences([preprocess(review)]),
                        padding="post", maxlen=MAX_LEN)
    score = float(model.predict(seq, verbose=0)[0][0])
    if score >= 0.5:
        st.success(f"Positive 😊  (confidence {score:.0%})")
    else:
        st.error(f"Negative 😞  (confidence {1 - score:.0%})")
    st.progress(score)
