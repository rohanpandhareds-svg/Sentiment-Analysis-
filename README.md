# Flipkart Review Sentiment Analysis: from ML model to front end

**Research internship · IISc Bangalore (RBCCPS) × strategicERP · May–Jul 2024**
Guide: Prof. Ravi Kumar

strategicERP wanted to see how a machine-learning model could be taken out of a
notebook and put in front of users. We used sentiment analysis of e-commerce
product reviews as the test case: an NLP model that predicts whether a customer is
satisfied from what they write, served through a simple web app.

<!-- Add a screenshot of the Streamlit app here: ![App demo](app.png) -->

## Results

| Metric | Score |
|---|---|
| Accuracy | **~91%** |
| F1 score | **~0.94** |
| ROC-AUC | **~0.91** |

Trained on **1.9 lakh+ Flipkart reviews** (189,853 after cleaning).
<!-- After re-running the notebook, add: "Majority-class baseline: XX%" -->

## How it works

```
Review text ──► Clean & lemmatize ──► Tokenize ──► Word2Vec embeddings ──► Bi-LSTM ──► Positive / Negative
                                                                                        │
                                                                     Streamlit app ◄────┘
```

1. **Data:** Flipkart Product Reviews (Kaggle): review summary + 1–5 star rating.
   Ratings 4–5 are labelled *positive*, 1–3 *negative*.
2. **Preprocessing (NLTK):** stopword removal, lemmatization (chosen over stemming
   because it keeps real root words, e.g. *easy* not *easi*), lowercasing, and removal
   of non-alphabetic characters.
3. **Features:** Keras tokenizer (1,000-word vocabulary, sequences padded to 200) and
   128-dimensional **Word2Vec** embeddings trained on the review corpus.
4. **Model:** frozen embedding layer → **Bidirectional LSTM (64)** → Dropout (0.5) →
   Dense (64) → Dense (32) → sigmoid output. An LSTM keeps context across a sentence,
   which a basic RNN tends to forget.
5. **Front end:** a **Streamlit** app loads the saved model and tokenizer and gives an
   instant prediction for any review typed in.

## Repository

| File | What it is |
|---|---|
| `sentiment_analysis.ipynb` | Data cleaning, preprocessing, training, evaluation, saving the model |
| `app.py` | Streamlit front end |
| `requirements.txt` | Python dependencies |

## Run it

```bash
pip install -r requirements.txt
# 1. Download flipkart_product.csv from Kaggle (link below) into this folder
# 2. Run sentiment_analysis.ipynb → creates sentiment_model.h5 and tokenizer.pkl
streamlit run app.py
```

Dataset: Flipkart Product Reviews on Kaggle: ADD DATASET LINK
(not included in this repo).

## Tech stack
Python · Pandas · NLTK · Gensim (Word2Vec) · TensorFlow / Keras · scikit-learn · Streamlit

## What I'd do next
- Compare against a simple TF-IDF + Logistic Regression baseline
- Handle class imbalance (class weights) and add a *neutral* class for 3-star reviews
- Serve the model through an API instead of Streamlit for production use
- Support Hinglish and regional-language reviews

*Code cleaned up for publication in 2026; model and approach are from the 2024 internship.*
