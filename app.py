import streamlit as st
import joblib
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nlp_utils import clean_text, heuristic_sentiment

st.set_page_config(page_title="Airline Tweet Sentiment Analyzer", page_icon="✈️", layout="centered")

@st.cache_resource
def load_model():
    try:
        return joblib.load("sentiment_model.joblib")
    except FileNotFoundError:
        return None

@st.cache_resource
def load_vader():
    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
        return SentimentIntensityAnalyzer()

st.title("✈️ Airline Tweet Sentiment Analyzer")
st.caption("Classify airline tweets as negative, neutral, or positive.")
with st.sidebar:
    st.header("About this project")
    st.write("An NLP fundamentals project using the Twitter US Airline Sentiment dataset.")
    st.write("Methods: heuristic lexicon, VADER and a trained TF-IDF machine-learning pipeline.")
    st.markdown("[Dataset on Kaggle](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)")

examples = {
    "Positive example": "Thank you for the amazing service! The crew was friendly.",
    "Negative example": "My flight was delayed and customer service was terrible.",
    "Neutral example": "My flight from Boston to New York is scheduled for 8 PM."
}
choice = st.selectbox("Try an example", ["Choose an example"] + list(examples))
default_text = examples.get(choice, "")
tweet = st.text_area("Enter an airline tweet", value=default_text, height=130, max_chars=5000)
approach = st.radio("Choose sentiment approach", ["ML Model", "VADER", "Heuristic"], horizontal=True)

if st.button("Predict sentiment", type="primary"):
    if not tweet.strip():
        st.warning("Please enter a tweet first.")
    else:
        if len(tweet) > 500:
            st.info("Input is longer than 500 characters; only the first 500 characters will be analysed.")
            tweet = tweet[:500]
        cleaned = clean_text(tweet)
        if not cleaned:
            st.warning("The input contains no usable text after cleaning.")
        else:
            st.subheader("Result")
            st.write("**Cleaned text:**", cleaned)
            if approach == "Heuristic":
                label, pos, neg = heuristic_sentiment(tweet)
                st.metric("Predicted sentiment", label.title())
                st.caption(f"Positive lexicon matches: {pos} · Negative lexicon matches: {neg}")
            elif approach == "VADER":
                scores = load_vader().polarity_scores(tweet)
                compound = scores["compound"]
                label = "positive" if compound >= 0.05 else ("negative" if compound <= -0.05 else "neutral")
                st.metric("Predicted sentiment", label.title())
                st.write("VADER scores")
                st.json(scores)
            else:
                model = load_model()
                if model is None:
                    st.error("The trained model is missing. Run NLP_PR1.ipynb through Task 6.7 to create sentiment_model.joblib.")
                else:
                    pred = model.predict([cleaned])[0]
                    st.metric("Predicted sentiment", str(pred).title())
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba([cleaned])[0]
                        st.write("Class probabilities")
                        st.bar_chart(pd.DataFrame({"Probability": probs}, index=model.classes_))
            with st.expander("Part-of-speech tags"):
                try:
                    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
                    nltk.download("averaged_perceptron_tagger", quiet=True)
                    tokens = nltk.word_tokenize(cleaned)
                    tags = nltk.pos_tag(tokens)
                    st.write(pd.DataFrame(tags, columns=["Word", "POS tag"]))
                except Exception as e:
                    st.info(f"POS tagging could not run: {e}")
