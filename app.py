import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("Fake News Detection using Gradient Boosting")

news = st.text_area("Enter News Article")

if st.button("Predict"):
    if news.strip() != "":
        transformed_news = vectorizer.transform([news])
        prediction = model.predict(transformed_news)

        if prediction[0] == 0:
            st.error("Fake News")
        else:
            st.success("Real News")
    else:
        st.warning("Please enter some text.")
