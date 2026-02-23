import streamlit as st
import pickle
import re
import pandas as pd

# ----------------------------
# Load trained model & vectorizer
# ----------------------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ----------------------------
# Text Cleaning Function
# ----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\d', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ----------------------------
# App UI
# ----------------------------
st.title("📰 Fake News Detection System")

st.write("Enter a news article below to check whether it is Fake or Real.")

user_input = st.text_area("Enter News Article")

if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned_text = clean_text(user_input)
        vectorized_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_text)

        if prediction[0] == 1:
            st.success("✅ This appears to be Real News.")
        else:
            st.error("❌ This appears to be Fake News.")
