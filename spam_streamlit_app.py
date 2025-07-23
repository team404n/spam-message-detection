import streamlit as st
import joblib
import string
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

# Load model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
encoder = joblib.load("label_encoder.pkl")

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

# Streamlit UI
st.set_page_config(page_title="Spam Message Classifier", layout="centered")
st.title("📩 Spam Message Classifier")
st.write("Enter a message below to classify it as Spam or Ham (not spam).")

# Input
user_input = st.text_area("Enter your message here:", height=150)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        cleaned = clean_text(user_input)
        vect_msg = vectorizer.transform([cleaned])
        prediction = model.predict(vect_msg)
        label = encoder.inverse_transform(prediction)[0]
        if label == "spam":
            st.error("🚫 This message is likely SPAM.")
        else:
            st.success("✅ This message is likely NOT SPAM (HAM).")
