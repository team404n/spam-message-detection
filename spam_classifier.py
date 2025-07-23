
import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

nltk.download('stopwords')
from nltk.corpus import stopwords

# Load the dataset
df = pd.read_csv("spam_fixed.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

df['cleaned'] = df['message'].apply(clean_text)
encoder = LabelEncoder()
df['label_num'] = encoder.fit_transform(df['label'])

# Vectorization
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['cleaned'])
y = df['label_num']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Prediction function
def predict_message(msg):
    msg_clean = clean_text(msg)
    msg_vector = vectorizer.transform([msg_clean])
    pred = model.predict(msg_vector)
    return encoder.inverse_transform(pred)[0]

# Test predictions
print(predict_message("Free entry in 2 a wkly comp to win FA Cup final tkts! Text FA to 87121"))
print(predict_message("Hey, are we still meeting for dinner tonight?"))

# Save models
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(encoder, "label_encoder.pkl")
