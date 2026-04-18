import streamlit as st
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

# Download NLTK data (if not already downloaded)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def load_data(file_path):
    """
    Loads the dataset from a CSV file.
    """
    df = pd.read_csv(file_path, encoding='latin-1')
    # Rename columns for clarity
    df = df.rename(columns={'v1': 'label', 'v2': 'message'})
    # Drop unnecessary columns
    df = df[['label', 'message']]
    # Convert labels to numerical (ham: 0, spam: 1)
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

def preprocess_text(text):
    """
    Performs data preprocessing on the text:
    - Removes punctuation and numbers.
    - Converts text to lowercase.
    - Tokenization.
    - Removes stop words.
    - Stemming.
    """
    # Remove punctuation and numbers
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text) # Remove numbers

    # Convert to lowercase
    text = text.lower()

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stop words and perform stemming
    processed_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]

    return ' '.join(processed_tokens)

def train_model(X_train, y_train):
    """
    Trains a Multinomial Naive Bayes model.
    """
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model and returns accuracy, precision, recall, F1-score, and confusion matrix.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, precision, recall, f1, cm

def main():
    """
    Main function to run the Streamlit GUI and orchestrate the ML pipeline.
    """
    st.set_page_config(page_title="Email Spam Detector", layout="wide")
    st.title("📧 Email Spam Detector")
    st.write("Enter an email message below to classify it as Spam or Ham.")

    # Load data
    df = load_data('spam.csv')

    # Preprocess messages
    df['processed_message'] = df['message'].apply(preprocess_text)

    # Feature Extraction using TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['processed_message'])
    y = df['label']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = train_model(X_train, y_train)

    # Model Evaluation
    accuracy, precision, recall, f1, cm = evaluate_model(model, X_test, y_test)

    st.sidebar.header("Model Evaluation Metrics")
    st.sidebar.write(f"**Accuracy:** {accuracy:.2f}")
    st.sidebar.write(f"**Precision:** {precision:.2f}")
    st.sidebar.write(f"**Recall:** {recall:.2f}")
    st.sidebar.write(f"**F1-Score:** {f1:.2f}")

    st.sidebar.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.sidebar.pyplot(fig)

    st.subheader("Enter Email for Prediction")
    user_input = st.text_area("Paste your email here:", height=200)

    if st.button("Predict"):
        if user_input:
            # Preprocess user input
            processed_input = preprocess_text(user_input)
            # Transform user input using the trained vectorizer
            input_vector = vectorizer.transform([processed_input])
            # Make prediction
            prediction = model.predict(input_vector)

            st.subheader("Prediction Result:")
            if prediction[0] == 1:
                st.error("This is a **SPAM** email.")
            else:
                st.success("This is a **HAM** email.")
        else:
            st.warning("Please enter some text to predict.")

if __name__ == "__main__":
    main()
