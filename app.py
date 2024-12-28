import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os

# Force NLTK to download resources
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.mkdir(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

# Download required NLTK resources
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)

ps = PorterStemmer()

def transform_text(text):
    # Step 1: Convert text to lowercase
    text = text.lower()
    # Step 2: Tokenize text
    text = nltk.word_tokenize(text)

    # Step 3: Remove non-alphanumeric characters
    y = [i for i in text if i.isalnum()]

    # Step 4: Remove stopwords and punctuation
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]

    # Step 5: Perform stemming
    y = [ps.stem(i) for i in y]

    return " ".join(y)

# Load pre-trained models
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError as e:
    st.error(f"File not found: {e}. Please ensure 'vectorizer.pkl' and 'model.pkl' are present.")
    st.stop()

# Streamlit app UI
st.title("Email/SMS Spam Classifier")

# Input area for the user
input_sms = st.text_area("Enter the message")

# Prediction button
if st.button('Predict'):
    if not input_sms.strip():
        st.warning("Please enter a message to classify.")
    else:
        # Step 1: Preprocess the input
        transformed_sms = transform_text(input_sms)

        # Step 2: Vectorize the input
        try:
            vector_input = tfidf.transform([transformed_sms])
        except Exception as e:
            st.error(f"Error during vectorization: {e}")
            st.stop()

        # Step 3: Make prediction
        try:
            result = model.predict(vector_input)[0]
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.stop()

        # Step 4: Display the result
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
