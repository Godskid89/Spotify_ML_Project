import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Random Forest model
@st.cache_data()
def load_model():
    return joblib.load('model/randomforest_model.pkl')

model = load_model()

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('data/universal.csv')

data = load_data()

# App Layout
st.title("Spotify Popularity Prediction App")
st.write("""
This app allows you to Predict whether a song will be popular or not based on it's Audio Features
""")

# Sidebar Navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Choose a section:", ["Data Exploration", "Make Prediction"])

# Section: Data Exploration
if options == "Data Exploration":
    st.header("Data Exploration")
    
    # Display Dataset Overview
    st.subheader("Dataset Overview")
    st.write(data.head())
    
    # Dataset Statistics
    st.subheader("Dataset Statistics")
    st.write(data.describe())
    
    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=False, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Section: Make Prediction
elif options == "Make Prediction":
    st.header("Make a Prediction")
    st.write("Input the song's audio features and metadata to predict its popularity.")

    # User Inputs for audio features
    danceability = st.slider("Danceability", 0.0, 1.0, 0.5, step=0.01)
    energy = st.slider("Energy", 0.0, 1.0, 0.5, step=0.01)
    acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5, step=0.01)
    instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0, step=0.01)
    liveness = st.slider("Liveness", 0.0, 1.0, 0.1, step=0.01)
    speechiness = st.slider("Speechiness", 0.0, 1.0, 0.05, step=0.01)
    tempo = st.number_input("Tempo (BPM)", 0, 250, 120)
    valence = st.slider("Valence (Musical Positivity)", 0.0, 1.0, 0.5, step=0.01)
    key = st.selectbox("Key (0 to 6)", range(0, 7))
    mode = st.selectbox("Mode (0 for minor, 1 for major)", [0, 1])

    # User Input for release date
    release_date = st.date_input("Select the Release Date", value=pd.Timestamp("2020-01-01"))

    # Calculate derived features
    today = pd.Timestamp.today()
    release_date = pd.Timestamp(release_date)  # Convert to Timestamp
    release_day = release_date.day
    release_month = release_date.month
    release_year = release_date.year
    release_recency = (today - release_date).days

    # Prepare input data
    input_data = pd.DataFrame({
        'danceability': [danceability],
        'energy': [energy],
        'acousticness': [acousticness],
        'instrumentalness': [instrumentalness],
        'liveness': [liveness],
        'speechiness': [speechiness],
        'tempo': [tempo],
        'valence': [valence],
        'release_recency': [release_recency],
        'key': [key],
        'mode': [mode],
        'release_day': [release_day],
        'release_month': [release_month],
        'release_year': [release_year]
    })

    # Ensure column order matches the training data
    expected_columns = model.feature_names_in_
    input_data = input_data.reindex(columns=expected_columns)

    # Prediction Button
    if st.button("Predict Popularity"):
        # Predict category and confidence
        predicted_category = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        confidence = np.max(probabilities) * 100

        # Map numeric categories to text labels
        category_mapping = {0: "Low Popularity", 1: "Medium Popularity", 2: "High Popularity"}
        predicted_label = category_mapping.get(predicted_category, "Unknown")

        # Display Results
        st.subheader("Prediction Results")
        st.write(f"**Predicted Popularity Category:** {predicted_label}")
        st.write(f"**Confidence Score:** {confidence:.2f}%")
