import streamlit as st
import pandas as pd
import joblib
import os

# Path to the pre-trained model
model_path = 'books_model.pkl'

# Check if the model file exists
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}. Make sure you have trained and saved the model.")
else:
    # Load pre-trained model
    model = joblib.load(model_path)

    # Define feature names
    features = ['pages', 'rating', 'year_published']

    st.title('Book Price Prediction')

    # Create input fields for features
    inputs = {}
    for feature in features:
        if feature == 'rating':
            min_value, max_value = 0.0, 5.0
            mean_value = 3.0
        elif feature == 'year_published':
            min_value, max_value = 1900, 2024
            mean_value = 2000
        else:
            min_value, max_value = 0, 1000
            mean_value = 0

        inputs[feature] = st.slider(feature, min_value, max_value, mean_value)

    # Convert inputs to DataFrame
    input_df = pd.DataFrame([inputs], columns=features)

    # Predict
    if st.button('Predict'):
        prediction = model.predict(input_df)
        st.write(f'Predicted Book Price: ${prediction[0]:.2f}')
