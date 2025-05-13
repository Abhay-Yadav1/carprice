import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Load all saved components
try:
    model = joblib.load('lr_model.pkl')
    scaler = joblib.load('scaler.pkl')
    pca = joblib.load('pca.pkl')
    numerical_cols = joblib.load('numerical_cols.pkl')
    categorical_info = joblib.load('categorical_info.pkl')
    original_columns = joblib.load('original_columns.pkl')
    df = pd.read_csv('CarPrice_Assignment.csv')  # For reference values
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# Streamlit app
st.title('Car Price Prediction')

def get_user_input():
    data = {}
    
    # Numerical inputs
    for col in numerical_cols:
        data[col] = st.sidebar.slider(
            f'{col}', 
            min_value=float(df[col].min()), 
            max_value=float(df[col].max()), 
            value=float(df[col].median())
        )
    
    # Categorical inputs
    for col, options in categorical_info.items():
        data[col] = st.sidebar.selectbox(f'{col}', options)
    
    # Create DataFrame with correct column order
    input_df = pd.DataFrame(columns=original_columns[:-1])  # Exclude target
    for col in input_df.columns:
        if col in data:
            input_df[col] = [data[col]]
        else:
            input_df[col] = [0]  # Fill missing with 0 (shouldn't happen)
    
    return input_df

def preprocess_input(input_df):
    # Make a copy
    processed = input_df.copy()
    
    # Label encode categorical variables
    for col in categorical_info.keys():
        le = LabelEncoder()
        le.fit(categorical_info[col])
        processed[col] = le.transform(processed[[col]].values.ravel())
    
    # Scale all features
    processed_scaled = scaler.transform(processed)
    
    # Apply PCA transformation
    processed_pca = pca.transform(processed_scaled)
    
    return processed_pca

# Get user input
user_input = get_user_input()

# Show user input
st.subheader('User Input Parameters')
st.write(user_input)

if st.button('Predict Price'):
    try:
        # Preprocess the input
        processed_input = preprocess_input(user_input)
        
        # Make prediction
        prediction = model.predict(processed_input)
        
        st.subheader('Prediction')
        st.write(f'Predicted Car Price: ${prediction[0]:,.2f}')
        
        # Show some context
        st.write('Price Range in Dataset:')
        st.write(f'Minimum: ${df["price"].min():,.2f}')
        st.write(f'Average: ${df["price"].mean():,.2f}')
        st.write(f'Maximum: ${df["price"].max():,.2f}')
        
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.write("Debug info:")
        st.write("Input shape:", processed_input.shape if 'processed_input' in locals() else "Not processed")
        st.write("Model expects:", pca.n_components_, "PCA components")