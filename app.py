import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Example coefficients (betas) based on literature
coefficients = {
    'intercept': -2.5,
    'age': 0.03,
    'duration_homelessness': -0.04,
    'mental_health': -0.5,
    'employment_status': 0.7,
    'substance_use': -0.6,
    'support_network': 0.5,
    'criminal_history': -0.4
}

# Define the logistic regression model with the coefficients
def predict_success(features):
    # Linear combination of features
    log_odds = (coefficients['intercept'] +
                coefficients['age'] * features['age'] +
                coefficients['duration_homelessness'] * features['duration_homelessness'] +
                coefficients['mental_health'] * features['mental_health'] +
                coefficients['employment_status'] * features['employment_status'] +
                coefficients['substance_use'] * features['substance_use'] +
                coefficients['support_network'] * features['support_network'] +
                coefficients['criminal_history'] * features['criminal_history'])
    
    # Convert log-odds to probability
    probability = 1 / (1 + np.exp(-log_odds))
    return probability

# Streamlit app
st.title("Case Management - Rehousing Prediction")

# SOAP Assessment
st.header("Initial Assessment (SOAP)")
subjective = st.text_area("Subjective (What the client reports)")
objective = st.text_area("Objective (What you observe)")
assessment = st.text_area("Assessment (Your analysis)")
plan = st.text_area("Plan (Next steps)")

# Goal Attainment Scale Assessment
st.header("Goal Attainment Scale (GAS)")
age = st.number_input("Age", min_value=0, max_value=120, value=30)
duration_homelessness = st.number_input("Duration of Homelessness (months)", min_value=0, value=12)
mental_health = st.slider("Mental Health Status (1 = poor, 5 = excellent)", 1, 5, 3)
employment_status = st.selectbox("Employment Status", ["Employed", "Unemployed"])
substance_use = st.selectbox("Substance Use History", ["None", "Occasional", "Regular"])
support_network = st.selectbox("Support Network", ["None", "Weak", "Strong"])
criminal_history = st.selectbox("Criminal History", ["None", "Minor", "Significant"])

# Map categorical variables to numerical
employment_status_map = {"Employed": 1, "Unemployed": 0}
substance_use_map = {"None": 0, "Occasional": 1, "Regular": 2}
support_network_map = {"None": 0, "Weak": 1, "Strong": 2}
criminal_history_map = {"None": 0, "Minor": 1, "Significant": 2}

# Prepare features for prediction
features = {
    'age': age,
    'duration_homelessness': duration_homelessness,
    'mental_health': mental_health,
    'employment_status': employment_status_map[employment_status],
    'substance_use': substance_use_map[substance_use],
    'support_network': support_network_map[support_network],
    'criminal_history': criminal_history_map[criminal_history]
}

# Predict rehousing success probability
probability = predict_success(features)

st.write(f"Probability of successful rehousing: {probability:.2f}")

# Additional code to handle model training and real data integration would be included here. 