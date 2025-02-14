import streamlit as st
import numpy as np
import pandas as pd
import joblib  # For loading the logistic model
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained logistic regression model
model = joblib.load("logistic_model.pkl")

# Streamlit UI
st.title("Postoperative Bleeding Risk Calculator")

# Input Fields
hb = st.number_input("HASBLED Score", min_value=0, max_value=5, value=1)
c2 = st.selectbox("Alcohol Consumption", ["No", "Yes"])
tai = st.selectbox("Antiplatelet agents", ["No", "Yes"])
oak = st.selectbox("Oral Anticoagulants", ["No", "Yes"])
brdg = st.selectbox("Perioperative Bridging", ["No", "Yes"])
age = st.number_input("Age (years)", min_value=18, max_value=100, value=50)
bmi = st.number_input("BMI", min_value=15.0, max_value=50.0, value=25.0, format="%.1f")
hem = st.number_input("Preoperative Hemoglobin (g/dL)", min_value=5.0, max_value=18.0, value=13.5, format="%.1f")
tc = st.number_input("Preoperative Platelet Count (x10^3/uL)", min_value=50, max_value=600, value=250)
cv = st.number_input("CHA2DS2VASc Score", min_value=0, max_value=9, value=2)

# Convert categorical variables to numerical format
c2 = 1 if c2 == "Yes" else 0
tai = 1 if tai == "Yes" else 0
oak = 1 if oak == "Yes" else 0
brdg = 1 if brdg == "Yes" else 0

# Convert inputs into a NumPy array for prediction
input_data = np.array([[hb, c2, tai, oak, brdg, age, bmi, hem, tc, cv]])

# Make prediction
risk_probability = model.predict_proba(input_data)[0][1]

# Display Risk Estimate
st.markdown(f"### Estimated Risk of Postoperative Bleeding: **{risk_probability * 100:.1f}%**")

# Function to plot Nomogram
def plot_nomogram(coefficients):
    predictors = ["HASBLED", "Alcohol", "Antiplatelet", "Oral Anticoagulants", "Bridging", "Age", "BMI", "Hemoglobin", "Platelets", "CHA2DS2VASc"]
    points = np.abs(coefficients * input_data.flatten())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=points, y=predictors, palette="coolwarm", ax=ax)
    ax.set_xlabel("Nomogram Points")
    ax.set_ylabel("Predictors")
    ax.set_title("Nomogram for Postoperative Bleeding Risk")
    
    st.pyplot(fig)

# Plot Nomogram using model coefficients
plot_nomogram(model.coef_[0])
