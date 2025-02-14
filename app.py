import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyreadstat
import joblib
import statsmodels.api as sm

# ---- Load SPSS Data ----
file_path = "data.sav"  # Update with your actual path
df, meta = pyreadstat.read_sav(file_path)

# ---- Preprocess Data ----
df_clean = df.rename(columns={
    "Blutungsevent": "outcome",
    "HASBLED": "hb",
    "Alcohol": "c2",
    "Medikament": "med",
    "Bridging": "brdg"
})

df_clean["outcome"] = df_clean["outcome"].astype(int)
df_clean["c2"] = df_clean["c2"].fillna(0).astype(int)

# Create binary indicators for medications
df_clean["tai"] = df_clean["med"].apply(lambda x: 1 if x in [1, 3, 4, 6] else 0)
df_clean["oak"] = df_clean["med"].apply(lambda x: 1 if x in [2, 3, 5, 6] else 0)
df_clean["brdg"] = df_clean["brdg"].apply(lambda x: 1 if x in [1, 2] else 0)

# Define predictors and outcome
predictors = ["hb", "c2", "tai", "oak", "brdg"]
X = df_clean[predictors]
y = df_clean["outcome"]

# ---- Train Logistic Regression Model ----
X = sm.add_constant(X)
model = sm.Logit(y, X).fit()

# ---- Save Model ----
joblib.dump(model, "logistic_model.pkl")

# ---- Streamlit UI ----
st.title("Postoperative Bleeding Risk Calculator")
st.write("Enter the patient details to estimate the risk of postoperative bleeding.")

# Input Fields
hb = st.number_input("HASBLED Score (hb)", min_value=0, max_value=5, value=1)
c2 = st.selectbox("Alcohol Consumption (c2)", ["No", "Yes"])
tai = st.selectbox("Antiplatelet Agents", ["No", "Yes"])
oak = st.selectbox("Oral Anticoagulants", ["No", "Yes"])
brdg = st.selectbox("Perioperative Bridging", ["No", "Yes"])

# Convert categorical variables
c2 = 1 if c2 == "Yes" else 0
tai = 1 if tai == "Yes" else 0
oak = 1 if oak == "Yes" else 0
brdg = 1 if brdg == "Yes" else 0

# Convert inputs into a NumPy array
input_data = np.array([[1, hb, c2, tai, oak, brdg]])  # 1 for constant

# Make prediction
risk_probability = model.predict(input_data)[0]

# Display Risk Estimate
st.markdown(f"### Estimated Risk of Postoperative Bleeding: **{risk_probability * 100:.1f}%**")

# ---- Plot Nomogram ----
def plot_nomogram(model):
    coefficients = model.params.drop("const")
    predictors = coefficients.index

    # Scale coefficients for visualization
    points = np.abs(coefficients) * 10  # Scale factor
    max_points = points.max()
    scaled_points = (points / max_points) * 100

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Bar plot for each predictor
    sns.barplot(x=scaled_points, y=predictors, palette="coolwarm", ax=ax1)
    
    ax1.set_xlabel("Nomogram Points")
    ax1.set_ylabel("Predictors")
    ax1.set_title("Nomogram for Postoperative Bleeding Risk")
    ax1.grid(axis="x", linestyle="--", alpha=0.7)

    # Overlay probability scale
    ax2 = ax1.twiny()
    total_points_range = np.linspace(0, max_points + 10, 100)
    probabilities = 1 / (1 + np.exp(-(model.params["const"] + coefficients.sum() * total_points_range)))
    ax2.plot(total_points_range, probabilities, color="brown")
    ax2.set_xlabel("Total Nomogram Points → Probability of Bleeding")
    ax2.set_xlim([0, max_points])
    
    st.pyplot(fig)

# Show Nomogram
plot_nomogram(model)
