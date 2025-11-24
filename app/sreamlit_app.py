# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import load_wine

st.title("Wine Dataset Clustering App")

# Load trained model and scaler
kmeans = joblib.load("artifacts/kmeans_model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")

# Load dataset features
data = load_wine()
feature_names = data.feature_names

# Create input sliders dynamically
st.subheader("Input Wine Features")
input_data = []
for feature in feature_names:
    min_val = float(np.min(data.data[:, feature_names.index(feature)]))
    max_val = float(np.max(data.data[:, feature_names.index(feature)]))
    mean_val = float(np.mean(data.data[:, feature_names.index(feature)]))
    val = st.slider(feature, min_val, max_val, mean_val)
    input_data.append(val)

# Convert to array and scale
X_input = np.array([input_data])
X_scaled = scaler.transform(X_input)

# Predict cluster
cluster = kmeans.predict(X_scaled)[0]

st.success(f"This wine sample belongs to **Cluster {cluster}**")
st.info("Clusters are based on chemical properties and group similar wines together.")
