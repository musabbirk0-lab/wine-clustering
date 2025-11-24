# train.py
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import joblib
import os

# ---------------------------
# Load dataset
# ---------------------------
data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)

# ---------------------------
# Preprocessing: scaling
# ---------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create artifacts folder
os.makedirs("artifacts", exist_ok=True)
joblib.dump(scaler, "artifacts/scaler.pkl")

# ---------------------------
# Hyperparameter tuning for KMeans using silhouette score
# ---------------------------
best_score = -1
best_k = 0
best_model = None
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=50)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"Clusters: {k}, Silhouette Score: {score:.3f}")
    if score > best_score:
        best_score = score
        best_k = k
        best_model = kmeans

print(f"\n✅ Best number of clusters: {best_k} (Silhouette Score: {best_score:.3f})")

# ---------------------------
# Save model and cluster labels
# ---------------------------
joblib.dump(best_model, "artifacts/kmeans_model.pkl")

X['Cluster'] = best_model.labels_
X.to_csv("artifacts/wine_clusters.csv", index=False)

# ---------------------------
# Optional: Elbow Method Plot
# ---------------------------
wcss = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=50)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

plt.figure(figsize=(6,4))
plt.plot(range(2,11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.grid(True)
plt.savefig("artifacts/elbow_plot.png")
plt.close()

print("✅ Training completed. Artifacts saved in 'artifacts/' folder.")
