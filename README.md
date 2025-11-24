
# Wine Clustering – Fully Optimized K-Means

Cluster wine samples using a fully optimized K-Means clustering pipeline on the built-in sklearn Wine dataset. Includes preprocessing, scaling, silhouette-based optimal cluster selection, hyperparameter tuning, elbow visualization, and an interactive Streamlit app for cluster predictions.

---

## 🚀 Features

- Data preprocessing and scaling (StandardScaler)  
- Hyperparameter tuning for KMeans clusters  
- Silhouette Score to select the best number of clusters  
- Elbow Method visualization for reference  
- Saves trained model, scaler, and cluster-labeled dataset  
- Streamlit app for interactive cluster predictions  

---

## 🛠️ Tech Stack

- Python  
- Pandas & NumPy  
- Scikit-learn (KMeans, preprocessing)  
- Matplotlib & Seaborn (visualizations)  
- Streamlit (interactive app)  
- Joblib (save/load model & scaler)  

---



---

## ⚡ Usage

1. Clone the repository:
```bash
git clone <your-github-repo-url>
cd wine-clustering
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train model (optional if using saved artifacts):

```bash
python train.py
```

4. Run Streamlit app:

```bash
streamlit run streamlit_app.py
```

5. Use sliders to input wine features and get cluster prediction.

---

## 📈 Evaluation

* Silhouette Score used to evaluate cluster quality
* Elbow Method visualizes WCSS for reference
* Clusters represent groups of wines with similar chemical properties

---

## 💡 Notes

* Model, scaler, and cluster-labeled dataset are saved in the `artifacts/` folder for deployment
* Streamlit app allows interactive predictions for any new wine sample

