
```markdown
# Customer Segmentation with K-Means Clustering

Cluster mall customers based on age, income, and spending score to identify meaningful segments for targeted marketing. This project includes preprocessing, scaling, optimal cluster selection, K-Means clustering, and deployment via Streamlit for interactive predictions.

---

## 🚀 Features

- **Data Preprocessing:** Handle numeric features and scale them for clustering.  
- **Optimal Cluster Selection:** Elbow Method & Silhouette Score to choose the best number of clusters.  
- **Clustering:** K-Means with 5 optimized clusters.  
- **Visualization:** 3D scatter plot of clusters.  
- **Deployment:** Streamlit app for interactive cluster predictions.

---

## 🛠️ Tech Stack

- **Python**  
- **Pandas & NumPy** (data manipulation)  
- **Scikit-learn** (KMeans, preprocessing, scaling)  
- **Matplotlib & Seaborn** (visualization)  
- **Streamlit** (interactive web app)  
- **Joblib** (save/load model & scaler)

---



---

## ⚡ Usage

1. Clone the repository:
```bash
git clone <your-github-repo-url>
cd customer-clustering
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run streamlit_clustering.py
```

4. Input customer details in the app and get predicted cluster.

---

## 📈 Evaluation

* Clusters are validated using **Silhouette Score**.
* Visualized in 3D for age, annual income, and spending score.
* Helps identify high-value or low-engagement customer segments.

---

## 💡 Notes

* Preprocessing and scaling are saved using **Joblib** for deployment.
* Streamlit app is fully interactive for real-time cluster predictions.
* Dataset used: [Mall Customers Dataset](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python)

