# UK-Traffic-Analysis-PySpark
UK Traffic Analysis using PySpark Big data analytics project using Apache Spark to perform data preprocessing, exploratory data analysis, clustering, and supervised machine learning on UK traffic volume data.


# UK Traffic Analysis using PySpark

## Big Data Analytics Project – Apache Spark

This repository contains a complete big data analytics pipeline implemented using **PySpark** on UK traffic volume data.  
The project covers data preprocessing, exploratory analysis, clustering, and supervised machine learning.

---

## 📊 Dataset
**UK_Traffic_Data_Clean.csv**

The dataset includes yearly traffic counts across UK regions, road types, and vehicle categories.

---

## 🛠 Technologies Used
- Python
- Apache Spark (PySpark)
- Spark MLlib
- Big Data Analytics

---

## 📁 Project Structure



---

## 🔹 Task 1: Data Preparation
- Dropping irrelevant columns
- Renaming columns for Spark compatibility
- Handling missing values using mean imputation
- Outlier removal using IQR method
- Feature scaling using MinMaxScaler

**File:** `task1_data_preparation.py`

---

## 🔹 Task 2: Exploratory Data Analysis (EDA)
- Total traffic volume by year
- Region-wise traffic distribution
- Vehicle type statistics
- Road type traffic comparison

**File:** `task2_exploratory_data_analysis.py`

---

## 🔹 Task 3: Unsupervised Learning (Clustering)
- Feature vectorization and scaling
- K-Means clustering
- Cluster evaluation using Silhouette Score
- Traffic pattern segmentation

**File:** `task3_clustering_kmeans.py`

---

## 🔹 Task 4: Supervised Learning
- Pseudo-label generation using K-Means
- Train-test split
- Decision Tree classification
- Model evaluation using accuracy metric

**File:** `task4_supervised_learning.py`

---

## ▶️ How to Run
Ensure Spark is installed and run any task using:

```bash
spark-submit src/taskX_filename.py

