# 🎵 Amazon Reviews Recommender System

This project implements an **item-based collaborative filtering** recommendation system using user reviews from Amazon. It explores various similarity measures to compute rating predictions and evaluates both the **accuracy (RMSE)** and **efficiency (execution time)** of the algorithms, with and without the use of **clustering**.

---

## 📚 Overview

The goal is to predict missing user-item ratings by comparing items based on similarity metrics derived from user rating patterns. The system:

- Computes item similarity using multiple distance measures.
- Predicts ratings using utility matrix information.
- Optionally applies clustering to reduce computational cost.
- Evaluates performance using **Root Mean Squared Error (RMSE)**.
- Measures runtime to assess algorithmic efficiency.

---

## 📁 Project Structure

.
├── Apertura.py # Converts raw .json Amazon data to .csv
├── finale.py # Main script for training and evaluating recommender
├── Digital_Music.csv # Example dataset from Amazon
├── utils/ # Utility functions (optional refactor)
│ └── similarity.py # Similarity/distance measures
├── README.md # Project documentation



## 🧪 Features

- **Prediction algorithm** based on item similarity.
- Supports 4 similarity measures:
  - Cosine Centered Similarity (CCS)
  - Jaccard Similarity (JS)
  - Jaccard for Bags (JSB)
  - Euclidean Distance (ED)
- **Optional clustering** for performance optimization.
- RMSE computation for evaluating prediction quality.
- Execution time tracking for performance benchmarking.

---

## 📊 Results

### RMSE (Accuracy)
| Similarity | Without Clustering | With Clustering |
|------------|--------------------|-----------------|
| CCS        | 0.04876            | 0.11444         |
| JS         | 0.04772            | 0.11428         |
| JSB        | 0.04795            | —               |
| ED         | 0.04749            | 0.11478         |

### Execution Time (in seconds)
| Similarity | Without Clustering | With Clustering |
|------------|--------------------|-----------------|
| CCS        | 51.92s             | 67.14s          |
| JS         | 322.16s            | 252.49s         |
| JSB        | 2826.77s           | —               |
| ED         | 94.07s             | 85.48s          |

---

## 🚀 Running the Code

1. Install Python 3.9+
2. Install required packages:

```bash
pip install pandas numpy scipy

Convert raw dataset (if needed):
python Apertura.py

Run the recommender system:
python finale.py

Output includes predicted ratings, RMSE scores, and execution time per similarity method.

🔗 Dataset Source
Default: Digital_Music.csv

Download more: https://nijianmo.github.io/amazon/index.html

⚠️ Must be in .csv format — use Apertura.py to convert.

👤 Author
Anna Gotti
📧 anna.gotti16@gmail.com










