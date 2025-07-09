# 🎥 Movie Rating Prediction with Python

This project aims to predict movie ratings based on features like **genre**, **director**, **runtime**, and **actors** using **regression models**. Built with Python and deployed using Streamlit, the application provides real-time predictions and insights from the IMDb Indian Movies dataset.

---

## 📊 Features

* Predict IMDb-style rating using:

  * Movie genres (multi-label)
  * Director
  * Runtime
  * Number of lead actors
* Show top 5 similar movies from dataset
* Plot actual vs predicted ratings
* Streamlit-based user interface

---

## 📂 Folder Structure

```
movie-rating-prediction/
├── imdb_india_rating_prediction.py       # Main script for data processing and model training
├── model_utils.py                        # Utility functions used by the Streamlit app
├── streamlit_app.py                      # Streamlit web app
├── IMDb_india_movies.csv                 # Raw dataset (from Kaggle)
├── processed_dataset.csv                 # Cleaned dataset used in app
├── ridge_model.pkl                       # Trained Ridge Regression model
├── genre_encoder.pkl                     # MultiLabelBinarizer for genres
├── director_encoder.pkl                  # LabelEncoder for directors
├── rating_comparison.png                 # Graph of actual vs predicted ratings
└── README.md                             # This file
```

---

## 🔧 Installation & Setup

1. Clone this repository:

```bash
git clone https://github.com/your-username/movie-rating-prediction.git
cd movie-rating-prediction
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install pandas scikit-learn streamlit seaborn matplotlib joblib
```

3. Run the training script to generate model files:

```bash
python imd
```
