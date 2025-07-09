# model_utils.py
import pandas as pd
import numpy as np
import joblib
import os
if not os.path.exists("processed_dataset.csv"):
    raise FileNotFoundError("You must run imdb_india_rating_prediction.py first to generate processed_dataset.csv")

# Load saved components
model = joblib.load('ridge_model.pkl')
mlb = joblib.load('genre_encoder.pkl')
le = joblib.load('director_encoder.pkl')
df_full = pd.read_csv("processed_dataset.csv")
df_full['Genres'] = df_full['Genres'].apply(eval)  # Convert string back to list

# Prediction function
def predict_rating(genres_list, director_name, runtime, actor_list):
    genre_vec = [1 if g in genres_list else 0 for g in mlb.classes_]
    try:
        director_code = le.transform([director_name])[0]
    except:
        director_code = -1
    actor_count = len([a for a in actor_list if a])
    features = np.array(genre_vec + [director_code, runtime, actor_count]).reshape(1, -1)
    return round(model.predict(features)[0], 2)

# Top 5 similar movies
def get_similar_movies(genres_list, runtime, top_n=5):
    df_copy = df_full.copy()
    df_copy['similarity'] = df_copy['Genres'].apply(lambda g: len(set(g) & set(genres_list)))
    df_copy['runtime_diff'] = abs(df_copy['Runtime'] - runtime)
    df_copy['score'] = df_copy['similarity'] - df_copy['runtime_diff'] / 100
    similar = df_copy.sort_values('score', ascending=False).head(top_n)
    return similar[['Name', 'Year', 'Genres', 'Runtime', 'Rating']]
