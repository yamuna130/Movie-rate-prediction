# imdb_india_rating_prediction.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
raw_df = pd.read_csv("IMDb_india_movies.csv", encoding='latin1')
print(raw_df.shape)
print(raw_df.columns)

# Clean and rename columns
raw_df.rename(columns={
    'Duration': 'Runtime',
    'Genre': 'Genres',
    'Rating': 'Rating',
    'Votes': 'Votes',
    'Director': 'Director',
    'Actor 1': 'Actor1',
    'Actor 2': 'Actor2',
    'Actor 3': 'Actor3'
}, inplace=True)

# Drop missing values in relevant columns
clean_df = raw_df.dropna(subset=['Genres', 'Director', 'Rating', 'Runtime'])

# Convert Runtime like '163 min' to number
clean_df['Runtime'] = clean_df['Runtime'].str.extract(r'(\d+)').astype(float)

# Convert Genres to list
clean_df['Genres'] = clean_df['Genres'].str.split(', ')

# Count non-null actors
clean_df['ActorCount'] = clean_df[['Actor1', 'Actor2', 'Actor3']].notnull().sum(axis=1)

# One-hot encode genres
mlb = MultiLabelBinarizer()
genre_encoded = mlb.fit_transform(clean_df['Genres'])

# Encode Director
le = LabelEncoder()
director_encoded = le.fit_transform(clean_df['Director'])

# Feature Matrix
X = np.hstack((genre_encoded, director_encoded.reshape(-1, 1), clean_df['Runtime'].values.reshape(-1, 1), clean_df['ActorCount'].values.reshape(-1, 1)))
y = clean_df['Rating'].values

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Ridge Regression Model
model = Ridge()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\nModel Evaluation:")
print("Mean Squared Error (MSE):", round(mean_squared_error(y_test, y_pred), 3))
print("R² Score:", round(r2_score(y_test, y_pred), 3))

# Save Model and Encoders
joblib.dump(model, 'ridge_model.pkl')
joblib.dump(mlb, 'genre_encoder.pkl')
joblib.dump(le, 'director_encoder.pkl')

# Save cleaned dataset for Streamlit
clean_df[['Name', 'Year', 'Genres', 'Runtime', 'Rating']].to_csv("processed_dataset.csv", index=False)
print("\n✅ Model and encoders saved for Streamlit app.")

# Plot actual vs predicted ratings and save the image
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Actual vs Predicted IMDb Ratings")
plt.grid(True)
plt.tight_layout()
plt.savefig("rating_comparison.png")
print("📈 Graph saved: rating_comparison.png")
