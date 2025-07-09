# streamlit_app.py
import streamlit as st
from model_utils import predict_rating, get_similar_movies, mlb, le

st.set_page_config(page_title="🎬 Movie Rating Predictor", layout="centered")

st.title("🎬 Movie Rating Predictor")
st.write("Estimate a movie's rating based on genre, director, runtime, and actors.")

# Genre selector
selected_genres = st.multiselect("Select Genres", mlb.classes_)

# Director dropdown with autocomplete
known_directors = sorted(le.classes_)
director_name = st.selectbox(
    "Select Director Name",
    known_directors,
    index=known_directors.index("Anurag Kashyap") if "Anurag Kashyap" in known_directors else 0
)

# Runtime
runtime = st.slider("Runtime (minutes)", min_value=30, max_value=300, value=140)

# Actors
actor1 = st.text_input("Actor 1")
actor2 = st.text_input("Actor 2")
actor3 = st.text_input("Actor 3")

# Predict button
if st.button("Predict Rating"):
    actors = [actor1, actor2, actor3]
    predicted = predict_rating(selected_genres, director_name, runtime, actors)
    st.success(f"⭐ Predicted IMDb Rating: {predicted}")

    # Show similar movies
    st.subheader("🎞 Top 5 Similar Movies")
    similar = get_similar_movies(selected_genres, runtime)
    st.dataframe(similar.reset_index(drop=True))

# Show actual vs predicted chart
st.subheader("📈 Actual vs Predicted Ratings")
st.image("rating_comparison.png", use_column_width=True)
