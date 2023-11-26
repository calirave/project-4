from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import random

app = Flask(__name__)
    
# Load the music CSV file
file_path = "data/Holiday_Songs_Spotify.csv"  
df = pd.read_csv(file_path)

# Preprocess the data 
df["track_name"] = df["track_name"].str.lower()
df["artist_name"] = df["artist_name"].str.lower()
df["album_name"] = df["album_name"].str.lower()
pd.set_option('display.max_colwidth', None)

# Combine relevant columns into a single column 
df["combined_features"] = df["track_name"] + '|' + df["artist_name"] + '|' + df["album_name"] 

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the text data
tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
import random

# Function to get recommendations
def get_recommendations(song_input, danceability_category, cosine_sim=cosine_sim):
    index = df[df["track_name"].str.lower() == song_input].index[0]

    danceability_mapping = {
        'low': 0.2,
        'medium': 0.5,
        'high': 0.8
    }

    danceability_threshold = danceability_mapping.get(danceability_category.lower())

    if danceability_threshold is None:
        return "Invalid danceability category. Please choose 'low', 'medium', or 'high'."

    # Filter songs based on danceability threshold
    filtered_df = df[(df['danceability'] >= danceability_threshold) & (df.index != index)]

    # If no songs meet the threshold, return a message
    if filtered_df.empty:
        return f"No songs found with {danceability_category} danceability."

    # Shuffle the indices for randomness
    shuffled_indices = list(filtered_df.index)
    random.shuffle(shuffled_indices)

    # Get recommendations based on the shuffled indices
    recommendations = []
    for shuffled_index in shuffled_indices:
        sim_scores = list(enumerate(cosine_sim[shuffled_index]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[:15]  # Get top 15 recommendations
        song_indices = [i[0] for i in sim_scores]
        recommendations.extend(df[["track_name", "artist_name", "album_name"]].iloc[song_indices].values)

    return recommendations[:15]  # Limit the final recommendations to 15

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle the user input
@app.route('/recommend', methods=['POST'])
def recommend():
    song_input = request.form['song_input'].lower()
    danceability_input = request.form['danceability_input'].lower()

    recommendations = get_recommendations(song_input, danceability_input)

    return render_template('index.html', song_input=song_input, danceability_input=danceability_input,recommendations = recommendations)

if __name__ == '__main__':
    app.run(debug=True)
