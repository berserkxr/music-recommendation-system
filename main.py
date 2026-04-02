import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# 1. load the data
df = pd.read_csv('spotify_songs.csv')

# 2. Select numerical features for the AI to listen to
features = ['danceability', 'energy', 'key', 'loudness', 'mode', 
            'speechiness', 'acousticness', 'instrumentalness', 
            'liveness', 'valence', 'tempo']

# 4. Normalize the data
# ensures 'tempo' [120] doesn't overpower 'energy' [0.8]
scaler = MinMaxScaler()
df_normalized = df.copy()
df_normalized[features] = scaler.fit_transform(df[features])

def get_hybrid_recommendations(song_name, num_recs = 5):
    # Find the song in the dataset
    try:
        song_data = df_normalized[df_normalized['track_name'].str.lower() == song_name.lower()].iloc[0]
    except IndexError:
        return "Song not found in the dataset! Try another."
    
    # CONTENT-BASED: Calculate similarity based on audio features
    song_vector = song_data[features].values.reshape(1, -1)
    all_vectors = df_normalized[features].values

    # This creates a list of similarity scores (0 to 1) for every song
    similarities = cosine_similarity(song_vector, all_vectors)[0]

    #COLLABORATIVE ELEMENT: We combine similarity with 'track_popularity'
    #We want songs that are similar but also popular (Hybrid approach)\
    df['sim_score'] = similarities

    # Filter out the original song and sort by similarity
    # Then take out the most popular song among the similar ones
    recommendations = df[df['track_name'].str.lower() != song_name.lower()]
    recommendations = recommendations.sort_values(by=['sim_score', 'track_popularity'], ascending = False)

    return recommendations[['track_name', 'track_artist', 'track_album_name', 'sim_score']].head(num_recs)

# --- TEST IT ---
print("Recommendations for 'Smells like teen spirit': \n")
print(get_hybrid_recommendations("Smells like teen spirit"))