import streamlit as st
import pandas as pd
from main import get_hybrid_recommendations

# --- UI setup ---
st.set_page_config(page_title="AI Music DJ", page_icon="🎵")
st.title("AI Music Recommender");
st.write("This app uses a **Hybrid Similarity Model** to find your next jam.")

# --- Load Data ---
def load_list():
    df = pd.read_csv('spotify_songs.csv')
    return sorted(df['track_name'].dropna().astype(str).unique())

song_list = load_list()

# --- User Interaction setup --- 
selected_song = st.selectbox("Type or select a song you like: ", song_list)
num_to_show = st.slider("How many recommendations?", 5, 20, 5)

if st.button("Find Similar Music"):
    with st.spinner("AI is analyzing audio features..."):
        results = get_hybrid_recommendations(selected_song, num_recs = num_to_show)

        if isinstance(results, str):
            st.error(results)
        else:
            st.success(f"Top {num_to_show} matches for you: ")

            #display the results in a table
            st.table(results)

# --- footer ---
st.caption("Powered by SciKit-Learn and Streamlit")