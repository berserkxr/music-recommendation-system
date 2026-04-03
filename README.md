# 🎵 Hybrid Music Recommender

A content-based + collaborative filtering recommendation system built on Spotify audio features. Given a song name, it finds the most sonically similar *and* popular tracks in the dataset.

---

## How It Works

This is a **hybrid recommender** that blends two recommendation strategies:

### 1. Content-Based Filtering
Uses audio features from Spotify's API to measure how sonically similar two songs are. Features used:

| Feature | Description |
|---|---|
| `danceability` | How suitable a track is for dancing (0–1) |
| `energy` | Perceptual intensity and activity (0–1) |
| `key` | Musical key of the track |
| `loudness` | Overall loudness in decibels |
| `mode` | Major (1) or minor (0) |
| `speechiness` | Presence of spoken words (0–1) |
| `acousticness` | Confidence the track is acoustic (0–1) |
| `instrumentalness` | Predicts whether a track has no vocals (0–1) |
| `liveness` | Detects presence of a live audience (0–1) |
| `valence` | Musical positiveness (0–1) |
| `tempo` | Estimated beats per minute |

Similarity is computed using **cosine similarity** — a score of `1.0` means the songs are identical in audio space; `0.0` means they share nothing in common.

### 2. Collaborative Element (Popularity Weighting)
Among similar songs, results are also ranked by `track_popularity`. This reflects the wisdom-of-the-crowd: if many listeners enjoy a sonically similar song, it's a stronger recommendation.

### Hybrid Score
```
final_rank = sort_by(sim_score DESC, track_popularity DESC)
```

The model doesn't simply return the most similar songs — it surfaces songs that are both sonically close **and** well-loved.

---

## Preprocessing

Before any similarity computation, features are normalized using `MinMaxScaler`. This prevents high-range features like `tempo` (e.g. `128 BPM`) from dominating low-range features like `energy` (e.g. `0.85`). After scaling, all features live in the `[0, 1]` range.

```python
scaler = MinMaxScaler()
df_normalized[features] = scaler.fit_transform(df[features])
```

---

## Usage

```python
recommendations = get_hybrid_recommendations("Blinding Lights", num_recs=5)
print(recommendations)
```

**Output:**

| track_name | track_artist | track_album_name | sim_score |
|---|---|---|---|
| Save Your Tears | The Weeknd | After Hours | 0.987 |
| ... | ... | ... | ... |

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `song_name` | `str` | required | Name of the seed song (case-insensitive) |
| `num_recs` | `int` | `5` | Number of recommendations to return |

---

## Requirements

```
pandas
scikit-learn
```

Install with:

```bash
pip install pandas scikit-learn
```

---

## Dataset

Expects a CSV file named `spotify_songs.csv` with at least the following columns:

- `track_name`
- `track_artist`
- `track_album_name`
- `track_popularity`
- All 11 audio feature columns listed above

A compatible dataset is available on [Kaggle — Spotify Song Attributes](https://www.kaggle.com/datasets/geomack/spotifyclassification).

---

## Limitations

- **Exact name matching only** — the lookup uses a simple string equality check. Typos or partial names will return "Song not found."
- **No user history** — the collaborative element is popularity-based, not personalized to individual listening history.
- **Single seed song** — the model recommends from one song at a time; playlist-level seeding is not yet supported.

---

## Potential Improvements

- [ ] Fuzzy song name matching (e.g. using `rapidfuzz`)
- [ ] Multi-song seed input (average the feature vectors)
- [ ] Genre filtering to avoid cross-genre matches
- [ ] True collaborative filtering using user play history
- [ ] Web UI with search autocomplete
