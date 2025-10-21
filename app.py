import ast
import os
import pickle
import difflib
import pandas as pd
import numpy as np
try:
    from flask import Flask, request, jsonify, render_template
    from flask_cors import CORS
except Exception as e:  # Provide a clearer message when dependencies are missing
    raise RuntimeError(
        "Missing required web dependencies: ensure 'flask' and 'flask-cors' are installed. "
        "Run: pip install -r requirements.txt"
    ) from e
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# Configuration
DATA_MOVIES = os.getenv('MOVIES_CSV', 'tmdb_5000_movies.csv')
DATA_CREDITS = os.getenv('CREDITS_CSV', 'tmdb_5000_credits.csv')
PICKLE_DIR = os.getenv('PICKLE_DIR', '.cache')
os.makedirs(PICKLE_DIR, exist_ok=True)

# Load and preprocess data on startup (safer reads)
try:
    movies = pd.read_csv(DATA_MOVIES, low_memory=False)
    credits = pd.read_csv(DATA_CREDITS, low_memory=False)
except FileNotFoundError as e:
    raise RuntimeError(f"Required CSV not found: {e}")

# Ensure id columns are numeric and compatible before merging
movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
credits['movie_id'] = pd.to_numeric(credits['movie_id'], errors='coerce')
# Drop rows with missing ids
movies = movies.dropna(subset=['id'])
credits = credits.dropna(subset=['movie_id'])
movies['id'] = movies['id'].astype(int)
credits['movie_id'] = credits['movie_id'].astype(int)

# Merge on numeric id columns (movies.id <-> credits.movie_id)
movies = movies.merge(credits, left_on='id', right_on='movie_id')

# Keep only needed columns
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Helper converters

def _to_list(text):
    try:
        L = []
        for i in ast.literal_eval(text):
            L.append(i['name'])
        return L
    except Exception:
        return []


def _to_list3(text):
    try:
        L = []
        counter = 0
        for i in ast.literal_eval(text):
            if counter < 3:
                L.append(i['name'])
            counter += 1
        return L
    except Exception:
        return []


def _fetch_director(text):
    L = []
    try:
        for i in ast.literal_eval(text):
            if i.get('job') == 'Director':
                L.append(i.get('name'))
    except Exception:
        pass
    return L


def _collapse(L):
    return [i.replace(' ', '') for i in L]

# Clean and build tags
movies.dropna(inplace=True)

movies['genres'] = movies['genres'].apply(_to_list)
movies['keywords'] = movies['keywords'].apply(_to_list)
movies['cast'] = movies['cast'].apply(_to_list)
movies['cast'] = movies['cast'].apply(lambda x: x[:3])
movies['crew'] = movies['crew'].apply(_fetch_director)

movies['cast'] = movies['cast'].apply(_collapse)
movies['crew'] = movies['crew'].apply(_collapse)
movies['genres'] = movies['genres'].apply(_collapse)
movies['keywords'] = movies['keywords'].apply(_collapse)

movies['overview'] = movies['overview'].apply(lambda x: x.split())
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

new = movies[['movie_id','title','tags']].copy()
new['tags'] = new['tags'].apply(lambda x: " ".join(x))

# Vectorize and compute similarity
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(new['tags']).toarray()
similarity = cosine_similarity(vector)

# Map title to index
title_to_index = pd.Series(new.index, index=new['title']).to_dict()


def _find_title(title):
    """Return best-matching canonical title from available titles or None.

    Strategy: exact case-insensitive -> startswith prefix -> difflib.get_close_matches
    """
    if not title:
        return None
    title_lower = title.strip().lower()
    # exact ci match
    for t in new['title']:
        if t.lower() == title_lower:
            return t
    # prefix search
    for t in new['title']:
        if t.lower().startswith(title_lower):
            return t
    # fuzzy
    matches = difflib.get_close_matches(title, new['title'].tolist(), n=1, cutoff=0.6)
    return matches[0] if matches else None


def recommend_movies(title, n=5):
    """Return list of recommended movie titles for the given title.

    Returns empty list when not found.
    """
    canonical = _find_title(title)
    if not canonical:
        return []
    idx = title_to_index.get(canonical)
    if idx is None:
        return []
    distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])
    recommendations = []
    for i in distances[1:n+1]:
        recommendations.append(new.iloc[i[0]].title)
    return recommendations


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/titles')
def titles():
    """Return titles or search by query param 'q' (prefix search)."""
    q = request.args.get('q', '').strip().lower()
    if not q:
        return jsonify(new['title'].tolist())
    # simple prefix search (case-insensitive)
    results = [t for t in new['title'].tolist() if t.lower().startswith(q)]
    return jsonify(results[:200])


@app.route('/recommend')
def recommend_api():
    title = request.args.get('title', '')
    if not title:
        return jsonify({'error': 'missing title parameter'}), 400
    recs = recommend_movies(title)
    return jsonify({'title': title, 'recommendations': recs})


def _load_or_compute_similarity():
    """Attempt to load vectorizer/similarity from pickle; otherwise compute and cache."""
    vec_path = os.path.join(PICKLE_DIR, 'cv.pkl')
    sim_path = os.path.join(PICKLE_DIR, 'similarity.npy')
    vec = None
    sim = None
    try:
        if os.path.exists(vec_path) and os.path.exists(sim_path):
            with open(vec_path, 'rb') as f:
                vec = pickle.load(f)
            sim = np.load(sim_path)
            return vec, sim
    except Exception:
        pass

    # compute
    vec = CountVectorizer(max_features=5000, stop_words='english')
    vector = vec.fit_transform(new['tags']).toarray()
    sim = cosine_similarity(vector)
    # cache
    try:
        with open(vec_path, 'wb') as f:
            pickle.dump(vec, f)
        np.save(sim_path, sim)
    except Exception:
        pass
    return vec, sim


if __name__ == '__main__':
    # (re)load vectorizer/similarity via helper to speed repeated runs
    cv, similarity = _load_or_compute_similarity()
    title_to_index = pd.Series(new.index, index=new['title']).to_dict()

    PORT = int(os.getenv('PORT', '5000'))
    app.run(host='0.0.0.0', port=PORT, debug=True)
