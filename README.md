# Movie Recommender Web App

This is a small Flask web frontend for the movie recommender. It loads the TMDB CSVs and serves a simple page where you can type a movie title and get recommendations.

Quick start (Linux/macOS):

1. Create virtual env and install:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Run the app:

```bash
python app.py
```

3. Open http://localhost:5000 in your browser.

Notes:
- The app builds the vectorizer and similarity matrix at startup (takes a few seconds).
- If you want to change behavior (number of recommendations), edit `recommend_movies` in `app.py`.
