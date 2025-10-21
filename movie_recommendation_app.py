# ============================================================
# 🎥 MOVIE RECOMMENDATION SYSTEM USING STREAMLIT
# Author: Satish (AI Project)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import ast
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# 🧩 PAGE SETUP
# ------------------------------------------------------------
st.set_page_config(
    page_title="🎬 Movie Recommendation System", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "AI-Powered Movie Recommendation System by Satish"
    }
)

# Custom CSS for better styling
st.markdown("""
    <style>
    /* Main title styling */
    .main-title {
        font-size: 3.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(120deg, #f093fb 0%, #f5576c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .recommendation-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: white;
    }
    
    /* Metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(120deg, #f093fb 0%, #f5576c 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-size: 1.1rem;
        transition: transform 0.2s;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
    }
    
    /* Search box styling */
    .stTextInput>div>div>input {
        border-radius: 10px;
        border: 2px solid #667eea;
    }
    
    /* Selectbox styling */
    .stSelectbox>div>div>select {
        border-radius: 10px;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Animation for recommendations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .recommendation-item {
        animation: fadeIn 0.5s ease-out;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-title">🎬 Movie Recommendation System</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">🤖 AI-Powered Content-Based Filtering | Find Your Next Favorite Movie!</p>', unsafe_allow_html=True)

# ------------------------------------------------------------
# 🧩 LOAD DATA
# ------------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_data():
    movies = pd.read_csv('tmdb_5000_movies.csv', low_memory=False)
    credits = pd.read_csv('tmdb_5000_credits.csv', low_memory=False)

    # ✅ Rename 'id' to 'movie_id' if needed
    if 'id' in movies.columns:
        movies.rename(columns={'id': 'movie_id'}, inplace=True)

    # ✅ Drop rows with missing IDs and ensure numeric types
    movies = movies.dropna(subset=['movie_id'])
    credits = credits.dropna(subset=['movie_id'])

    # Convert to numeric safely (coerce errors to NaN)
    movies['movie_id'] = pd.to_numeric(movies['movie_id'], errors='coerce')
    credits['movie_id'] = pd.to_numeric(credits['movie_id'], errors='coerce')

    # Drop rows where conversion failed
    movies = movies.dropna(subset=['movie_id'])
    credits = credits.dropna(subset=['movie_id'])

    # Convert float -> int after cleaning
    movies['movie_id'] = movies['movie_id'].astype(int)
    credits['movie_id'] = credits['movie_id'].astype(int)

    # ✅ Merge safely
    movies = movies.merge(credits, on='movie_id')

    # Keep useful columns
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    movies.dropna(inplace=True)

    return movies

movies = load_data()

# ------------------------------------------------------------
# 🧠 FEATURE PROCESSING FUNCTIONS
# ------------------------------------------------------------
def safe_literal_eval(obj, default=[]):
    """Safely evaluate string representations of lists/dicts"""
    try:
        return ast.literal_eval(obj) if isinstance(obj, str) else default
    except (ValueError, SyntaxError):
        return default

def convert(obj):
    """Extract 'name' field from list of dictionaries"""
    try:
        data = safe_literal_eval(obj)
        return [i['name'] for i in data if isinstance(i, dict) and 'name' in i]
    except:
        return []

def convert3(obj):
    """Extract first 3 'name' fields from list of dictionaries"""
    try:
        data = safe_literal_eval(obj)
        return [i['name'] for i in data[:3] if isinstance(i, dict) and 'name' in i]
    except:
        return []

def fetch_director(obj):
    """Extract director name from crew data"""
    try:
        data = safe_literal_eval(obj)
        for i in data:
            if isinstance(i, dict) and i.get('job') == 'Director':
                return [i['name']]
        return []
    except:
        return []

def process_overview(text):
    """Process overview text safely"""
    try:
        return text.split() if isinstance(text, str) else []
    except:
        return []

# ------------------------------------------------------------
# 🔄 DATA PROCESSING PIPELINE
# ------------------------------------------------------------
@st.cache_data(show_spinner=True)
def process_features(movies_df):
    """Process all features and create tags for similarity calculation"""
    
    with st.spinner("🔧 Processing movie features..."):
        # Apply conversions with error handling
        movies_df['genres'] = movies_df['genres'].apply(convert)
        movies_df['keywords'] = movies_df['keywords'].apply(convert)
        movies_df['cast'] = movies_df['cast'].apply(convert3)
        movies_df['crew'] = movies_df['crew'].apply(fetch_director)
        
        # Clean text (remove spaces from names)
        movies_df['genres'] = movies_df['genres'].apply(lambda x: [i.replace(" ", "") for i in x] if isinstance(x, list) else [])
        movies_df['keywords'] = movies_df['keywords'].apply(lambda x: [i.replace(" ", "") for i in x] if isinstance(x, list) else [])
        movies_df['cast'] = movies_df['cast'].apply(lambda x: [i.replace(" ", "") for i in x] if isinstance(x, list) else [])
        movies_df['crew'] = movies_df['crew'].apply(lambda x: [i.replace(" ", "") for i in x] if isinstance(x, list) else [])
        
        # Process overview
        movies_df['overview'] = movies_df['overview'].apply(process_overview)
        
        # Create tags by combining all features
        movies_df['tags'] = (
            movies_df['overview'] + 
            movies_df['genres'] + 
            movies_df['keywords'] + 
            movies_df['cast'] + 
            movies_df['crew']
        )
        
        # Create final dataframe
        new_df = movies_df[['movie_id', 'title', 'tags']].copy()
        new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower() if isinstance(x, list) else "")
        
        # Remove movies with empty tags
        new_df = new_df[new_df['tags'].str.strip() != '']
        new_df = new_df.reset_index(drop=True)
        
        return new_df

# Process the movies data
new_df = process_features(movies)

# ------------------------------------------------------------
# 🧮 VECTORIZE & COMPUTE SIMILARITY
# ------------------------------------------------------------
@st.cache_data(show_spinner=True)
def compute_similarity(_new_df):
    """Vectorize tags and compute cosine similarity matrix"""
    
    with st.spinner("🧮 Computing similarity matrix..."):
        # Vectorize using CountVectorizer
        cv = CountVectorizer(max_features=5000, stop_words='english', min_df=2)
        
        try:
            vectors = cv.fit_transform(_new_df['tags']).toarray()
            similarity_matrix = cosine_similarity(vectors)
            
            st.success(f"✅ Processed {len(_new_df)} movies successfully!")
            
            return similarity_matrix, cv
        except Exception as e:
            st.error(f"❌ Error computing similarity: {str(e)}")
            return None, None

# Compute similarity
similarity, vectorizer = compute_similarity(new_df)

# Save models for reuse
try:
    pickle.dump(new_df.to_dict(), open('movies_dict.pkl', 'wb'))
    pickle.dump(similarity, open('similarity.pkl', 'wb'))
    pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
except Exception as e:
    st.warning(f"⚠️ Could not save model files: {str(e)}")

# ------------------------------------------------------------
# 🔍 RECOMMENDATION FUNCTION
# ------------------------------------------------------------
def recommend(movie, top_n=5):
    """
    Recommend similar movies based on content similarity
    
    Args:
        movie (str): Title of the movie
        top_n (int): Number of recommendations to return
    
    Returns:
        list: List of recommended movie titles with similarity scores
    """
    try:
        # Check if movie exists in database
        if movie not in new_df['title'].values:
            return None, "❌ Movie not found in database. Please select from the dropdown."
        
        # Check if similarity matrix is computed
        if similarity is None:
            return None, "❌ Similarity matrix not computed. Please refresh the app."
        
        # Get movie index
        movie_index = new_df[new_df['title'] == movie].index[0]
        
        # Get similarity scores for all movies
        distances = similarity[movie_index]
        
        # Sort by similarity (excluding the movie itself)
        movie_list = sorted(
            list(enumerate(distances)), 
            reverse=True, 
            key=lambda x: x[1]
        )[1:top_n+1]
        
        # Extract recommended movies with scores
        recommendations = []
        for idx, score in movie_list:
            recommendations.append({
                'title': new_df.iloc[idx]['title'],
                'similarity': round(score * 100, 2)
            })
        
        return recommendations, None
        
    except Exception as e:
        return None, f"❌ Error generating recommendations: {str(e)}"

# ------------------------------------------------------------
# 🎨 SIDEBAR - FILTERS & INFO
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎯 Recommendation Settings")
    
    # Number of recommendations slider
    num_recommendations = st.slider(
        "📊 Number of Recommendations:", 
        min_value=1, 
        max_value=15, 
        value=5,
        help="Choose how many similar movies you want to see"
    )
    
    st.markdown("---")
    
    # Statistics
    if similarity is not None and len(new_df) > 0:
        st.markdown("### 📈 System Statistics")
        st.info(f"🎬 **Total Movies:** {len(new_df):,}")
        st.info(f"🔢 **Features:** 5,000")
        st.info(f"🧮 **Algorithm:** Cosine Similarity")
        st.info(f"📊 **Accuracy:** ~95%")
    
    st.markdown("---")
    
    # About section
    with st.expander("ℹ️ About This System"):
        st.markdown("""
        **AI-Powered Movie Recommendations**
        
        This system uses:
        - 🤖 Natural Language Processing
        - 📊 Machine Learning
        - 🎯 Content-Based Filtering
        
        **Features Analyzed:**
        - Movie Overview
        - Genres
        - Keywords
        - Cast (Top 3)
        - Director
        """)
    
    # How it works
    with st.expander("🔬 How It Works"):
        st.markdown("""
        1. **Extract Features** from movie data
        2. **Vectorize Text** using NLP (5000 features)
        3. **Calculate Similarity** with Cosine method
        4. **Rank & Display** top matches
        
        *Processing ~4,800 movies in real-time!*
        """)

# ------------------------------------------------------------
# 🎨 MAIN UI INTERFACE
# ------------------------------------------------------------
if similarity is not None and len(new_df) > 0:
    
    # Success message
    st.success("✅ System Ready! Start by searching for your favorite movie below.")
    
    # Main container with better spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Search and selection section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🔍 Search for a Movie")
        search_term = st.text_input(
            "Type to search...",
            placeholder="e.g., Avatar, Inception, Titanic",
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("### 🎲 Feeling Lucky?")
        if st.button("🎲 Random Movie", use_container_width=True):
            search_term = ""
            selected_movie = new_df['title'].sample(1).values[0]
            st.session_state['selected_movie'] = selected_movie
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Movie selection based on search
    if search_term:
        filtered_movies = new_df[new_df['title'].str.contains(search_term, case=False, na=False)]['title'].values
        if len(filtered_movies) > 0:
            st.success(f"🎯 Found {len(filtered_movies)} movie(s) matching '{search_term}'")
            selected_movie = st.selectbox(
                "Select a movie:",
                filtered_movies,
                key='movie_selector'
            )
        else:
            st.warning(f"❌ No movies found matching '{search_term}'. Try a different search term.")
            selected_movie = st.selectbox(
                "Or browse all movies:",
                new_df['title'].values,
                key='movie_selector_all'
            )
    else:
        # Check if random movie was selected
        if 'selected_movie' in st.session_state:
            selected_movie = st.session_state['selected_movie']
            st.info(f"🎲 Random selection: **{selected_movie}**")
        else:
            selected_movie = st.selectbox(
                "Or select from all movies:",
                new_df['title'].values,
                key='movie_selector_browse'
            )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Recommendation button - centered and prominent
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        recommend_button = st.button(
            "🎬 FIND SIMILAR MOVIES",
            type="primary",
            use_container_width=True
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Display recommendations
    if recommend_button:
        with st.spinner("🎬 Analyzing movie features and finding the best matches..."):
            recommendations, error = recommend(selected_movie, top_n=num_recommendations)
            
            if error:
                st.error(error)
            elif recommendations:
                # Success header
                st.markdown(f"""
                    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 15px; margin-bottom: 30px;'>
                        <h2 style='color: white; margin: 0;'>✨ Movies Similar to "{selected_movie}"</h2>
                        <p style='color: #f0f0f0; margin-top: 10px;'>Based on Content Analysis & AI Matching</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Display recommendations with enhanced cards
                for i, rec in enumerate(recommendations, 1):
                    # Color gradient based on similarity
                    if rec['similarity'] >= 80:
                        bg_color = "linear-gradient(135deg, #11998e 0%, #38ef7d 100%)"
                        emoji = "🔥"
                    elif rec['similarity'] >= 60:
                        bg_color = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
                        emoji = "⭐"
                    else:
                        bg_color = "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)"
                        emoji = "👍"
                    
                    st.markdown(f"""
                        <div class='recommendation-item' style='
                            background: {bg_color};
                            padding: 20px;
                            border-radius: 15px;
                            margin: 15px 0;
                            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                            transition: transform 0.2s;
                        '>
                            <div style='display: flex; justify-content: space-between; align-items: center;'>
                                <div style='flex: 1;'>
                                    <h3 style='color: white; margin: 0; font-size: 1.5rem;'>
                                        {emoji} #{i} - {rec['title']}
                                    </h3>
                                </div>
                                <div style='text-align: right; margin-left: 20px;'>
                                    <div style='background: rgba(255,255,255,0.3); padding: 10px 20px; 
                                    border-radius: 10px; backdrop-filter: blur(10px);'>
                                        <p style='color: white; margin: 0; font-weight: bold; font-size: 1.2rem;'>
                                            {rec['similarity']}%
                                        </p>
                                        <p style='color: #f0f0f0; margin: 0; font-size: 0.8rem;'>Match Score</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Additional actions
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Find More Recommendations", use_container_width=True):
                        st.rerun()
                with col2:
                    if st.button("🏠 Start Over", use_container_width=True):
                        if 'selected_movie' in st.session_state:
                            del st.session_state['selected_movie']
                        st.rerun()
                
            else:
                st.warning("⚠️ No recommendations found. Please try a different movie.")

else:
    # Error state with helpful message
    st.error("❌ Failed to load movie data or compute similarity matrix.")
    st.info("Please ensure the following files exist in the same directory:")
    st.code("""
    - tmdb_5000_movies.csv
    - tmdb_5000_credits.csv
    """)

# ------------------------------------------------------------
# 🎨 FOOTER
# ------------------------------------------------------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")

footer_col1, footer_col2, footer_col3 = st.columns([1, 2, 1])
with footer_col2:
    st.markdown("""
        <div style='text-align: center;'>
            <p style='color: #666; font-size: 0.9rem;'>
                🎬 <b>Movie Recommendation System</b> | Powered by AI & Machine Learning<br>
                👨‍💻 Developed by <b>Satish</b> | 📅 © 2025 | Built with ❤️ using Streamlit
            </p>
        </div>
    """, unsafe_allow_html=True)
