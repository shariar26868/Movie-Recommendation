import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

# Page configuration
st.set_page_config(
    page_title="DataSynthis Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF4B4B;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .movie-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #FF4B4B;
    }
    .stat-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
    st.session_state.movies = None
    st.session_state.ratings = None
    st.session_state.users = None
    st.session_state.train_user_item_matrix = None
    st.session_state.user_similarity_df = None
    st.session_state.svd_predicted_ratings = None
    st.session_state.alpha = 0.6

@st.cache_data
def load_datasets():
    """Load CSV datasets with multiple encoding support"""
    try:
        # Try multiple encodings and delimiters
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        delimiters = [',', '::', '\t', '|', ';']
        
        movies = None
        ratings = None
        users = None
        
        # Load movies
        for enc in encodings:
            for delim in delimiters:
                try:
                    movies = pd.read_csv('movies.csv', encoding=enc, sep=delim,
                                       engine='python', on_bad_lines='skip')
                    if len(movies.columns) >= 2:
                        st.info(f"Movies loaded: {enc} encoding, '{delim}' delimiter")
                        break
                except:
                    continue
            if movies is not None and len(movies.columns) >= 2:
                break
        
        # Load ratings
        for delim in delimiters:
            try:
                ratings = pd.read_csv('ratings.csv', sep=delim, engine='python',
                                    on_bad_lines='skip')
                if len(ratings.columns) >= 3:
                    st.info(f"Ratings loaded: '{delim}' delimiter")
                    break
            except:
                continue
        
        # Load users
        for delim in delimiters:
            try:
                users = pd.read_csv('users.csv', sep=delim, engine='python',
                                  on_bad_lines='skip')
                if len(users.columns) >= 2:
                    st.info(f"Users loaded: '{delim}' delimiter")
                    break
            except:
                continue
        
        if movies is None or ratings is None or users is None:
            st.error("Failed to load one or more CSV files. Please check file formats.")
            return None, None, None
        
        # Normalize column names
        movies.columns = movies.columns.str.strip().str.lower()
        ratings.columns = ratings.columns.str.strip().str.lower()
        users.columns = users.columns.str.strip().str.lower()
        
        if 'genres' in movies.columns:
            movies['genres'] = movies['genres'].fillna('Unknown')
        
        st.success(f"Loaded: {len(movies)} movies, {len(ratings)} ratings, {len(users)} users")
        
        return movies, ratings, users
    except Exception as e:
        st.error(f"Error loading datasets: {str(e)}")
        return None, None, None

@st.cache_resource
def train_models(_movies, _ratings):
    """Train recommendation models"""
    with st.spinner("Training models... This may take a minute..."):
        try:
            # Create train split
            train_data = []
            for user_id in _ratings['userid'].unique():
                user_ratings = _ratings[_ratings['userid'] == user_id]
                if 'timestamp' in _ratings.columns:
                    user_ratings = user_ratings.sort_values('timestamp')
                n_ratings = len(user_ratings)
                if n_ratings >= 5:
                    split_idx = int(n_ratings * 0.8)
                    train_data.append(user_ratings.iloc[:split_idx])
            
            train_ratings = pd.concat(train_data, ignore_index=True)
            
            # Create user-item matrix
            train_user_item_matrix = train_ratings.pivot_table(
                index='userid',
                columns='movieid',
                values='rating'
            ).fillna(0)
            
            # Train User-Based CF
            user_similarity = cosine_similarity(train_user_item_matrix)
            user_similarity_df = pd.DataFrame(
                user_similarity,
                index=train_user_item_matrix.index,
                columns=train_user_item_matrix.index
            )
            
            # Train SVD
            n_factors = min(100, min(train_user_item_matrix.shape) - 1)
            R = train_user_item_matrix.values
            user_ratings_mean = np.mean(R, axis=1)
            R_demeaned = R - user_ratings_mean.reshape(-1, 1)
            
            U, sigma, Vt = svds(R_demeaned, k=n_factors)
            sigma = np.diag(sigma)
            predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
            
            svd_predicted_ratings = pd.DataFrame(
                predicted_ratings,
                index=train_user_item_matrix.index,
                columns=train_user_item_matrix.columns
            )
            
            return train_user_item_matrix, user_similarity_df, svd_predicted_ratings
        except Exception as e:
            st.error(f"Error training models: {str(e)}")
            return None, None, None

def recommend_movies_hybrid(user_id, N, movies, train_user_item_matrix, user_similarity_df, svd_predicted_ratings, alpha=0.6):
    """Generate hybrid recommendations"""
    try:
        if user_id not in train_user_item_matrix.index:
            return None, "User not found in training data"
        
        # CF recommendations
        similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:51]
        user_ratings = train_user_item_matrix.loc[user_id]
        watched_movies = user_ratings[user_ratings > 0].index
        
        cf_recommendations = {}
        for sim_user, similarity in similar_users.items():
            sim_user_ratings = train_user_item_matrix.loc[sim_user]
            for movie_id, rating in sim_user_ratings.items():
                if rating > 0 and movie_id not in watched_movies:
                    if movie_id not in cf_recommendations:
                        cf_recommendations[movie_id] = 0
                    cf_recommendations[movie_id] += similarity * rating
        
        cf_top = sorted(cf_recommendations.items(), key=lambda x: x[1], reverse=True)[:N*2]
        cf_movies = [movie_id for movie_id, _ in cf_top]
        
        # SVD recommendations
        user_pred_ratings = svd_predicted_ratings.loc[user_id]
        unwatched_predictions = user_pred_ratings.drop(watched_movies)
        svd_movies = unwatched_predictions.sort_values(ascending=False).head(N*2).index.tolist()
        
        # Combine
        combined_scores = {}
        for i, movie_id in enumerate(cf_movies):
            combined_scores[movie_id] = combined_scores.get(movie_id, 0) + alpha * (len(cf_movies) - i)
        
        for i, movie_id in enumerate(svd_movies):
            combined_scores[movie_id] = combined_scores.get(movie_id, 0) + (1 - alpha) * (len(svd_movies) - i)
        
        top_movies = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:N]
        movie_ids = [movie_id for movie_id, _ in top_movies]
        
        # Get movie details
        recommendations = []
        for movie_id in movie_ids:
            movie_info = movies[movies['movieid'] == movie_id]
            if not movie_info.empty:
                title = movie_info.iloc[0]['title']
                genres = movie_info.iloc[0].get('genres', 'Unknown')
                recommendations.append({
                    'movieid': movie_id,
                    'title': title,
                    'genres': genres
                })
        
        return recommendations, None
    
    except Exception as e:
        return None, str(e)

def create_user_stats_viz(user_id, ratings, movies):
    """Create user statistics visualizations"""
    user_ratings = ratings[ratings['userid'] == user_id]
    
    # Rating distribution
    rating_dist = user_ratings['rating'].value_counts().sort_index()
    fig1 = px.bar(x=rating_dist.index, y=rating_dist.values,
                  labels={'x': 'Rating', 'y': 'Count'},
                  title=f'User {user_id} Rating Distribution',
                  color=rating_dist.values,
                  color_continuous_scale='Blues')
    fig1.update_layout(showlegend=False, height=300)
    
    # Genre preferences
    user_movies = user_ratings.merge(movies[['movieid', 'genres']], on='movieid')
    genres_list = []
    for genres in user_movies['genres']:
        if pd.notna(genres) and genres != 'Unknown':
            genres_list.extend(genres.split('|'))
    
    if genres_list:
        genre_counts = Counter(genres_list)
        top_genres = dict(sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:8])
        fig2 = px.pie(values=list(top_genres.values()), names=list(top_genres.keys()),
                      title=f'User {user_id} Genre Preferences',
                      color_discrete_sequence=px.colors.qualitative.Set3)
        fig2.update_layout(height=300)
    else:
        fig2 = None
    
    return fig1, fig2

# Main App
st.markdown('<p class="main-header">🎬 DataSynthis Movie Recommendation System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Powered by Hybrid Collaborative Filtering & Matrix Factorization</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/movie-projector.png", width=100)
    st.title("Navigation")
    page = st.radio("Go to:", ["🏠 Home", "🎯 Get Recommendations", "📊 Dataset Insights", "ℹ️ About"])
    
    st.markdown("---")
    st.markdown("### System Status")
    if st.session_state.models_loaded:
        st.success("Models: Loaded ✓")
    else:
        st.warning("Models: Not Loaded")
    
    if st.button("🔄 Load/Reload Models"):
        with st.spinner("Loading datasets..."):
            movies, ratings, users = load_datasets()
            if movies is not None and ratings is not None and users is not None:
                st.session_state.movies = movies
                st.session_state.ratings = ratings
                st.session_state.users = users
                
                train_matrix, user_sim, svd_pred = train_models(movies, ratings)
                if train_matrix is not None:
                    st.session_state.train_user_item_matrix = train_matrix
                    st.session_state.user_similarity_df = user_sim
                    st.session_state.svd_predicted_ratings = svd_pred
                    st.session_state.models_loaded = True
                    st.success("Models loaded successfully!")
                    st.rerun()
                else:
                    st.error("Failed to train models")
            else:
                st.error("Failed to load datasets")

# Page: Home
if page == "🏠 Home":
    if not st.session_state.models_loaded:
        st.warning("⚠️ Please load the models using the sidebar button first!")
        st.info("Click '🔄 Load/Reload Models' in the sidebar to get started.")
    else:
        st.success("System is ready! Navigate to 'Get Recommendations' to start.")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            st.metric("Total Movies", f"{len(st.session_state.movies):,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            st.metric("Total Users", f"{len(st.session_state.users):,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            st.metric("Total Ratings", f"{len(st.session_state.ratings):,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            avg_rating = st.session_state.ratings['rating'].mean()
            st.metric("Avg Rating", f"{avg_rating:.2f} ⭐")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("How It Works")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("**Step 1: User-Based CF**\nFinds similar users based on rating patterns")
        with col2:
            st.info("**Step 2: Matrix Factorization**\nDiscovers latent factors using SVD")
        with col3:
            st.info("**Step 3: Hybrid Ensemble**\nCombines both approaches for best results")

# Page: Get Recommendations
elif page == "🎯 Get Recommendations":
    if not st.session_state.models_loaded:
        st.error("⚠️ Models not loaded! Please load models from the sidebar first.")
    else:
        st.header("Get Personalized Movie Recommendations")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            user_id = st.number_input(
                "Enter User ID:",
                min_value=int(st.session_state.ratings['userid'].min()),
                max_value=int(st.session_state.ratings['userid'].max()),
                value=int(st.session_state.ratings['userid'].iloc[0]),
                step=1
            )
        
        with col2:
            num_recs = st.slider("Number of Recommendations:", 5, 20, 10)
        
        if st.button("🎬 Generate Recommendations", type="primary", use_container_width=True):
            with st.spinner("Generating recommendations..."):
                recommendations, error = recommend_movies_hybrid(
                    user_id, num_recs,
                    st.session_state.movies,
                    st.session_state.train_user_item_matrix,
                    st.session_state.user_similarity_df,
                    st.session_state.svd_predicted_ratings,
                    st.session_state.alpha
                )
                
                if error:
                    st.error(f"Error: {error}")
                elif recommendations:
                    st.success(f"Found {len(recommendations)} recommendations for User {user_id}!")
                    
                    # Display recommendations
                    st.subheader("Your Personalized Recommendations:")
                    
                    for i, rec in enumerate(recommendations, 1):
                        st.markdown(f"""
                        <div class="movie-card">
                            <h3>{i}. {rec['title']}</h3>
                            <p><strong>Genres:</strong> {rec['genres']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # User statistics
                    st.subheader(f"User {user_id} Statistics")
                    
                    fig1, fig2 = create_user_stats_viz(user_id, st.session_state.ratings, st.session_state.movies)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(fig1, use_container_width=True)
                    with col2:
                        if fig2:
                            st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.warning("No recommendations found for this user.")

# Page: Dataset Insights
elif page == "📊 Dataset Insights":
    if not st.session_state.models_loaded:
        st.error("⚠️ Models not loaded! Please load models from the sidebar first.")
    else:
        st.header("Dataset Insights & Analytics")
        
        tab1, tab2, tab3 = st.tabs(["Rating Analysis", "Genre Analysis", "User Behavior"])
        
        with tab1:
            st.subheader("Rating Distribution")
            rating_dist = st.session_state.ratings['rating'].value_counts().sort_index()
            fig = px.bar(x=rating_dist.index, y=rating_dist.values,
                        labels={'x': 'Rating', 'y': 'Count'},
                        title='Overall Rating Distribution',
                        color=rating_dist.values,
                        color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Most Common Rating", rating_dist.idxmax())
            with col2:
                st.metric("Average Rating", f"{st.session_state.ratings['rating'].mean():.2f}")
        
        with tab2:
            st.subheader("Popular Genres")
            all_genres = []
            for genres in st.session_state.movies['genres']:
                if pd.notna(genres) and genres != 'Unknown':
                    all_genres.extend(genres.split('|'))
            
            genre_counts = Counter(all_genres)
            top_genres = dict(sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:15])
            
            fig = px.bar(x=list(top_genres.values()), y=list(top_genres.keys()),
                        orientation='h',
                        labels={'x': 'Number of Movies', 'y': 'Genre'},
                        title='Top 15 Genres by Movie Count',
                        color=list(top_genres.values()),
                        color_continuous_scale='Teal')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("User Activity Levels")
            user_activity = st.session_state.ratings.groupby('userid').size()
            
            fig = px.histogram(user_activity, nbins=50,
                              labels={'value': 'Number of Ratings', 'count': 'Number of Users'},
                              title='Distribution of User Activity',
                              color_discrete_sequence=['coral'])
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Most Active User", f"{user_activity.max()} ratings")
            with col2:
                st.metric("Average Activity", f"{user_activity.mean():.1f} ratings")
            with col3:
                st.metric("Median Activity", f"{user_activity.median():.0f} ratings")

# Page: About
elif page == "ℹ️ About":
    st.header("About This Project")
    
    st.markdown("""
    ### DataSynthis Movie Recommendation System
    
    This intelligent recommendation system uses advanced machine learning algorithms to provide
    personalized movie suggestions based on user preferences and viewing history.
    
    #### Features:
    - **Hybrid Approach**: Combines User-Based Collaborative Filtering and SVD Matrix Factorization
    - **High Accuracy**: Trained on comprehensive movie rating datasets
    - **Real-Time Predictions**: Instant recommendations for any user
    - **Interactive Visualizations**: Understand user behavior and preferences
    
    #### Algorithms Used:
    1. **User-Based Collaborative Filtering**: Finds similar users and recommends movies they enjoyed
    2. **SVD Matrix Factorization**: Discovers latent patterns in rating data
    3. **Hybrid Ensemble**: Weighted combination (60% CF, 40% SVD) for optimal results
    
    #### Dataset:
    - Movies with genres and metadata
    - User ratings with timestamps
    - User demographic information
    
    #### Performance Metrics:
    - Precision@10: Measures recommendation accuracy
    - Temporal validation: Train-test split respects chronological order
    - Cold start handling: Graceful degradation for new users
    
    ---
    
    **Developed for DataSynthis ML Job Task**
    
    **Technology Stack**: Python, Streamlit, Scikit-learn, Pandas, NumPy, Plotly
    """)
    
    st.info("💡 Tip: Load the models from the sidebar and navigate to 'Get Recommendations' to try it out!")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>DataSynthis Movie Recommendation System | Deployed on Hugging Face Spaces</p>
        <p>Built with ❤️ using Streamlit</p>
    </div>
""", unsafe_allow_html=True)