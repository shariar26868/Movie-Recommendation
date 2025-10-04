# 🎬 DataSynthis Movie Recommendation System

A hybrid movie recommendation system combining Collaborative Filtering and SVD Matrix Factorization. Built with Streamlit for an interactive user experience.

## Features

- **Hybrid Algorithm**: 60% Collaborative Filtering + 40% SVD Matrix Factorization
- **Personalized Recommendations**: Tailored suggestions for each user
- **Interactive Dashboard**: Real-time visualizations and analytics
- **Dataset Insights**: Rating distribution, genre analysis, user behavior

## Quick Start

1. **Clone and Install**
   ```bash
   git clone https://github.com/yourusername/movie-recommender.git
   cd movie-recommender
   pip install -r requirements.txt
   ```

2. **Prepare Datasets**
   
   Place these CSV files in the project root:
   - `movies.csv` - columns: movieid, title, genres
   - `ratings.csv` - columns: userid, movieid, rating, timestamp
   - `users.csv` - columns: userid, [other user info]

3. **Run the App**
   ```bash
   streamlit run app.py
   ```

4. **Use the System**
   - Click "Load/Reload Models" in the sidebar
   - Navigate to "Get Recommendations"
   - Enter a User ID and generate recommendations

## Technologies

- **Python 3.8+**
- **Streamlit** - Web interface
- **Scikit-learn** - ML algorithms
- **Pandas & NumPy** - Data processing
- **SciPy** - SVD implementation
- **Plotly** - Interactive visualizations

## Algorithm Details

**User-Based Collaborative Filtering**
- Computes user similarity using cosine similarity
- Recommends movies liked by similar users

**SVD Matrix Factorization**
- Decomposes the user-item matrix into latent factors
- Predicts ratings for unwatched movies

**Hybrid Ensemble**
- Combines both approaches with weighted scoring
- Configurable alpha parameter (default: 0.6)

## Project Structure

```
movie-recommender/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── movies.csv         # Movie dataset
├── ratings.csv        # Ratings dataset
├── users.csv          # User dataset
└── README.md          # This file
```

## Requirements

```txt
streamlit
pandas
numpy
scipy
scikit-learn
plotly
```

## License

MIT License - Feel free to use this project for learning and development purposes.

