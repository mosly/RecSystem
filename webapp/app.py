import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.express as px
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Create a patched version of Streamlit's path watching to avoid PyTorch errors
# This is needed because PyTorch uses custom class attributes that conflict with Streamlit's file watcher
import streamlit.watcher.path_watcher
original_watch_file = streamlit.watcher.path_watcher.watch_file

def patched_watch_file(path, callback):
    # Skip watching PyTorch-related files
    if 'torch' in path:
        return
    return original_watch_file(path, callback)

# Apply the patch
streamlit.watcher.path_watcher.watch_file = patched_watch_file

# Now it's safe to import torch
import torch

# Set paths - use flexible path finding for both local and Streamlit Cloud deployment
def get_project_root():
    """Get the path to the project root folder"""
    # When running locally
    if os.path.exists(os.path.join(os.path.dirname(__file__), '..')):
        return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # When running on Streamlit Cloud (try to find data in the same directory as the app)
    if os.path.exists(os.path.join(os.path.dirname(__file__), 'data')):
        return os.path.abspath(os.path.dirname(__file__))
    
    # Last resort: current working directory
    return os.getcwd()

PROJECT_ROOT = get_project_root()
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
ML_100K_DIR = os.path.join(PROJECT_ROOT, 'data', 'ml-100k')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
EVAL_DIR = os.path.join(PROJECT_ROOT, 'evaluation')

# Function to load and process the original MovieLens data to get genre information
@st.cache_data
def load_movie_genres():
    """Load movie genres from the original MovieLens dataset"""
    try:
        # Check if the original data file exists
        if os.path.exists(os.path.join(ML_100K_DIR, 'u.item')):
            # Define column names for the u.item file
            cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL']
            # Add genre columns
            genre_cols = ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 
                          'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 
                          'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 
                          'Sci-Fi', 'Thriller', 'War', 'Western']
            cols.extend(genre_cols)
            
            # Read the data with '|' separator
            movies_data = pd.read_csv(os.path.join(ML_100K_DIR, 'u.item'), 
                                     sep='|', 
                                     names=cols, 
                                     encoding='latin-1')
            
            # Create a genre_str column by joining all the genres that apply to each movie
            def create_genre_string(row):
                genres = []
                for genre in genre_cols:
                    if row[genre] == 1:
                        genres.append(genre)
                return ', '.join(genres) if genres else 'Unknown'
            
            movies_data['genre_str'] = movies_data.apply(create_genre_string, axis=1)
            
            # Return just the movie_id and genre_str columns
            return movies_data[['movie_id', 'genre_str']]
        
        st.warning("Original movie data file not found. Genre information will be unavailable.")
        return pd.DataFrame(columns=['movie_id', 'genre_str'])
    
    except Exception as e:
        st.error(f"Error loading genre data: {e}")
        return pd.DataFrame(columns=['movie_id', 'genre_str'])

# Create a function to load datasets
@st.cache_data
def load_data():
    """Load and cache the datasets"""
    try:
        # First, check if we need to upload data when in Streamlit Cloud
        if not os.path.exists(os.path.join(PROCESSED_DIR, 'movie_content_mapping.csv')):
            st.warning("Data files not found in the expected path. Please use the file uploader below.")
            
            # Add file uploaders for required files
            with st.expander("Upload Data Files", expanded=True):
                movies_file = st.file_uploader("Upload movie_content_mapping.csv", type="csv")
                users_file = st.file_uploader("Upload user_mapping.csv", type="csv")
                train_file = st.file_uploader("Upload train.csv", type="csv")
                test_file = st.file_uploader("Upload test.csv", type="csv")
                
                if not all([movies_file, users_file, train_file, test_file]):
                    st.stop()
                
                # Read uploaded files
                movies_df = pd.read_csv(movies_file)
                users_df = pd.read_csv(users_file)
                train_df = pd.read_csv(train_file)
                test_df = pd.read_csv(test_file)
        else:
            # Load processed data files using the correct path
            movies_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'movie_content_mapping.csv'))
            users_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'user_mapping.csv'))
            train_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'train.csv'))
            test_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'test.csv'))
        
        # Always try to load genre information from original dataset
        genres_df = load_movie_genres()
        
        # Always merge genres (whether genre_str exists or not)
        if not genres_df.empty:
            if 'genre_str' in movies_df.columns:
                # Replace existing genre_str column to ensure it has the most updated data
                movies_df = movies_df.drop(columns=['genre_str'])
                
            # Merge the genres dataframe
            movies_df = pd.merge(movies_df, genres_df, on='movie_id', how='left')
            # Fill missing genres
            movies_df['genre_str'] = movies_df['genre_str'].fillna('Unknown')
        elif 'genre_str' not in movies_df.columns:
            # Create an empty genre column if we couldn't load genre data
            movies_df['genre_str'] = 'Unknown'
            
        # Load metrics if available, or create placeholders
        try:
            # Try loading metrics files from expected locations
            if os.path.exists(os.path.join(PROCESSED_DIR, 'all_models_metrics.csv')):
                metrics_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'all_models_metrics.csv'))
            else:
                metrics_df = pd.DataFrame(columns=['model', 'rmse', 'mae'])
                
            if os.path.exists(os.path.join(EVAL_DIR, 'ranking_metrics.csv')):
                ranking_df = pd.read_csv(os.path.join(EVAL_DIR, 'ranking_metrics.csv'))
            else:
                ranking_df = pd.DataFrame(columns=['model', 'k', 'precision', 'recall', 'ndcg'])
        except:
            # Default empty dataframes if files aren't found
            metrics_df = pd.DataFrame(columns=['model', 'rmse', 'mae'])
            ranking_df = pd.DataFrame(columns=['model', 'k', 'precision', 'recall', 'ndcg'])
        
        return {
            'movies': movies_df,
            'users': users_df,
            'train': train_df,
            'test': test_df,
            'metrics': metrics_df,
            'ranking': ranking_df
        }
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Function to make recommendations
def get_recommendations(user_id, model_type, num_recommendations=10):
    """
    Generate movie recommendations for a user
    This is a simulated function as the actual model loading and inference would be more complex
    """
    try:
        # In a real implementation, you would load the model and generate actual recommendations
        data = load_data()
        
        if data is None:
            return pd.DataFrame()
            
        all_movies = data['movies']
        
        if model_type == 'popular':
            # Popular items recommendations (based on training data)
            movie_counts = data['train']['movie_encoded'].value_counts().reset_index()
            movie_counts.columns = ['movie_encoded', 'count']
            popular_movies = movie_counts.sort_values('count', ascending=False)['movie_encoded'].values[:num_recommendations]
            
            # Get movie info for the popular movies
            recommended_movies = all_movies[all_movies['movie_encoded'].isin(popular_movies)]
            # Ensure we only return columns that exist
            result_cols = ['movie_id', 'title']
            if 'genre_str' in all_movies.columns:
                result_cols.append('genre_str')
            return recommended_movies[result_cols].head(num_recommendations)
            
        elif model_type in ['neumf', 'lstm', 'content']:
            # Simulate personalized recommendations
            # In a real implementation, this would use the actual models
            
            # For simulated results, we'll select a random subset of movies but ensure they're different for different users
            np.random.seed(user_id)  # Use user_id as seed for deterministic results per user
            movie_indices = np.random.choice(len(all_movies), num_recommendations, replace=False)
            
            recommended_movies = all_movies.iloc[movie_indices]
            # Ensure we only return columns that exist
            result_cols = ['movie_id', 'title']
            if 'genre_str' in all_movies.columns:
                result_cols.append('genre_str')
            result = recommended_movies[result_cols].copy()
            
            # Add simulated ratings
            result['predicted_rating'] = np.round(np.random.uniform(3.5, 5.0, size=len(result)), 1)
            return result.sort_values('predicted_rating', ascending=False)
            
        else:
            return pd.DataFrame()
    
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
        return pd.DataFrame()

# Function to load and display model evaluation metrics
def show_model_evaluation():
    """Display model evaluation metrics and visualizations"""
    try:
        data = load_data()
        if data is None:
            return
            
        # Display error metrics
        st.subheader("Error Metrics (RMSE and MAE)")
        metrics_df = data['metrics']
        st.dataframe(metrics_df)
        
        # Show evaluation figures
        st.subheader("Evaluation Visualizations")
        col1, col2 = st.columns(2)
        
        try:
            with col1:
                rmse_mae_fig = Image.open(os.path.join(EVAL_DIR, 'rmse_mae_comparison.png'))
                st.image(rmse_mae_fig, caption="RMSE & MAE Comparison")
            
            with col2:
                heatmap_fig = Image.open(os.path.join(EVAL_DIR, 'all_metrics_heatmap.png'))
                st.image(heatmap_fig, caption="All Metrics Heatmap")
        except Exception as e:
            st.warning(f"Could not load evaluation images: {e}")
            
        # Display ranking metrics
        st.subheader("Ranking Metrics")
        ranking_df = data['ranking']
        
        # Create a pivot table for better visualization
        pivot_df = ranking_df.pivot_table(
            index='model', 
            columns='k', 
            values=['precision', 'recall', 'ndcg']
        ).reset_index()
        
        st.write("Precision, Recall, and NDCG at different k values")
        st.dataframe(pivot_df)
        
        # Show plots for ranking metrics
        try:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                precision_fig = Image.open(os.path.join(EVAL_DIR, 'precision_comparison.png'))
                st.image(precision_fig, caption="Precision Comparison")
                
            with col2:
                recall_fig = Image.open(os.path.join(EVAL_DIR, 'recall_comparison.png'))
                st.image(recall_fig, caption="Recall Comparison")
                
            with col3:
                ndcg_fig = Image.open(os.path.join(EVAL_DIR, 'ndcg_comparison.png'))
                st.image(ndcg_fig, caption="NDCG Comparison")
        except Exception as e:
            st.warning(f"Could not load ranking metrics images: {e}")
    
    except Exception as e:
        st.error(f"Error showing evaluation: {e}")

# Function to show data exploration visualizations
def show_data_exploration():
    """Display data exploration visualizations"""
    try:
        data = load_data()
        if data is None:
            return
            
        st.subheader("Dataset Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Movies Dataset Sample")
            st.dataframe(data['movies'].head())
        
        with col2:
            st.write("Users Dataset Sample")
            st.dataframe(data['users'].head())
            
        # Show rating distribution
        st.subheader("Rating Distribution")
        
        try:
            rating_dist_fig = Image.open(os.path.join(PROCESSED_DIR, 'rating_distribution.png'))
            st.image(rating_dist_fig, caption="Rating Distribution")
        except Exception as e:
            # If image doesn't exist, create a simple visualization
            train_ratings = data['train']['rating']
            
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(train_ratings, bins=5, kde=True, ax=ax)
            ax.set_title('Rating Distribution')
            ax.set_xlabel('Rating')
            ax.set_ylabel('Count')
            st.pyplot(fig)
    
    except Exception as e:
        st.error(f"Error in data exploration: {e}")

# Main function
def main():
    st.set_page_config(
        page_title="Movie Recommendation System",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🎬 Movie Recommendation System")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Get Recommendations", "Model Evaluation", "Data Exploration"])
    
    # Home page
    if page == "Home":
        st.markdown("""
        ## Welcome to the Movie Recommendation System!
        
        This interactive application demonstrates different recommendation approaches using the MovieLens 100K dataset. The system implements several deep learning-based recommendation models:
        
        1. **Neural Matrix Factorization (NeuMF)**: Combines matrix factorization with neural networks
        2. **LSTM-based Sequential Recommender**: Captures temporal patterns in user preferences
        3. **Content-Based Filtering**: Utilizes movie genre information for recommendations
        
        ### Dataset
        
        The MovieLens 100K dataset contains:
        - 100,000 ratings (1-5) from 943 users on 1,682 movies
        - Each user has rated at least 20 movies
        - Simple demographic information for users (age, gender, occupation, zip code)
        - Movie information including title, release date, and genre
        
        ### Features
        
        - **Get Recommendations**: Generate personalized movie recommendations using different models
        - **Model Evaluation**: Compare the performance of different recommendation approaches
        - **Data Exploration**: Explore the dataset and understand its characteristics
        
        Use the sidebar to navigate between different sections of the application.
        """)
    
    # Recommendations page
    elif page == "Get Recommendations":
        st.header("Get Movie Recommendations")
        
        data = load_data()
        if data is None:
            st.error("Could not load data. Please check the data files.")
            return
            
        # Select user
        max_user_id = data['users']['user_id'].max()
        user_id = st.number_input("Select User ID", min_value=1, max_value=max_user_id, value=1)
        
        # Select model
        model_type = st.selectbox(
            "Select Recommendation Model",
            ["popular", "neumf", "lstm", "content"],
            format_func=lambda x: {
                "popular": "Popularity-Based",
                "neumf": "Neural Matrix Factorization",
                "lstm": "LSTM Sequential Recommender",
                "content": "Content-Based Filtering"
            }[x]
        )
        
        # Number of recommendations
        num_recs = st.slider("Number of Recommendations", min_value=5, max_value=20, value=10)
        
        # Generate recommendations
        if st.button("Get Recommendations"):
            with st.spinner("Generating recommendations..."):
                recommendations = get_recommendations(user_id, model_type, num_recs)
                
                if not recommendations.empty:
                    st.success(f"Generated {len(recommendations)} recommendations for User {user_id} using {model_type} model")
                    
                    # Display recommendations
                    if 'predicted_rating' in recommendations.columns:
                        recommendations = recommendations.sort_values('predicted_rating', ascending=False)
                        st.dataframe(recommendations)
                        
                        # Create a bar chart of recommendations
                        # Make sure we only use columns that exist
                        hover_data = []
                        if 'genre_str' in recommendations.columns:
                            hover_data = ['genre_str']
                            
                        fig = px.bar(
                            recommendations,
                            x='title',
                            y='predicted_rating',
                            hover_data=hover_data,
                            title=f'Top {num_recs} Movie Recommendations for User {user_id}',
                            labels={'title': 'Movie', 'predicted_rating': 'Predicted Rating'},
                            color='predicted_rating',
                            color_continuous_scale='reds'
                        )
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig)
                    else:
                        st.dataframe(recommendations)
                else:
                    st.error("Could not generate recommendations.")
    
    # Model Evaluation page
    elif page == "Model Evaluation":
        st.header("Model Evaluation")
        show_model_evaluation()
        
        # Display evaluation summary
        try:
            with open(os.path.join(EVAL_DIR, 'evaluation_summary.md'), 'r') as file:
                summary_content = file.read()
                st.markdown(summary_content)
        except Exception as e:
            st.warning(f"Could not load evaluation summary: {e}")
    
    # Data Exploration page
    elif page == "Data Exploration":
        st.header("Data Exploration")
        show_data_exploration()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("Developed by Mostafa Mosly, Zienab Ahmed, Mohamed Abdelgwad, and Nourhan Moawd")

if __name__ == "__main__":
    main()