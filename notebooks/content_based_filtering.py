#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Content-Based Filtering for Movie Recommendation
===============================================
This script implements a content-based filtering approach for movie recommendation
using movie genre features from the MovieLens dataset.
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Define paths
PROCESSED_DIR = '../data/processed'
DATA_DIR = '../data/ml-100k'
MODELS_DIR = '../models'

# Create models directory if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)

# Load the processed data
print("Loading processed data...")
train_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'train.csv'))
test_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'test.csv'))
metadata = pd.read_csv(os.path.join(PROCESSED_DIR, 'metadata.csv')).iloc[0]
movie_content_mapping = pd.read_csv(os.path.join(PROCESSED_DIR, 'movie_content_mapping.csv'))

# Load movie data with genres
movie_columns = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
genres = ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 
          'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
          'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movie_columns.extend(genres)

movies_df = pd.read_csv(os.path.join(DATA_DIR, 'u.item'), 
                        sep='|', 
                        names=movie_columns,
                        encoding='latin-1')

# Load genre features
genre_features = np.load(os.path.join(PROCESSED_DIR, 'movie_genre_features.npy'))

num_users = metadata['num_users']
num_movies = metadata['num_movies']
embedding_dim = int(metadata['embedding_dim'])
min_rating = metadata['min_rating']
max_rating = metadata['max_rating']

print(f"Number of users: {num_users}")
print(f"Number of movies: {num_movies}")
print(f"Number of genres: {len(genres)}")
print(f"Rating range: [{min_rating}, {max_rating}]")

# Process movie metadata
print("Processing movie metadata...")

# Create genre strings for each movie
movies_df['genre_str'] = movies_df[genres].apply(
    lambda x: ' '.join([genres[i] for i, val in enumerate(x) if val == 1]), 
    axis=1
)

# Add year from title
movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)$')
movies_df['title_no_year'] = movies_df['title'].str.replace(r' \(\d{4}\)$', '', regex=True)

# Print some examples
print("\nSample movies with genre strings:")
print(movies_df[['movie_id', 'title', 'genre_str']].head())

# Create TF-IDF vectors for genre strings
tfidf = TfidfVectorizer(min_df=3, max_features=200, stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['genre_str'].fillna(''))

# Calculate cosine similarity between movies
genre_sim = cosine_similarity(tfidf_matrix)

# Save similarity matrix
np.save(os.path.join(PROCESSED_DIR, 'genre_similarity.npy'), genre_sim)

# Function to get movie recommendations based on content similarity
def get_content_recommendations(movie_id, sim_matrix, movies_df, top_n=10):
    # Get movie index
    idx = movies_df[movies_df['movie_id'] == movie_id].index[0]
    
    # Get similarity scores
    sim_scores = list(enumerate(sim_matrix[idx]))
    
    # Sort by similarity
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top N similar movies
    sim_scores = sim_scores[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    
    return movies_df.iloc[movie_indices][['movie_id', 'title', 'genre_str']]

# Test content-based recommendations
print("\nContent-based recommendations example:")
test_movie_id = 1  # Toy Story
print(f"Recommendations for movie: {movies_df[movies_df['movie_id'] == test_movie_id]['title'].values[0]}")
print(get_content_recommendations(test_movie_id, genre_sim, movies_df))

# Define a neural network model for content-based filtering
class ContentBasedNN(nn.Module):
    def __init__(self, num_users, num_items, num_genres, embedding_dim=50):
        super(ContentBasedNN, self).__init__()
        
        # User embedding
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        
        # Genre feature processing
        self.genre_fc = nn.Sequential(
            nn.Linear(num_genres, embedding_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embedding_dim),
            nn.Dropout(0.2)
        )
        
        # Prediction layers
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 1)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        
        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, user_indices, genre_features):
        # User embedding
        user_embedding = self.user_embedding(user_indices)
        
        # Process genre features
        genre_embedding = self.genre_fc(genre_features)
        
        # Concatenate user and genre embeddings
        concat = torch.cat([user_embedding, genre_embedding], dim=1)
        
        # Final prediction
        rating = self.fc_layers(concat)
        
        # Scale to rating range
        rating = torch.sigmoid(rating) * (max_rating - min_rating) + min_rating
        
        return rating.squeeze()

# Create PyTorch Dataset for content-based filtering
class ContentBasedDataset(Dataset):
    def __init__(self, ratings_df, genre_features, movie_mapping):
        self.ratings_df = ratings_df
        self.genre_features = torch.tensor(genre_features, dtype=torch.float)
        self.movie_mapping = movie_mapping
        
        # Prepare data
        self.users = torch.tensor(ratings_df['user_encoded'].values, dtype=torch.long)
        self.movies = torch.tensor(ratings_df['movie_encoded'].values, dtype=torch.long)
        self.ratings = torch.tensor(ratings_df['rating'].values, dtype=torch.float)
        
    def __len__(self):
        return len(self.ratings_df)
    
    def __getitem__(self, idx):
        user = self.users[idx]
        movie = self.movies[idx]
        rating = self.ratings[idx]
        
        # Get genre features for the movie
        genre_feature = self.genre_features[movie]
        
        return {
            'user': user,
            'movie': movie,
            'genre_feature': genre_feature,
            'rating': rating
        }

# Create datasets and dataloaders
train_dataset = ContentBasedDataset(train_df, genre_features, movie_content_mapping)
test_dataset = ContentBasedDataset(test_df, genre_features, movie_content_mapping)

batch_size = 256
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss function, and optimizer
model = ContentBasedNN(num_users, num_movies, len(genres))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Training loop
num_epochs = 10
train_losses = []
test_losses = []
test_rmse = []
test_mae = []

print("\nTraining Content-Based Neural Network model...")
for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    for batch in train_dataloader:
        user_indices = batch['user']
        genre_features = batch['genre_feature']
        ratings = batch['rating']
        
        # Forward pass
        outputs = model(user_indices, genre_features)
        loss = criterion(outputs, ratings)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * len(ratings)
    
    train_loss /= len(train_dataset)
    train_losses.append(train_loss)
    
    # Evaluation
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            user_indices = batch['user']
            genre_features = batch['genre_feature']
            ratings = batch['rating']
            
            # Forward pass
            outputs = model(user_indices, genre_features)
            loss = criterion(outputs, ratings)
            
            test_loss += loss.item() * len(ratings)
            
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(ratings.cpu().numpy())
    
    test_loss /= len(test_dataset)
    test_losses.append(test_loss)
    
    # Calculate metrics
    current_rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    current_mae = mean_absolute_error(all_targets, all_preds)
    test_rmse.append(current_rmse)
    test_mae.append(current_mae)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, RMSE: {current_rmse:.4f}, MAE: {current_mae:.4f}")

# Save the model
torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'content_based_model.pth'))

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training and Test Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(PROCESSED_DIR, 'content_based_loss.png'))

# Plot RMSE and MAE
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), test_rmse, label='RMSE')
plt.plot(range(1, num_epochs+1), test_mae, label='MAE')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Test RMSE and MAE')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(PROCESSED_DIR, 'content_based_metrics.png'))

print(f"Final RMSE: {test_rmse[-1]:.4f}")
print(f"Final MAE: {test_mae[-1]:.4f}")
print("Content-based model training completed and saved!")

# Save metrics for later comparison
metrics_df = pd.DataFrame({
    'model': ['Content-Based'],
    'rmse': [test_rmse[-1]],
    'mae': [test_mae[-1]]
})
metrics_df.to_csv(os.path.join(PROCESSED_DIR, 'content_based_metrics.csv'), index=False)

# Load collaborative filtering metrics for comparison
collab_metrics = pd.read_csv(os.path.join(PROCESSED_DIR, 'collaborative_metrics.csv'))

# Combine all metrics
all_metrics = pd.concat([collab_metrics, metrics_df])
all_metrics.to_csv(os.path.join(PROCESSED_DIR, 'all_models_metrics.csv'), index=False)

# Print comparison of all models
print("\nAll Models Comparison:")
print(all_metrics)

# Create a hybrid recommendation function that combines collaborative and content-based approaches
def hybrid_recommend(user_id, movie_id, cb_model, cf_model, genre_features, alpha=0.5):
    """
    Generate hybrid recommendations by combining content-based and collaborative filtering predictions.
    
    Parameters:
    -----------
    user_id : int
        User ID
    movie_id : int
        Movie ID
    cb_model : ContentBasedNN
        Content-based model
    cf_model : NeuMF or LSTMRecommender
        Collaborative filtering model
    genre_features : numpy.ndarray
        Genre features for movies
    alpha : float
        Weight for collaborative filtering (1-alpha is weight for content-based)
        
    Returns:
    --------
    float
        Predicted rating
    """
    # This is a placeholder for the hybrid recommendation function
    # In a real implementation, we would load both models and combine their predictions
    
    # For demonstration purposes, we'll just return a weighted average of the two models' predictions
    # In practice, you would need to implement this function based on your specific models
    
    return f"Hybrid recommendation for user {user_id} and movie {movie_id} with alpha={alpha}"

print("\nHybrid recommendation example:")
print(hybrid_recommend(1, 1, model, None, genre_features))
