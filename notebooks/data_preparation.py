#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data Preparation for Deep Learning-Based Recommender System
==========================================================
This script loads and preprocesses the MovieLens 100K dataset for use in 
both collaborative filtering and content-based filtering approaches.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Define paths
DATA_DIR = '../data/ml-100k'
PROCESSED_DIR = '../data/processed'

# Create processed directory if it doesn't exist
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Load the data
print("Loading MovieLens 100K dataset...")

# Ratings data
# Format: UserID::MovieID::Rating::Timestamp
column_names = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings_df = pd.read_csv(os.path.join(DATA_DIR, 'u.data'), 
                         sep='\t', 
                         names=column_names,
                         encoding='latin-1')

# Movie data
# Format: MovieID::Title::Release Date::Video Release Date::IMDb URL::Genre1|Genre2|...
movie_columns = ['movie_id', 'title', 'release_date', 'video_release_date', 
                'imdb_url']
# Add genre columns
genres = ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 
          'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
          'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movie_columns.extend(genres)

movies_df = pd.read_csv(os.path.join(DATA_DIR, 'u.item'), 
                        sep='|', 
                        names=movie_columns,
                        encoding='latin-1')

# User data
# Format: UserID::Gender::Age::Occupation::Zip Code
user_columns = ['user_id', 'gender', 'age', 'occupation', 'zip_code']
users_df = pd.read_csv(os.path.join(DATA_DIR, 'u.user'), 
                       sep='|', 
                       names=user_columns,
                       encoding='latin-1')

print(f"Loaded {len(ratings_df)} ratings from {ratings_df['user_id'].nunique()} users on {ratings_df['movie_id'].nunique()} movies")
print(f"Loaded {len(movies_df)} movies with {len(genres)} genres")
print(f"Loaded {len(users_df)} users")

# Basic data exploration
print("\nRatings distribution:")
print(ratings_df['rating'].value_counts().sort_index())

print("\nRatings statistics:")
print(ratings_df['rating'].describe())

# Plot ratings distribution
plt.figure(figsize=(10, 6))
ratings_df['rating'].value_counts().sort_index().plot(kind='bar')
plt.title('Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.savefig(os.path.join(PROCESSED_DIR, 'rating_distribution.png'))

# Preprocess data for collaborative filtering
print("\nPreprocessing data for collaborative filtering...")

# Encode user and movie IDs
user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()

ratings_df['user_encoded'] = user_encoder.fit_transform(ratings_df['user_id'])
ratings_df['movie_encoded'] = movie_encoder.fit_transform(ratings_df['movie_id'])

# Save encoders mapping for later use
user_mapping = pd.DataFrame({
    'user_id': ratings_df['user_id'].unique(),
    'user_encoded': user_encoder.transform(ratings_df['user_id'].unique())
})
user_mapping.to_csv(os.path.join(PROCESSED_DIR, 'user_mapping.csv'), index=False)

movie_mapping = pd.DataFrame({
    'movie_id': ratings_df['movie_id'].unique(),
    'movie_encoded': movie_encoder.transform(ratings_df['movie_id'].unique())
})
movie_mapping.to_csv(os.path.join(PROCESSED_DIR, 'movie_mapping.csv'), index=False)

# Split data into train and test sets (80% train, 20% test)
train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)

print(f"Training set: {len(train_df)} samples")
print(f"Test set: {len(test_df)} samples")

# Save processed data
train_df.to_csv(os.path.join(PROCESSED_DIR, 'train.csv'), index=False)
test_df.to_csv(os.path.join(PROCESSED_DIR, 'test.csv'), index=False)

# Preprocess data for content-based filtering
print("\nPreprocessing data for content-based filtering...")

# Extract genre features
genre_features = movies_df[genres].values

# Normalize genre features
genre_features = genre_features.astype(np.float32)

# Save genre features
np.save(os.path.join(PROCESSED_DIR, 'movie_genre_features.npy'), genre_features)

# Create a mapping from movie_encoded to genre_features
movie_content_mapping = pd.DataFrame({
    'movie_id': movies_df['movie_id'],
    'title': movies_df['title']
})
movie_content_mapping = movie_content_mapping.merge(movie_mapping, on='movie_id')
movie_content_mapping.to_csv(os.path.join(PROCESSED_DIR, 'movie_content_mapping.csv'), index=False)

# Create PyTorch Dataset classes for collaborative filtering
class MovieLensDataset(Dataset):
    def __init__(self, ratings_df):
        self.users = torch.tensor(ratings_df['user_encoded'].values, dtype=torch.long)
        self.movies = torch.tensor(ratings_df['movie_encoded'].values, dtype=torch.long)
        self.ratings = torch.tensor(ratings_df['rating'].values, dtype=torch.float)
        
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        return {
            'user': self.users[idx],
            'movie': self.movies[idx],
            'rating': self.ratings[idx]
        }

# Create datasets
train_dataset = MovieLensDataset(train_df)
test_dataset = MovieLensDataset(test_df)

# Save some metadata for model building
metadata = {
    'num_users': len(user_encoder.classes_),
    'num_movies': len(movie_encoder.classes_),
    'num_genres': len(genres),
    'embedding_dim': 50,  # We'll use 50-dimensional embeddings
    'min_rating': ratings_df['rating'].min(),
    'max_rating': ratings_df['rating'].max()
}

# Save metadata as a DataFrame for easy loading
metadata_df = pd.DataFrame([metadata])
metadata_df.to_csv(os.path.join(PROCESSED_DIR, 'metadata.csv'), index=False)

print("\nData preparation completed successfully!")
print(f"Processed data saved to {PROCESSED_DIR}")
print(f"Number of users: {metadata['num_users']}")
print(f"Number of movies: {metadata['num_movies']}")
print(f"Number of genres: {metadata['num_genres']}")
