#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RNN/LSTM for Sequential Recommendation
======================================
This script implements an RNN/LSTM model for sequential recommendation
on the MovieLens dataset, treating user ratings as a sequence.
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
from collections import defaultdict

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Define paths
PROCESSED_DIR = '../data/processed'
MODELS_DIR = '../models'

# Create models directory if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)

# Load the processed data
print("Loading processed data...")
train_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'train.csv'))
test_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'test.csv'))
metadata = pd.read_csv(os.path.join(PROCESSED_DIR, 'metadata.csv')).iloc[0]

num_users = metadata['num_users']
num_movies = metadata['num_movies']
embedding_dim = int(metadata['embedding_dim'])
min_rating = metadata['min_rating']
max_rating = metadata['max_rating']

print(f"Number of users: {num_users}")
print(f"Number of movies: {num_movies}")
print(f"Embedding dimension: {embedding_dim}")
print(f"Rating range: [{min_rating}, {max_rating}]")

# Prepare sequential data
# Group ratings by user and sort by timestamp
print("Preparing sequential data...")
train_df = train_df.sort_values(['user_encoded', 'timestamp'])
test_df = test_df.sort_values(['user_encoded', 'timestamp'])

# Create sequences for each user
def create_sequences(df, seq_length=5):
    """Create sequences of movie ratings for each user."""
    sequences = []
    targets = []
    user_groups = df.groupby('user_encoded')
    
    for user_id, group in user_groups:
        if len(group) < seq_length + 1:
            continue
            
        # Sort by timestamp
        group = group.sort_values('timestamp')
        
        # Create sequences
        for i in range(len(group) - seq_length):
            seq_movies = group['movie_encoded'].iloc[i:i+seq_length].values
            seq_ratings = group['rating'].iloc[i:i+seq_length].values
            target_movie = group['movie_encoded'].iloc[i+seq_length]
            target_rating = group['rating'].iloc[i+seq_length]
            
            sequences.append({
                'user_id': user_id,
                'seq_movies': seq_movies,
                'seq_ratings': seq_ratings,
                'target_movie': target_movie,
                'target_rating': target_rating
            })
    
    return sequences

# Create sequences
seq_length = 5
train_sequences = create_sequences(train_df, seq_length)
test_sequences = create_sequences(test_df, seq_length)

print(f"Created {len(train_sequences)} training sequences")
print(f"Created {len(test_sequences)} test sequences")

# Create PyTorch Dataset for sequences
class SequentialMovieLensDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        return {
            'user_id': torch.tensor(sequence['user_id'], dtype=torch.long),
            'seq_movies': torch.tensor(sequence['seq_movies'], dtype=torch.long),
            'seq_ratings': torch.tensor(sequence['seq_ratings'], dtype=torch.float),
            'target_movie': torch.tensor(sequence['target_movie'], dtype=torch.long),
            'target_rating': torch.tensor(sequence['target_rating'], dtype=torch.float)
        }

# Create datasets and dataloaders
train_dataset = SequentialMovieLensDataset(train_sequences)
test_dataset = SequentialMovieLensDataset(test_sequences)

batch_size = 64
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the LSTM model for sequential recommendation
class LSTMRecommender(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super(LSTMRecommender, self).__init__()
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim * 2,  # Item embedding + rating
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim + embedding_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
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
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
        # Initialize LSTM
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                
    def forward(self, user_ids, seq_movies, seq_ratings, target_movies):
        batch_size = user_ids.size(0)
        
        # Embed users
        user_embeds = self.user_embedding(user_ids)  # (batch_size, embedding_dim)
        
        # Embed sequence items
        seq_item_embeds = self.item_embedding(seq_movies)  # (batch_size, seq_len, embedding_dim)
        
        # Expand ratings to match embedding dimension
        seq_ratings_expanded = seq_ratings.unsqueeze(-1).expand(-1, -1, embedding_dim)
        
        # Combine item embeddings with ratings
        seq_features = torch.cat([seq_item_embeds, seq_ratings_expanded], dim=-1)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(seq_features)
        
        # Get the last output from LSTM
        lstm_last = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # Embed target movies
        target_movie_embeds = self.item_embedding(target_movies)  # (batch_size, embedding_dim)
        
        # Concatenate LSTM output with target movie embedding
        combined = torch.cat([lstm_last, target_movie_embeds], dim=1)
        
        # Pass through fully connected layers
        output = self.fc_layers(combined)
        
        # Scale to rating range
        output = torch.sigmoid(output) * (max_rating - min_rating) + min_rating
        
        return output.squeeze()

# Initialize model, loss function, and optimizer
model = LSTMRecommender(num_users, num_movies, embedding_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Training loop
num_epochs = 10
train_losses = []
test_losses = []
test_rmse = []
test_mae = []

print("Training LSTM Recommender model...")
for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    for batch in train_dataloader:
        user_ids = batch['user_id']
        seq_movies = batch['seq_movies']
        seq_ratings = batch['seq_ratings']
        target_movies = batch['target_movie']
        target_ratings = batch['target_rating']
        
        # Forward pass
        outputs = model(user_ids, seq_movies, seq_ratings, target_movies)
        loss = criterion(outputs, target_ratings)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * len(target_ratings)
    
    train_loss /= len(train_dataset)
    train_losses.append(train_loss)
    
    # Evaluation
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            user_ids = batch['user_id']
            seq_movies = batch['seq_movies']
            seq_ratings = batch['seq_ratings']
            target_movies = batch['target_movie']
            target_ratings = batch['target_rating']
            
            # Forward pass
            outputs = model(user_ids, seq_movies, seq_ratings, target_movies)
            loss = criterion(outputs, target_ratings)
            
            test_loss += loss.item() * len(target_ratings)
            
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(target_ratings.cpu().numpy())
    
    test_loss /= len(test_dataset)
    test_losses.append(test_loss)
    
    # Calculate metrics
    current_rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    current_mae = mean_absolute_error(all_targets, all_preds)
    test_rmse.append(current_rmse)
    test_mae.append(current_mae)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, RMSE: {current_rmse:.4f}, MAE: {current_mae:.4f}")

# Save the model
torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'lstm_recommender_model.pth'))

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training and Test Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(PROCESSED_DIR, 'lstm_loss.png'))

# Plot RMSE and MAE
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), test_rmse, label='RMSE')
plt.plot(range(1, num_epochs+1), test_mae, label='MAE')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Test RMSE and MAE')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(PROCESSED_DIR, 'lstm_metrics.png'))

print(f"Final RMSE: {test_rmse[-1]:.4f}")
print(f"Final MAE: {test_mae[-1]:.4f}")
print("LSTM Recommender model training completed and saved!")

# Save metrics for later comparison
metrics_df = pd.DataFrame({
    'model': ['LSTM'],
    'rmse': [test_rmse[-1]],
    'mae': [test_mae[-1]]
})
metrics_df.to_csv(os.path.join(PROCESSED_DIR, 'lstm_metrics.csv'), index=False)

# Load NeuMF metrics for comparison
neumf_metrics = pd.read_csv(os.path.join(PROCESSED_DIR, 'neumf_metrics.csv'))

# Combine metrics
all_metrics = pd.concat([neumf_metrics, metrics_df])
all_metrics.to_csv(os.path.join(PROCESSED_DIR, 'collaborative_metrics.csv'), index=False)

# Print comparison
print("\nModel Comparison:")
print(all_metrics)
