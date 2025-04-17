#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Neural Matrix Factorization (NeuMF) for Collaborative Filtering
===============================================================
This script implements the Neural Matrix Factorization (NeuMF) model
for collaborative filtering on the MovieLens dataset.
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

# Create PyTorch Dataset
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

# Create datasets and dataloaders
train_dataset = MovieLensDataset(train_df)
test_dataset = MovieLensDataset(test_df)

batch_size = 256
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the Neural Matrix Factorization (NeuMF) model
class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, layers=[64, 32, 16, 8]):
        super(NeuMF, self).__init__()
        
        # GMF part
        self.user_gmf_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_gmf_embedding = nn.Embedding(num_items, embedding_dim)
        
        # MLP part
        self.user_mlp_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_mlp_embedding = nn.Embedding(num_items, embedding_dim)
        
        # MLP layers
        self.mlp_layers = nn.ModuleList()
        input_size = 2 * embedding_dim
        
        for i, layer_size in enumerate(layers):
            self.mlp_layers.append(nn.Linear(input_size, layer_size))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.BatchNorm1d(layer_size))
            input_size = layer_size
        
        # Output layer
        self.output_layer = nn.Linear(layers[-1] + embedding_dim, 1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        # Initialize embeddings
        nn.init.normal_(self.user_gmf_embedding.weight, std=0.01)
        nn.init.normal_(self.item_gmf_embedding.weight, std=0.01)
        nn.init.normal_(self.user_mlp_embedding.weight, std=0.01)
        nn.init.normal_(self.item_mlp_embedding.weight, std=0.01)
        
        # Initialize MLP layers
        for layer in self.mlp_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                
        # Initialize output layer
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
        
    def forward(self, user_indices, item_indices):
        # GMF part
        user_gmf_embedding = self.user_gmf_embedding(user_indices)
        item_gmf_embedding = self.item_gmf_embedding(item_indices)
        gmf_vector = user_gmf_embedding * item_gmf_embedding
        
        # MLP part
        user_mlp_embedding = self.user_mlp_embedding(user_indices)
        item_mlp_embedding = self.item_mlp_embedding(item_indices)
        mlp_vector = torch.cat([user_mlp_embedding, item_mlp_embedding], dim=1)
        
        for layer in self.mlp_layers:
            mlp_vector = layer(mlp_vector)
        
        # Concatenate GMF and MLP parts
        vector = torch.cat([gmf_vector, mlp_vector], dim=1)
        
        # Output layer
        rating = self.output_layer(vector)
        
        # Scale to rating range
        rating = torch.sigmoid(rating) * (max_rating - min_rating) + min_rating
        
        return rating.squeeze()

# Initialize model, loss function, and optimizer
model = NeuMF(num_users, num_movies, embedding_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)

# Training loop
num_epochs = 10
train_losses = []
test_losses = []
test_rmse = []
test_mae = []

print("Training NeuMF model...")
for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    for batch in train_dataloader:
        user_indices = batch['user']
        movie_indices = batch['movie']
        ratings = batch['rating']
        
        # Forward pass
        outputs = model(user_indices, movie_indices)
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
            movie_indices = batch['movie']
            ratings = batch['rating']
            
            # Forward pass
            outputs = model(user_indices, movie_indices)
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
torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'neumf_model.pth'))

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training and Test Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(PROCESSED_DIR, 'neumf_loss.png'))

# Plot RMSE and MAE
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), test_rmse, label='RMSE')
plt.plot(range(1, num_epochs+1), test_mae, label='MAE')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Test RMSE and MAE')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(PROCESSED_DIR, 'neumf_metrics.png'))

print(f"Final RMSE: {test_rmse[-1]:.4f}")
print(f"Final MAE: {test_mae[-1]:.4f}")
print("NeuMF model training completed and saved!")

# Save metrics for later comparison
metrics_df = pd.DataFrame({
    'model': ['NeuMF'],
    'rmse': [test_rmse[-1]],
    'mae': [test_mae[-1]]
})
metrics_df.to_csv(os.path.join(PROCESSED_DIR, 'neumf_metrics.csv'), index=False)
