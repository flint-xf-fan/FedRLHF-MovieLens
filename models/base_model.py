print("Importing base_model.py")

import torch
import torch.nn as nn

class MovieRatingPredictor(nn.Module):
    def __init__(self, num_users, num_movies, num_movie_features, embedding_dim=50, hidden_dims=[100, 50]):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        
        layers = []
        input_dim = embedding_dim * 2 + num_movie_features
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, user_ids, movie_ids, movie_features):
        user_embeds = self.user_embedding(user_ids)
        movie_embeds = self.movie_embedding(movie_ids)
        x = torch.cat([user_embeds, movie_embeds, movie_features], dim=1)
        output = self.layers(x)
        return output.view(-1, 1)