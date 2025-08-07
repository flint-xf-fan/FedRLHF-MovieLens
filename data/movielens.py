print("Importing movielens.py")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader

# MovieLensDataset class
class MovieLensDataset(Dataset):
    def __init__(self, ratings_path='datasets/ml-latest-small/ratings.csv', 
                 movies_path='datasets/ml-latest-small/movies.csv', 
                 test_size=0.2, val_size=0.1):
        self.ratings = pd.read_csv(ratings_path)
        self.movies = pd.read_csv(movies_path)
        self.df = pd.merge(self.ratings, self.movies[['movieId', 'title', 'genres']], on='movieId', how='left')
        
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()
        
        self.df['user_id'] = self.user_encoder.fit_transform(self.df['userId'])
        self.df['movie_id'] = self.movie_encoder.fit_transform(self.df['movieId'])
        
        self.df['high_rating'] = (self.df['rating'] >= 4).astype(int)
        
        self.train, self.test = train_test_split(self.df, test_size=test_size, stratify=self.df['user_id'])
        self.train, self.val = train_test_split(self.train, test_size=val_size/(1-test_size), stratify=self.train['user_id'])
    
    def __len__(self):
        return len(self.train)

    def __getitem__(self, idx):
        row = self.train.iloc[idx]
        X = torch.tensor([row['user_id'], row['movie_id']], dtype=torch.long)
        y = torch.tensor(row['high_rating'], dtype=torch.float32)
        movie_features = torch.tensor(self.get_movie_features().iloc[row['movie_id']].values, dtype=torch.float32)
        return X, y, movie_features

    def get_user_data(self, user_id):
        user_data = self.train[self.train['user_id'] == user_id]
        X = user_data[['user_id', 'movie_id']].values
        y = user_data['high_rating'].values
        return X, y

    def get_num_users(self):
        return self.df['user_id'].nunique()
    
    def get_num_movies(self):
        return self.df['movie_id'].nunique()

    def get_movie_features(self):
        genre_dummies = self.movies['genres'].str.get_dummies(sep='|')
        movie_features = pd.concat([self.movies[['movieId']], genre_dummies], axis=1)
        all_movie_ids = set(self.df['movieId'].unique()) | set(self.movies['movieId'].unique())
        movie_id_map = {id: i for i, id in enumerate(sorted(all_movie_ids))}
        movie_features['movie_id'] = movie_features['movieId'].map(movie_id_map)
        return movie_features.set_index('movie_id').drop('movieId', axis=1)

    def get_dataloader(self, user_id, batch_size):
        X, y = self.get_user_data(user_id)
        X = torch.tensor(X, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        movie_features = self.get_movie_features().iloc[X[:, 1]].values
        movie_features = torch.tensor(movie_features, dtype=torch.float32)
        
        dataset = torch.utils.data.TensorDataset(X, y, movie_features)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)