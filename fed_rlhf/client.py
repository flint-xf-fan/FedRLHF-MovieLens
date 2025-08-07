print("Importing client.py")

import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl
import numpy as np
from torch.utils.data import DataLoader

from models.base_model import MovieRatingPredictor
from models.reward_model import RewardModel

from scipy.stats import spearmanr  


class SimulatedHumanFeedback:
    def __init__(self, true_preferences, noise_level=0.1):
        self.true_preferences = true_preferences
        self.noise_level = noise_level

    def provide_feedback(self, user_id, movie_id, predicted_rating):
        true_rating = self.true_preferences.get((user_id, movie_id), 0)
        noisy_rating = true_rating + np.random.normal(0, self.noise_level)
        
        if predicted_rating > noisy_rating + 0.5:
            return -1  # Too high
        elif predicted_rating < noisy_rating - 0.5:
            return 1   # Too low
        else:
            return 0   # About right

    def get_comparative_feedback(self, user_id, movie_id1, movie_id2, predicted_preference):
        rating1 = self.true_preferences.get((user_id, movie_id1), 0)
        rating2 = self.true_preferences.get((user_id, movie_id2), 0)
        
        true_preference = 1 if rating1 > rating2 else -1 if rating1 < rating2 else 0
        
        if true_preference != predicted_preference:
            return true_preference
        else:
            return 0  # Correct prediction

class MovieLensClient(fl.client.NumPyClient):
    def __init__(self, user_id, dataset, device, batch_size=32):
        self.user_id = user_id
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        
        num_users = dataset.get_num_users()
        num_movies = dataset.get_num_movies()
        movie_features = dataset.get_movie_features()
        self.movie_features = torch.tensor(movie_features.values, dtype=torch.float32).to(device)
        
        self.model = MovieRatingPredictor(num_users, num_movies, movie_features.shape[1]).to(device)
        self.reward_model = RewardModel(input_dim=2).to(device)  # Assuming the model output is 2-dimensional
        
        self.optimizer = optim.Adam(self.model.parameters())
        self.reward_optimizer = optim.Adam(self.reward_model.parameters())
        
        self.criterion = nn.BCELoss()
        
        self.gamma = 0.99
        self.epsilon = 0.1
        self.q_table = {}
        
        true_preferences = {(row['userId'], row['movieId']): row['rating'] 
                            for _, row in dataset.df.iterrows()}
        self.human_feedback = SimulatedHumanFeedback(true_preferences)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        print(f"Starting training for client {self.user_id}")
        
        dataloader = self.dataset.get_dataloader(self.user_id, self.batch_size)

        for _ in range(5):  # Local epochs
            for batch_X, batch_y, batch_features in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_features = batch_features.to(self.device)

                predictions = self.model(batch_X[:, 0], batch_X[:, 1], batch_features).view(-1, 1)

                # Get simulated human feedback
                direct_feedback = [
                    (torch.tensor([[self.user_id, movie_id.item()]], dtype=torch.float32).to(self.device),
                    self.human_feedback.provide_feedback(self.user_id, movie_id.item(), pred.item()))
                    for movie_id, pred in zip(batch_X[:, 1], predictions)
                ]

                # Get comparative feedback
                movie_pairs = [(batch_X[i, 1].item(), batch_X[j, 1].item()) for i in range(len(batch_X)) for j in range(i + 1, len(batch_X))]
                comparative_feedback = self.get_comparative_feedback(self.user_id, movie_pairs)

                # Update reward model
                self.update_reward_model(direct_feedback, comparative_feedback)

                # Update policy model
                self.update_policy_model(batch_X, batch_y, batch_features, [fb for _, fb in direct_feedback])

        return self.get_parameters(config={}), len(dataloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        
        dataloader = self.dataset.get_dataloader(self.user_id, self.batch_size)
        
        total_loss = 0
        correct = 0
        total = 0
        all_y = []
        all_predictions = []

        with torch.no_grad():
            for batch_X, batch_y, batch_features in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_features = batch_features.to(self.device)

                predictions = self.model(batch_X[:, 0], batch_X[:, 1], batch_features).view(-1, 1)
                loss = self.criterion(predictions, batch_y)
                total_loss += loss.item() * len(batch_y)
                correct += ((predictions > 0.5) == batch_y).float().sum().item()
                total += len(batch_y)

                all_y.extend(batch_y.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())

        accuracy = correct / total
        avg_loss = total_loss / total

        ### NEW SPEARMAN CORRELATION CALCULATION ###
        spearman_corr, _ = spearmanr(np.array(all_y), np.array(all_predictions))

        return float(avg_loss), total, {"accuracy": float(accuracy), "spearman_correlation": float(spearman_corr)}

    def update_reward_model(self, direct_feedback, comparative_feedback):
        # Update based on direct feedback
        for state, reward in direct_feedback:
            # Ensure state has exactly 2 features (matching input_dim of RewardModel)
            reward_pred = self.reward_model(state)  # State should already be the correct size
            loss = nn.MSELoss()(reward_pred, torch.tensor([reward], dtype=torch.float32).to(self.device))
            self.reward_optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.reward_optimizer.step()

        # Update based on comparative feedback
        for (movie_id1, movie_id2), preference in comparative_feedback:
            # Ensure state1 and state2 have exactly 2 features (user_id and movie_id)
            state1 = torch.tensor([[self.user_id, movie_id1]], dtype=torch.float32).to(self.device)
            state2 = torch.tensor([[self.user_id, movie_id2]], dtype=torch.float32).to(self.device)

            reward1 = self.reward_model(state1)
            reward2 = self.reward_model(state2)

            if preference == 1:
                loss = nn.ReLU()(1 - (reward1 - reward2))
            elif preference == -1:
                loss = nn.ReLU()(1 - (reward2 - reward1))
            else:
                loss = torch.abs(reward1 - reward2)

            self.reward_optimizer.zero_grad()
            loss.backward()
            self.reward_optimizer.step()

    def update_policy_model(self, X, y, movie_features, feedback):
        for user, movie, target, feature, fb in zip(X[:, 0], X[:, 1], y, movie_features, feedback):
            state = (int(user.item()), int(movie.item()))
            if state not in self.q_table:
                self.q_table[state] = [0, 0]
            
            action = 1 if fb > 0 else 0 if fb < 0 else torch.argmax(torch.tensor(self.q_table[state])).item()
            
            with torch.no_grad():
                next_q = self.model(user.unsqueeze(0), movie.unsqueeze(0), feature.unsqueeze(0)).item()
            
            old_q = self.q_table[state][action]
            reward = 1 if action == target.item() else -1
            self.q_table[state][action] = old_q + 0.1 * (reward + self.gamma * next_q - old_q)
        
        for _ in range(3):  # Policy model epochs
            for (user, movie), q_values in self.q_table.items():
                user_tensor = torch.tensor([user], dtype=torch.long).to(self.device)
                movie_tensor = torch.tensor([movie], dtype=torch.long).to(self.device)
                movie_feature = self.movie_features[movie_tensor]
                
                prediction = self.model(user_tensor, movie_tensor, movie_feature).view(1, 1)
                target = torch.tensor([[torch.argmax(torch.tensor(q_values))]], dtype=torch.float32).to(self.device)
                
                loss = self.criterion(prediction, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def get_comparative_feedback(self, user_id, movie_pairs):
        comparative_feedback = []
        for movie_id1, movie_id2 in movie_pairs:
            pred1 = self.model(torch.tensor([user_id], dtype=torch.long).to(self.device),
                               torch.tensor([movie_id1], dtype=torch.long).to(self.device),
                               self.movie_features[movie_id1].unsqueeze(0)).item()
            pred2 = self.model(torch.tensor([user_id], dtype=torch.long).to(self.device),
                               torch.tensor([movie_id2], dtype=torch.long).to(self.device),
                               self.movie_features[movie_id2].unsqueeze(0)).item()
            
            predicted_preference = 1 if pred1 > pred2 else -1 if pred1 < pred2 else 0
            feedback = self.human_feedback.get_comparative_feedback(user_id, movie_id1, movie_id2, predicted_preference)
            comparative_feedback.append(((movie_id1, movie_id2), feedback))
        
        return comparative_feedback