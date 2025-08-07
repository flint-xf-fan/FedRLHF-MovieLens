print("Starting main.py")
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle  # Added for saving data

# Add the parent directory to sys.path to find FedRLHF module
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
print(f"Added to Python path: {parent_dir}")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")

from fed_rlhf.client import MovieLensClient
from fed_rlhf.server import MovieLensServer
from data.movielens import MovieLensDataset
from utils.visualization import plot_learning_curves_with_variance

print("All imports successful")

import torch
import flwr as fl

from configure import N_RUNS, N_CLIENTS, N_ROUNDS_FED, BATCH_SIZE

# Create the folder if it doesn't exist
os.makedirs("training_metrics_results", exist_ok=True)

def main():
    dataset_version = 'ml-latest-small'
    # dataset_version = 'ml-25m'
    dataset_path = 'datasets/' + dataset_version

    dataset = MovieLensDataset(
        ratings_path=os.path.join(dataset_path, 'ratings.csv'),
        movies_path=os.path.join(dataset_path, 'movies.csv')
    )
    
    num_clients = N_CLIENTS
    num_runs = N_RUNS
    n_rounds_fed = N_ROUNDS_FED

    all_round_accuracies = []
    all_client_spearman_correlations = {}
    all_client_accuracies = {}

    for run in range(num_runs):
        print(f"Starting run {run + 1}/{num_runs}")
        seed = 123
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        clients = []
        for i in range(num_clients):
            print('init client ', i)
            client = MovieLensClient(
                user_id=i,
                dataset=dataset,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                batch_size=BATCH_SIZE
            )
            clients.append(client)
        


        strategy = MovieLensServer(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
        )
        
        fl.server.start_server(
            server_address="127.0.0.1:8081",
            strategy=strategy,
            config=fl.server.ServerConfig(num_rounds=n_rounds_fed),
            grpc_max_message_length=1024 * 1024 * 1024  # 1 GB,
        )
        
        all_round_accuracies.append(strategy.round_accuracies)

        # Collect Spearman correlations per client
        for client_id, spearman_corrs in strategy.client_spearman_correlations.items():
            if client_id not in all_client_spearman_correlations:
                all_client_spearman_correlations[client_id] = []
            all_client_spearman_correlations[client_id].extend(spearman_corrs)

        # Collect client accuracies per round
        for client_id, accuracies in strategy.client_accuracies.items():
            if client_id not in all_client_accuracies:
                all_client_accuracies[client_id] = []
            all_client_accuracies[client_id].append(accuracies)

    # Calculate mean and standard deviation for accuracies
    mean_round_accuracies = np.mean(all_round_accuracies, axis=0)
    std_round_accuracies = np.std(all_round_accuracies, axis=0)
    
    # Organize Spearman correlations for visualization
    client_spearman_corrs = {client: np.array(corrs) for client, corrs in all_client_spearman_correlations.items()}

    mean_client_accuracies = {client: np.mean(accs, axis=0) for client, accs in all_client_accuracies.items()}
    std_client_accuracies = {client: np.std(accs, axis=0) for client, accs in all_client_accuracies.items()}
    
    # Prepare data to be saved
    training_data = {
        'all_round_accuracies': all_round_accuracies,
        'all_client_spearman_correlations': all_client_spearman_correlations,
        'all_client_accuracies': all_client_accuracies,
        'mean_round_accuracies': mean_round_accuracies,
        'std_round_accuracies': std_round_accuracies,
        'mean_client_accuracies': mean_client_accuracies,
        'std_client_accuracies': std_client_accuracies,
        'client_spearman_corrs': client_spearman_corrs,
        'num_clients': num_clients,
        'client_num_examples_per_round': strategy.client_num_examples_per_round,
        'num_runs': num_runs,
        'n_rounds_fed': n_rounds_fed,
        'batch_size': BATCH_SIZE,
        'dataset_version': dataset_version
    }

    # Define the path for saving the data
    save_filename = dataset_version + '_' + str(N_CLIENTS) + '_clients_training_data_seed_' + str(seed) + '.pkl'
    save_path = os.path.join('training_metrics_results', save_filename)

    # Save the training data using pickle
    with open(save_path, 'wb') as f:
        pickle.dump(training_data, f)

    print(f"Training metrics saved to {save_path}")

    # Optionally, you can keep the plotting function commented out or remove it
    # If you wish to plot later, use a separate script to load the saved data and generate plots

    # rounds = list(range(1, len(mean_round_accuracies) + 1))
    # plot_learning_curves_with_variance(
    #     rounds,
    #     global_mean=mean_round_accuracies,
    #     global_std=std_round_accuracies,
    #     client_means=mean_client_accuracies,  # Use mean_client_accuracies instead of strategy.client_accuracies
    #     client_stds=std_client_accuracies,
    #     client_spearman_correlations=client_spearman_corrs
    # )

if __name__ == "__main__":
    main()
