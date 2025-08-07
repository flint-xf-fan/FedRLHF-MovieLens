import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys
from glob import glob

# Append the parent directory to the system path to import custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from FedRLHF.utils.visualization import plot_learning_curves_with_variance,plot_learning_curves_with_variance_vertical_layout

def load_and_process_data(file_pattern):
    """
    Loads and processes training data from multiple pickle files.
    
    Args:
        file_pattern (str): Glob pattern to match pickle files.
    
    Returns:
        tuple: Contains mean and standard deviation of round accuracies,
               mean and standard deviation of client accuracies,
               and mean client Spearman correlations.
    """
    all_data = []
    for file_path in glob(file_pattern):
        with open(file_path, 'rb') as f:
            all_data.append(pickle.load(f))
    
    # Calculate means across all runs for weighted round accuracies
    mean_round_accuracies = np.mean([data['mean_round_accuracies'] for data in all_data], axis=0)
    std_round_accuracies = np.std([data['mean_round_accuracies'] for data in all_data], axis=0)
    
    # Calculate means and stds for client accuracies across all runs
    mean_client_accuracies = {
        client: np.mean([data['mean_client_accuracies'][client] for data in all_data], axis=0)
        for client in all_data[0]['mean_client_accuracies']
    }
    std_client_accuracies = {
        client: np.std([data['mean_client_accuracies'][client] for data in all_data], axis=0)
        for client in all_data[0]['mean_client_accuracies']
    }
    
    # Calculate mean Spearman correlations across all runs
    mean_client_spearman_corrs = {
        client: np.mean([data['client_spearman_corrs'][client] for data in all_data], axis=0)
        for client in all_data[0]['client_spearman_corrs']
    }
    
    return (
        mean_round_accuracies,
        std_round_accuracies,
        mean_client_accuracies,
        std_client_accuracies,
        mean_client_spearman_corrs
    )

# Define the path pattern to the saved data
save_path_pattern = os.path.join('training_metrics_results', 'ml-latest-small_10_clients_training_data_seed_*.pkl')

# Load and process the training data
mean_round_accuracies, std_round_accuracies, mean_client_accuracies, std_client_accuracies, mean_client_spearman_corrs = load_and_process_data(save_path_pattern)

# Generate rounds list based on the number of rounds
rounds = list(range(1, len(mean_round_accuracies) + 1))

# # Plot the learning curves with variance
# plot_learning_curves_with_variance(
#     rounds,
#     global_mean=mean_round_accuracies,            # Weighted mean accuracies
#     global_std=std_round_accuracies,             # Standard deviation of weighted mean accuracies
#     client_means=mean_client_accuracies,         # Mean accuracies per client
#     client_stds=std_client_accuracies,           # Std dev of accuracies per client
#     client_spearman_correlations=mean_client_spearman_corrs  # Mean Spearman correlations per client
# )

# Plot the learning curves with variance, but stack the two subplots vertically -- for main body
plot_learning_curves_with_variance_vertical_layout(
    rounds,
    global_mean=mean_round_accuracies,            # Weighted mean accuracies
    global_std=std_round_accuracies,             # Standard deviation of weighted mean accuracies
    client_means=mean_client_accuracies,         # Mean accuracies per client
    client_stds=std_client_accuracies,           # Std dev of accuracies per client
    client_spearman_correlations=mean_client_spearman_corrs  # Mean Spearman correlations per client
)

plt.show()
