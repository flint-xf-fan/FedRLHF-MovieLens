import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from glob import glob

def load_and_process_data(file_pattern):
    """
    Loads and processes client accuracies and Spearman correlations from multiple pickle files.
    
    Args:
        file_pattern (str): Glob pattern to match pickle files.
    
    Returns:
        tuple: Contains mean client accuracies and mean client Spearman correlations.
    """
    all_data = []
    for file_path in glob(file_pattern):
        with open(file_path, 'rb') as f:
            all_data.append(pickle.load(f))

    # Calculate means across all runs for client accuracies
    mean_client_accuracies = {
        client: np.mean([data['mean_client_accuracies'][client] for data in all_data], axis=0)
        for client in all_data[0]['mean_client_accuracies']
    }
    
    # Calculate means across all runs for client Spearman correlations
    mean_client_spearman_corrs = {
        client: np.mean([data['client_spearman_corrs'][client] for data in all_data], axis=0)
        for client in all_data[0]['client_spearman_corrs']
    }

    return mean_client_accuracies, mean_client_spearman_corrs

def load_global_accuracies(file_pattern):
    """
    Loads and computes the mean global (weighted) accuracies across multiple pickle files.
    
    Args:
        file_pattern (str): Glob pattern to match pickle files.
    
    Returns:
        tuple: Contains mean global accuracies and their standard deviations.
    """
    all_data = []
    for file_path in glob(file_pattern):
        with open(file_path, 'rb') as f:
            all_data.append(pickle.load(f))
    
    # Calculate means across all runs for weighted global accuracies
    mean_round_accuracies = np.mean([data['mean_round_accuracies'] for data in all_data], axis=0)
    std_round_accuracies = np.std([data['mean_round_accuracies'] for data in all_data], axis=0)
    
    return mean_round_accuracies, std_round_accuracies

def main():
    # Define the path pattern to the saved data
    save_path_pattern = os.path.join('training_metrics_results', 'ml-latest-small_50_clients_training_data_seed_*.pkl')
    
    # Load and process the training data
    mean_client_accuracies, mean_client_spearman_corrs = load_and_process_data(save_path_pattern)
    mean_round_accuracies, std_round_accuracies = load_global_accuracies(save_path_pattern)
    
    # Determine the number of rounds
    n_rounds = len(mean_round_accuracies)
    rounds = list(range(1, n_rounds + 1))  # Assuming rounds start at 1
    
    # Prepare data for plotting
    accuracy_data = []
    spearman_data = []
    for round_idx in range(n_rounds):
        accuracies = [mean_client_accuracies[client][round_idx] for client in mean_client_accuracies]
        spearman_corrs = [mean_client_spearman_corrs[client][round_idx] for client in mean_client_spearman_corrs]
        accuracy_data.append(accuracies)
        spearman_data.append(spearman_corrs)
    
    # Handle NaN values in Spearman correlations
    spearman_data = [
        [val if not np.isnan(val) else 0 for val in spearman_round]
        for spearman_round in spearman_data
    ]
    
    # Set the visual theme for seaborn
    sns.set_theme(style="whitegrid")
    
    # Create subplots for violin and box plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # -------------------
    # Top Subplot: Violin Plot for Client Accuracies
    # -------------------
    violin_parts = ax1.violinplot(
        accuracy_data,
        positions=rounds,
        showmeans=True,
        showextrema=True,
        showmedians=False,
        widths=0.8  # Adjusted width for better visualization
    )
    
    # Customize the violin plots
    ax1.set_xlabel('Rounds', fontsize=16)
    ax1.set_ylabel('Client Accuracy', fontsize=16)
    ax1.set_title('Distribution of K=' + str(len(mean_client_accuracies)) + ' Client Accuracies per Round', fontsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    
    # Optional: Customize violin plot colors (keeping default as per your instruction)
    # If needed, you can adjust colors here without altering the overall style
    
    # Overlay the server-reported global (weighted) accuracy as a single continuous line
    ax1.plot(
        rounds,
        mean_round_accuracies,
        label='Global Model',
        color='red',
        linestyle='--',
        linewidth=2
    )
    
    # Optionally, add a shaded area representing the standard deviation
    ax1.fill_between(
        rounds,
        mean_round_accuracies - std_round_accuracies,
        mean_round_accuracies + std_round_accuracies,
        color='red',
        alpha=0.2,
        # label='Mean Accuracy of 50 Clients ± Std Dev'
    )
    
    # Add legend to differentiate global accuracy
    ax1.legend(loc='upper left', fontsize=14)
    
    # -------------------
    # Bottom Subplot: Box Plot for Client Spearman Rank Correlations
    # -------------------
    box_parts = ax2.boxplot(
        spearman_data,
        positions=rounds,
        widths=0.5,
        patch_artist=True,
        showfliers=False
    )
    
    # Customize the box plots
    ax2.set_xlabel('Rounds', fontsize=16)
    ax2.set_ylabel('Spearman Rank Correlation Distribution', fontsize=16)
    ax2.set_title('Distribution of K=' + str(len(mean_client_accuracies)) + ' Client Spearman Rank Correlations per Round', fontsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    
    # Customize box plot colors
    for patch in box_parts['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    # Remove unnecessary legends if any (since no global line is added here)
    
    # Add grid for better readability
    ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    
    plt.tight_layout()
    
    # Save the figure as a PDF
    plt.savefig('violin_box_plot_k' + str(len(mean_client_accuracies)) + '.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print('Plots have been generated and saved as violin_box_plot_k' + str(len(mean_client_accuracies)) + '.pdf.')

if __name__ == "__main__":
    main()
