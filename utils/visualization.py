import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_learning_curves(rounds, global_performance, client_performances):
    plt.figure(figsize=(12, 6))
    
    # Plot global performance
    plt.plot(rounds, global_performance, label='Global Model', linewidth=3, color='black')
    
    # Plot client performances
    for client, perf in client_performances.items():
        plt.plot(rounds[:len(perf)], perf, label=f'{client}', alpha=0.7)
    
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy')
    plt.title('Learning Curves: Global vs Client Models')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('learning_curves.png')
    plt.close()

# def plot_learning_curves_with_variance(rounds, global_mean, global_std, client_means, client_stds):
#     plt.figure(figsize=(12, 8))
    
#     # Set the style 
#     sns.set_theme(style="darkgrid")
#     # palette = ["#4C72B0", "#DD8452", "#C44E52"] 

    
#     # Plot client performances
#     colors = plt.cm.Set1(np.linspace(0, 1, len(client_means)))
#     for (client, mean), (_, std), color in zip(client_means.items(), client_stds.items(), colors):
#         plt.plot(rounds[:len(mean)], mean, label=f'{client}', alpha=0.8, color=color, linewidth=2)
#         plt.fill_between(rounds[:len(mean)], mean - std, mean + std, alpha=0.2, color=color)
    
#     # Plot global performance
#     plt.plot(rounds, global_mean, label='Global Model', linewidth=3, color='red', linestyle='--')
#     plt.fill_between(rounds, global_mean - global_std, global_mean + global_std, color='red', alpha=0.1)
    
#     plt.xlabel('Rounds', fontsize=14)
#     plt.ylabel('Accuracy', fontsize=14)
#     plt.title('Learning Curves: Global vs Client Models', fontsize=16)
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
#     plt.grid(True, alpha=0.3)
    
#     # Remove top and right spines
#     plt.gca().spines['top'].set_visible(False)
#     plt.gca().spines['right'].set_visible(False)
    
#     # Increase tick label font size
#     plt.tick_params(axis='both', which='major', labelsize=10)
    
#     plt.tight_layout()
#     plt.savefig('learning_curves_with_variance.png', dpi=300, bbox_inches='tight')
#     plt.close()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# def plot_learning_curves_with_variance(rounds, global_mean, global_std, client_means, client_stds, spearman_per_round):  ### UPDATED SIGNATURE ###
#     # Set the Seaborn style
#     sns.set_theme(style="darkgrid")

#     # Create subplots: One for accuracy, one for Spearman's correlation
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))  # Two subplots, vertically stacked

#     # First Subplot: Accuracy (Global and Clients)
#     # Plot global performance
#     ax1.plot(rounds, global_mean, label='Global Model', linewidth=2, color='red', linestyle='--')
#     ax1.fill_between(rounds, global_mean - global_std, global_mean + global_std, color='red', alpha=0.2)

#     # Plot client performances
#     num_clients = len(client_means)
#     palette = plt.cm.get_cmap("tab10", num_clients)  # Choose a colormap and number of colors

#     for i, ((client, mean), (_, std)) in enumerate(zip(client_means.items(), client_stds.items())):
#         color = palette(i)
#         ax1.plot(rounds[:len(mean)], mean, label=f'{client}', alpha=0.8, color=color, linewidth=2)
#         ax1.fill_between(rounds[:len(mean)], mean - std, mean + std, alpha=0.2, color=color)

#     ax1.set_xlabel('Rounds', fontsize=14)
#     ax1.set_ylabel('Accuracy', fontsize=14)
#     ax1.set_title('Learning Curves: Global vs Client Accuracy', fontsize=14)
#     ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
#     ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

#     # Second Subplot: Spearman's Rank Correlation
#     for client_idx, spearman_corr in enumerate(spearman_per_round.T):
#         ax2.plot(rounds, spearman_corr, label=f'Client {client_idx}', alpha=0.8, linewidth=2)

#     ax2.set_xlabel('Rounds', fontsize=14)
#     ax2.set_ylabel('Spearman Rank Correlation', fontsize=14)
#     ax2.set_title('Learning Curves: Client Spearman Rank Correlation', fontsize=14)
#     ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
#     ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

#     # Adjust layout
#     plt.tight_layout()
#     plt.savefig('learning_curves_with_two_metrics.png', dpi=300, bbox_inches='tight')
#     plt.close()


def plot_learning_curves_with_variance(rounds, global_mean, global_std, client_means, client_stds, client_spearman_correlations):
    sns.set_theme(style="darkgrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, len(client_means)))


    # Print values in terminal
    print("Global Model:")
    for round, mean, std in zip(rounds, global_mean, global_std):
        print(f"Round {round}: Mean = {mean:.4f}, Std = {std:.4f}")

    print("\nClient Models:")
    for client, accuracies in client_means.items():
        print(f"\n{client}:")
        for round, acc in zip(rounds[:len(accuracies)], accuracies):
            print(f"Round {round}: Accuracy = {acc:.4f}")

    print("\nClient Spearman Correlations:")
    for client, spearman_corrs in client_spearman_correlations.items():
        print(f"\n{client}:")
        for round, corr in zip(rounds[:len(spearman_corrs)], spearman_corrs):
            print(f"Round {round}: Correlation = {corr:.4f}")

    # First Subplot: Accuracy (Global and Clients)
    ax1.plot(rounds, global_mean, label='Global Model', linewidth=3, color='red', linestyle='--')
    ax1.fill_between(rounds, global_mean - global_std, global_mean + global_std, color='red', alpha=0.2)

    for (client, accuracies), color in zip(client_means.items(), colors):
        ax1.plot(rounds[:len(accuracies)], accuracies, label=client, alpha=0.8, linewidth=2, color=color)

    ax1.set_xlabel('Rounds', fontsize=16)
    ax1.set_ylabel('Accuracy', fontsize=16)
    ax1.set_title('Learning Curves: Mean Global vs Client Accuracy', fontsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    # Second Subplot: Spearman's Rank Correlation
    for (client, spearman_corrs), color in zip(client_spearman_correlations.items(), colors):
        ax2.plot(rounds[:len(spearman_corrs)], spearman_corrs, label=client, alpha=0.8, linewidth=2, color=color)

    ax2.set_xlabel('Rounds', fontsize=16)
    ax2.set_ylabel('Spearman Rank Correlation', fontsize=16)
    ax2.set_title('Learning Curves: Mean Client Spearman Rank Correlation', fontsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(handles), 
               bbox_to_anchor=(0.05, 0.02, 0.9, 0.02),
               mode="expand", borderaxespad=0.,
               fontsize=16, handlelength=2, columnspacing=1)

    fig.subplots_adjust(bottom=0.15)
    # plt.savefig('mean_learning_curves_with_two_metrics.png', dpi=300, bbox_inches='tight')
    plt.savefig('K_' + str(len(client_means)) + '_mean_learning_curves_with_two_metrics.pdf', dpi=300, bbox_inches='tight')
    plt.close()


def plot_learning_curves_with_variance_vertical_layout(rounds, global_mean, global_std, client_means, client_stds, client_spearman_correlations):
    sns.set_theme(style="darkgrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))  # Slightly reduced width
    colors = plt.cm.tab10(np.linspace(0, 1, len(client_means)))

    # First Subplot: Accuracy (Global and Clients)
    ax1.plot(rounds, global_mean, label='Global Model', linewidth=3, color='red', linestyle='--')
    ax1.fill_between(rounds, global_mean - global_std, global_mean + global_std, color='red', alpha=0.2)

    for (client, accuracies), color in zip(client_means.items(), colors):
        ax1.plot(rounds[:len(accuracies)], accuracies, label=client, alpha=0.8, linewidth=2, color=color)

    ax1.set_xlabel('Rounds', fontsize=16)
    ax1.set_ylabel('Accuracy', fontsize=16)
    ax1.set_title('Learning Curves: Mean Global vs Client Accuracy', fontsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    # Second Subplot: Spearman's Rank Correlation
    for (client, spearman_corrs), color in zip(client_spearman_correlations.items(), colors):
        ax2.plot(rounds[:len(spearman_corrs)], spearman_corrs, label=client, alpha=0.8, linewidth=2, color=color)

    ax2.set_xlabel('Rounds', fontsize=16)
    ax2.set_ylabel('Spearman Rank Correlation', fontsize=16)
    ax2.set_title('Learning Curves: Mean Client Spearman Rank Correlation', fontsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    
    # Create a single legend for both subplots
    handles, labels = ax1.get_legend_handles_labels()
    
    # put the global legend to the last 
    def move_first_to_last(t):

        if len(t) > 0:
            first_element = t.pop(0)
            t.append(first_element)

        return t
    handles_ordered = move_first_to_last(handles)
    labels_ordered = move_first_to_last(labels)

    fig.legend(handles_ordered, labels_ordered, loc='upper center', ncol=6,  # Increased to 6 columns
               bbox_to_anchor=(0.527, 0.1),  # Moved slightly up
               fontsize=13, handlelength=1.8, columnspacing=1.3,
               bbox_transform=fig.transFigure)  # Use figure coordinates

    fig.subplots_adjust(bottom=0.15, hspace=0.3)  # Adjusted bottom margin and added space between subplots
    plt.savefig(f'K_{len(client_means)}_mean_learning_curves_with_two_metrics-Vertical.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_personalization_heatmap(user_movie_preferences):
    plt.figure(figsize=(15, 10))
    sns.heatmap(user_movie_preferences, cmap='YlGnBu')
    plt.xlabel('Movies')
    plt.ylabel('Users')
    plt.title('User-Movie Preference Heatmap')
    plt.savefig('preference_heatmap.png')
    plt.close()