"""
Visualization script for comparing state encoding performances in Linear Q-learning Snake.
Generates plots to analyze learning curves, convergence, and final performance metrics.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
ENCODINGS = {
    'basic': 'linear_q_training_stats_basic.csv',
    'bodyaware': 'linear_q_training_stats_bodyaware.csv',
    'distance': 'linear_q_training_stats_distance.csv',
    'localgrid': 'linear_q_training_stats_localgrid.csv',
    'raycasting': 'linear_q_training_stats_raycasting.csv',
}

COLORS = {
    'basic': '#1f77b4',
    'bodyaware': '#ff7f0e',
    'distance': '#2ca02c',
    'localgrid': '#d62728',
    'raycasting': '#9467bd',
}

WINDOW_SIZE = 50  # For rolling averages

CURRENT_DIR = Path(__file__).resolve().parent
TRAINING_CSV_DIR = CURRENT_DIR / "training_csv"
PLOTS_DIR = CURRENT_DIR / "training_plots"

def load_data():
    data = {}
    for encoding, filename in ENCODINGS.items():
        filepath = TRAINING_CSV_DIR / filename
        if filepath.exists():
            data[encoding] = pd.read_csv(filepath)
        else:
            print(f"Warning: {filename} not found")
    return data

def compute_rolling_avg(df, column, window=WINDOW_SIZE):
    return df[column].rolling(window=window, min_periods=1).mean()

def compute_success_rate(df, window=WINDOW_SIZE):
    return df['win'].rolling(window=window, min_periods=1).mean() * 100

def plot_rolling_avg_score(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for encoding, df in data.items():
        rolling_avg = compute_rolling_avg(df, 'score', WINDOW_SIZE)
        ax.plot(df['episode'], rolling_avg, label=encoding, color=COLORS[encoding], linewidth=2)
    
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel(f'Rolling Avg Score ({WINDOW_SIZE}-ep window)', fontsize=11)
    ax.set_title('Learning Convergence - Rolling Average Score', fontsize=13, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'linear_q_plot_1_rolling_avg_score.png', dpi=150)
    plt.close()

def plot_rolling_avg_reward(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for encoding, df in data.items():
        rolling_avg = compute_rolling_avg(df, 'reward', WINDOW_SIZE)
        ax.plot(df['episode'], rolling_avg, label=encoding, color=COLORS[encoding], linewidth=2)
    
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel(f'Rolling Avg Reward ({WINDOW_SIZE}-ep window)', fontsize=11)
    ax.set_title('Learning Convergence - Rolling Average Reward', fontsize=13, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'linear_q_plot_2_rolling_avg_reward.png', dpi=150)
    plt.close()

def plot_success_rate(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for encoding, df in data.items():
        success_rate = compute_success_rate(df, WINDOW_SIZE)
        ax.plot(df['episode'], success_rate, label=encoding, color=COLORS[encoding], linewidth=2)
    
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel(f'Success Rate (%) - {WINDOW_SIZE}-ep window', fontsize=11)
    ax.set_title('Training Success Rate - Rolling Percentage of Wins', fontsize=13, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'linear_q_plot_3_success_rate.png', dpi=150)
    plt.close()

def plot_final_performance(data):
    encodings = list(data.keys())
    
    avg_rewards = []
    avg_scores = []
    avg_steps = []
    total_wins = []
    
    for encoding, df in data.items():
        avg_rewards.append(df['reward'].mean())
        avg_scores.append(df['score'].mean())
        avg_steps.append(df['steps'].mean())
        total_wins.append(df['win'].sum())
    
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle('Final Performance Summary - All State Encodings', fontsize=14, fontweight='bold')
    
    # Subplot 1: Avg Reward
    axes[0, 0].bar(encodings, avg_rewards, color=[COLORS[e] for e in encodings], alpha=0.7)
    axes[0, 0].set_ylabel('Average Reward', fontsize=11)
    axes[0, 0].set_title('Average Episode Reward', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Subplot 2: Avg Score
    axes[0, 1].bar(encodings, avg_scores, color=[COLORS[e] for e in encodings], alpha=0.7)
    axes[0, 1].set_ylabel('Average Score', fontsize=11)
    axes[0, 1].set_title('Average Food Collected', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Subplot 3: Avg Steps
    axes[1, 0].bar(encodings, avg_steps, color=[COLORS[e] for e in encodings], alpha=0.7)
    axes[1, 0].set_ylabel('Average Steps', fontsize=11)
    axes[1, 0].set_title('Average Steps per Episode', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Subplot 4: Total Wins
    axes[1, 1].bar(encodings, total_wins, color=[COLORS[e] for e in encodings], alpha=0.7)
    axes[1, 1].set_ylabel('Total Wins', fontsize=11)
    axes[1, 1].set_title('Total Episodes with Perfect Score (Win)', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    for ax in axes.flat:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'linear_q_plot_4_final_performance.png', dpi=150)
    plt.close()

def plot_training_efficiency(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    milestones = [1, 2, 3, 4, 5, 8, 10]
    
    for encoding, df in data.items():
        episodes_to_milestone = []
        for milestone in milestones:
            first_episode = df[df['score'] >= milestone]['episode'].min()
            episodes_to_milestone.append(first_episode if not pd.isna(first_episode) else len(df))
        
        ax.plot(milestones, episodes_to_milestone, marker='o', label=encoding, 
                color=COLORS[encoding], linewidth=2, markersize=7)
    
    ax.set_xlabel('Score Milestone (Food Collected)', fontsize=11)
    ax.set_ylabel('Episodes Required', fontsize=11)
    ax.set_title('Training Efficiency - Episodes to Reach Score Milestones', fontsize=13, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'linear_q_plot_5_training_efficiency.png', dpi=150)
    plt.close()

def plot_reward_distribution(data):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    reward_data = [data[encoding]['reward'].values for encoding in ENCODINGS.keys()]
    
    bp = ax.boxplot(reward_data, labels=list(ENCODINGS.keys()), patch_artist=True)
    
    for patch, encoding in zip(bp['boxes'], ENCODINGS.keys()):
        patch.set_facecolor(COLORS[encoding])
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Episode Reward', fontsize=11)
    ax.set_title('Reward Distribution - All Training Episodes', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'linear_q_plot_6_reward_distribution.png', dpi=150)
    plt.close()

def plot_stability_variance(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for encoding, df in data.items():
        rolling_variance = df['reward'].rolling(window=WINDOW_SIZE, min_periods=1).var()
        ax.plot(df['episode'], rolling_variance, label=encoding, color=COLORS[encoding], linewidth=2)
    
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel(f'Rolling Reward Variance ({WINDOW_SIZE}-ep window)', fontsize=11)
    ax.set_title('Training Stability - Reward Variance Over Time', fontsize=13, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'linear_q_plot_7_stability_variance.png', dpi=150)
    plt.close()

def main():
    print("Loading training data...")
    data = load_data()
    
    if not data:
        print("Error: No training data files found!")
        return
    
    print(f"Loaded data for {len(data)} encoding(s): {', '.join(data.keys())}")
    
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print("\nGenerating plots...")
    
    # Core plots
    print("  - Plot 1: Rolling average score...")
    plot_rolling_avg_score(data)
    
    print("  - Plot 2: Rolling average reward...")
    plot_rolling_avg_reward(data)
    
    print("  - Plot 3: Success rate...")
    plot_success_rate(data)
    
    print("  - Plot 4: Final performance summary...")
    plot_final_performance(data)
    
    # Optional plots
    print("  - Plot 5: Training efficiency...")
    plot_training_efficiency(data)
    
    print("  - Plot 6: Reward distribution...")
    plot_reward_distribution(data)
    
    print("  - Plot 7: Stability variance...")
    plot_stability_variance(data)
    
    print("\nAll plots saved successfully!")
    print("Generated files:")
    print("  - linear_q_plot_1_rolling_avg_score.png")
    print("  - linear_q_plot_2_rolling_avg_reward.png")
    print("  - linear_q_plot_3_success_rate.png")
    print("  - linear_q_plot_4_final_performance.png")
    print("  - linear_q_plot_5_training_efficiency.png")
    print("  - linear_q_plot_6_reward_distribution.png")
    print("  - linear_q_plot_7_stability_variance.png")

if __name__ == '__main__':
    main()