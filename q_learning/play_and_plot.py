import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

from snake_env import Snake_Env
from q_learning_agent import Q_learning_Agent
from state_encoding import State_Encoding
from state_encoding_distance import DistanceStateEncoding
from state_encoding_raycasting import RayCastingStateEncoding
from state_encoding_localgrid import LocalGridStateEncoding
from state_encoding_bodyawareness import BodyAwarenessStateEncoding

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CURRENT_DIR = Path(__file__).resolve().parent
GAME_DIR = PROJECT_ROOT / "game"
TRAINING_CSV_DIR = CURRENT_DIR / "training_csv"
TEST_CSV_DIR = CURRENT_DIR / "test_csv"
PLOTS_DIR = CURRENT_DIR / "test_plots"

for path in (PROJECT_ROOT, GAME_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Color scheme for consistent visualization
COLORS = {
    'basic': '#1f77b4',
    'bodyaware': '#ff7f0e',
    'distance': '#2ca02c',
    'localgrid': '#d62728',
    'raycasting': '#9467bd',
}

WINDOW_SIZE = 50  # For rolling averages

# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_q_table(filepath):
    try:
        with open(filepath, 'rb') as f:
            q_table_dict = pickle.load(f)
        q_table = defaultdict(lambda: np.zeros(4, dtype=np.float32))
        q_table.update(q_table_dict)
        return q_table
    except FileNotFoundError:
        print(f"Error: Q-table file not found at {filepath}")
        return None

def load_all_q_tables():
    q_tables = {}
    encodings = ['basic', 'bodyaware', 'distance', 'localgrid', 'raycasting']
    
    models_dir = CURRENT_DIR / "models"
    for encoding in encodings:
        q_table_file = models_dir / f"q_learning_q_table_{encoding}.pkl"
        q_table = load_q_table(q_table_file)
        if q_table is not None:
            q_tables[encoding] = q_table
        else:
            print(f"Warning: Could not load Q-table for {encoding}")
    
    return q_tables

def load_training_data():
    data = {}
    encodings = ['basic', 'bodyaware', 'distance', 'localgrid', 'raycasting']
    
    for encoding in encodings:
        filepath = TRAINING_CSV_DIR / f"training_stats_{encoding}.csv"
        if filepath.exists():
            data[encoding] = pd.read_csv(filepath)
        else:
            print(f"Warning: Training stats for {encoding} not found")
    
    return data

# ==============================================================================
# PLAYING/TESTING THE AGENT
# ==============================================================================

def play_episode(env, agent, state_encoder):
    state = env.reset()
    episode_reward = 0.0
    episode_done = False
    
    while not episode_done:
        action = agent.choose_action(state)
        next_state, reward, done, info = env.step(action)
        
        episode_reward += reward
        episode_done = done
        state = next_state
    
    # Check if episode was a win (filled entire board)
    total_cells = env.board.cols * env.board.rows
    is_win = len(env.snake.snake) >= total_cells
    
    return episode_reward, info['score'], info['steps'], is_win

def test_all_encodings(q_tables, num_episodes=100, show_progress=True):
    encodings_dict = {
        'basic': State_Encoding(),
        'distance': DistanceStateEncoding(),
        'raycasting': RayCastingStateEncoding(),
        'localgrid': LocalGridStateEncoding(),
        'bodyaware': BodyAwarenessStateEncoding(),
    }
    
    results = {}
    
    for encoding_name, state_encoder in encodings_dict.items():
        if encoding_name not in q_tables:
            print(f"Skipping {encoding_name}: Q-table not loaded")
            continue
        
        if show_progress:
            print(f"\nTesting {encoding_name.upper()} encoding ({num_episodes} episodes)...")
        
        results[encoding_name] = {
            'episode': [],
            'reward': [],
            'score': [],
            'steps': [],
            'win': []
        }
        
        env = Snake_Env(
            render_mode=False,  # No rendering for fast testing
            max_steps_per_episode=1000,
            food_reward=10,
            death_penalty=-10,
            per_step_reward=-0.1,
            reward_for_winning=1000,
            state_encoder=state_encoder,
        )
        
        agent = Q_learning_Agent(
            action_size=4,
            learning_rate=0.1,
            gamma=0.9,
            epsilon=0.0,  # No exploration, only exploitation
            epsilon_min=0.0,
            epsilon_decay=1.0,
        )
        agent.q_table = q_tables[encoding_name]
        
        for episode in range(num_episodes):
            episode_reward, score, steps, is_win = play_episode(env, agent, state_encoder)
            
            results[encoding_name]['episode'].append(episode)
            results[encoding_name]['reward'].append(episode_reward)
            results[encoding_name]['score'].append(score)
            results[encoding_name]['steps'].append(steps)
            results[encoding_name]['win'].append(int(is_win))
            
            if show_progress and (episode + 1) % 20 == 0:
                print(f"  Episodes {episode + 1}/{num_episodes} - "
                      f"Avg Reward: {np.mean(results[encoding_name]['reward']):.2f}, "
                      f"Avg Score: {np.mean(results[encoding_name]['score']):.2f}")
        
        env.close() if hasattr(env, 'close') else None
        
        # Convert to DataFrame for easier plotting
        results[encoding_name] = pd.DataFrame(results[encoding_name])
    
    return results

# ==============================================================================
# PLOTTING FUNCTIONS
# ==============================================================================

def compute_rolling_avg(df, column, window=WINDOW_SIZE):
    return df[column].rolling(window=window, min_periods=1).mean()

def compute_success_rate(df, window=WINDOW_SIZE):
    return df['win'].rolling(window=window, min_periods=1).mean() * 100

def plot_rolling_avg_score(data, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        standalone = True
    else:
        standalone = False
    
    for encoding in sorted(data.keys()):
        df = data[encoding]
        rolling_avg = compute_rolling_avg(df, 'score', WINDOW_SIZE)
        ax.plot(df['episode'], rolling_avg, label=encoding, 
                color=COLORS[encoding], linewidth=2)
    
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel(f'Rolling Avg Score ({WINDOW_SIZE}-ep window)', fontsize=11)
    ax.set_title('Learning Convergence - Rolling Average Score', 
                 fontsize=13, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    if standalone:
        plt.tight_layout()
        return fig, ax
    return ax

def plot_rolling_avg_reward(data, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        standalone = True
    else:
        standalone = False
    
    for encoding in sorted(data.keys()):
        df = data[encoding]
        rolling_avg = compute_rolling_avg(df, 'reward', WINDOW_SIZE)
        ax.plot(df['episode'], rolling_avg, label=encoding, 
                color=COLORS[encoding], linewidth=2)
    
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel(f'Rolling Avg Reward ({WINDOW_SIZE}-ep window)', fontsize=11)
    ax.set_title('Learning Convergence - Rolling Average Reward', 
                 fontsize=13, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    if standalone:
        plt.tight_layout()
        return fig, ax
    return ax

def plot_success_rate(data, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        standalone = True
    else:
        standalone = False
    
    for encoding in sorted(data.keys()):
        df = data[encoding]
        success_rate = compute_success_rate(df, WINDOW_SIZE)
        ax.plot(df['episode'], success_rate, label=encoding, 
                color=COLORS[encoding], linewidth=2)
    
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel(f'Success Rate (%) - {WINDOW_SIZE}-ep window', fontsize=11)
    ax.set_title('Training Success Rate - Rolling Percentage of Wins', 
                 fontsize=13, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    if standalone:
        plt.tight_layout()
        return fig, ax
    return ax

def plot_reward_variance(data, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        standalone = True
    else:
        standalone = False
    
    for encoding in sorted(data.keys()):
        df = data[encoding]
        rolling_variance = df['reward'].rolling(window=WINDOW_SIZE, min_periods=1).var()
        ax.plot(df['episode'], rolling_variance, label=encoding, 
                color=COLORS[encoding], linewidth=2)
    
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel(f'Rolling Reward Variance ({WINDOW_SIZE}-ep window)', fontsize=11)
    ax.set_title('Training Stability - Reward Variance Over Time', 
                 fontsize=13, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    if standalone:
        plt.tight_layout()
        return fig, ax
    return ax

def plot_reward_distribution(data, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        standalone = True
    else:
        standalone = False
    
    encodings = sorted(data.keys())
    reward_data = [data[encoding]['reward'].values for encoding in encodings]
    
    bp = ax.boxplot(reward_data, labels=encodings, patch_artist=True)
    
    for patch, encoding in zip(bp['boxes'], encodings):
        patch.set_facecolor(COLORS[encoding])
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Episode Reward', fontsize=11)
    ax.set_title('Reward Distribution - All Training Episodes', 
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    if standalone:
        plt.tight_layout()
        return fig, ax
    return ax

def plot_final_performance_summary(data, ax=None):
    if ax is None:
        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
        standalone = True
    else:
        axes = ax
        standalone = False
    
    encodings = sorted(data.keys())
    
    avg_rewards = [data[e]['reward'].mean() for e in encodings]
    avg_scores = [data[e]['score'].mean() for e in encodings]
    avg_steps = [data[e]['steps'].mean() for e in encodings]
    total_wins = [data[e]['win'].sum() for e in encodings]
    
    # Subplot 1: Avg Reward
    axes[0, 0].bar(encodings, avg_rewards, color=[COLORS[e] for e in encodings], alpha=0.7)
    axes[0, 0].set_ylabel('Average Reward', fontsize=11)
    axes[0, 0].set_title('Average Episode Reward', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Subplot 2: Avg Score
    axes[0, 1].bar(encodings, avg_scores, color=[COLORS[e] for e in encodings], alpha=0.7)
    axes[0, 1].set_ylabel('Average Score', fontsize=11)
    axes[0, 1].set_title('Average Food Collected', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Subplot 3: Avg Steps
    axes[1, 0].bar(encodings, avg_steps, color=[COLORS[e] for e in encodings], alpha=0.7)
    axes[1, 0].set_ylabel('Average Steps', fontsize=11)
    axes[1, 0].set_title('Average Steps per Episode', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Subplot 4: Total Wins
    axes[1, 1].bar(encodings, total_wins, color=[COLORS[e] for e in encodings], alpha=0.7)
    axes[1, 1].set_ylabel('Total Wins', fontsize=11)
    axes[1, 1].set_title('Total Episodes with Perfect Score (Win)', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    if standalone:
        fig.suptitle('Final Performance Summary - All State Encodings', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig, axes
    
    return axes

def plot_training_efficiency(data, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        standalone = True
    else:
        standalone = False
    
    milestones = [1, 2, 3, 4, 5, 8, 10]
    
    for encoding in sorted(data.keys()):
        df = data[encoding]
        episodes_to_milestone = []
        for milestone in milestones:
            first_episode = df[df['score'] >= milestone]['episode'].min()
            episodes_to_milestone.append(first_episode if not pd.isna(first_episode) else len(df) + 100)
        
        ax.plot(milestones, episodes_to_milestone, marker='o', label=encoding,
                color=COLORS[encoding], linewidth=2, markersize=7)
    
    ax.set_xlabel('Score Milestone (Food Collected)', fontsize=11)
    ax.set_ylabel('Episodes Required', fontsize=11)
    ax.set_title('Training Efficiency - Episodes to Reach Score Milestones', 
                 fontsize=13, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    if standalone:
        plt.tight_layout()
        return fig, ax
    return ax

def generate_and_save_plots(data, output_dir=None):
    if output_dir is None:
        output_dir = PLOTS_DIR
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("GENERATING AND SAVING PLOTS".center(80))
    print("="*80 + "\n")
    
    # Plot 1: Rolling Average Score
    print("  Generating: Rolling Average Score...")
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_rolling_avg_score(data, ax=ax)
    fig.tight_layout()
    filepath = output_dir / "plot_1_rolling_avg_score.png"
    fig.savefig(filepath, dpi=150)
    print(f"    ✓ Saved: {filepath.name}")
    
    # Plot 2: Rolling Average Reward
    print("  Generating: Rolling Average Reward...")
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_rolling_avg_reward(data, ax=ax)
    fig.tight_layout()
    filepath = output_dir / "plot_2_rolling_avg_reward.png"
    fig.savefig(filepath, dpi=150)
    print(f"    ✓ Saved: {filepath.name}")
    
    # Plot 3: Success Rate
    print("  Generating: Training Success Rate...")
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_success_rate(data, ax=ax)
    fig.tight_layout()
    filepath = output_dir / "plot_3_success_rate.png"
    fig.savefig(filepath, dpi=150)
    print(f"    ✓ Saved: {filepath.name}")
    
    # Plot 4: Reward Variance (Stability)
    print("  Generating: Training Stability (Reward Variance)...")
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_reward_variance(data, ax=ax)
    fig.tight_layout()
    filepath = output_dir / "plot_4_stability_variance.png"
    fig.savefig(filepath, dpi=150)
    print(f"    ✓ Saved: {filepath.name}")
    
    # Plot 5: Training Efficiency
    print("  Generating: Training Efficiency...")
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_training_efficiency(data, ax=ax)
    fig.tight_layout()
    filepath = output_dir / "plot_5_training_efficiency.png"
    fig.savefig(filepath, dpi=150)
    print(f"    ✓ Saved: {filepath.name}")
    
    # Plot 6: Reward Distribution
    print("  Generating: Reward Distribution...")
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_reward_distribution(data, ax=ax)
    fig.tight_layout()
    filepath = output_dir / "plot_6_reward_distribution.png"
    fig.savefig(filepath, dpi=150)
    print(f"    ✓ Saved: {filepath.name}")
    
    # Plot 7: Final Performance Summary (4-panel)
    print("  Generating: Final Performance Summary...")
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle('Final Performance Summary - All State Encodings', 
                 fontsize=14, fontweight='bold')
    
    encodings = sorted(data.keys())
    avg_rewards = [data[e]['reward'].mean() for e in encodings]
    avg_scores = [data[e]['score'].mean() for e in encodings]
    avg_steps = [data[e]['steps'].mean() for e in encodings]
    total_wins = [data[e]['win'].sum() for e in encodings]
    
    # Subplot 1: Avg Reward
    axes[0, 0].bar(encodings, avg_rewards, color=[COLORS[e] for e in encodings], alpha=0.7)
    axes[0, 0].set_ylabel('Average Reward', fontsize=11)
    axes[0, 0].set_title('Average Episode Reward', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Subplot 2: Avg Score
    axes[0, 1].bar(encodings, avg_scores, color=[COLORS[e] for e in encodings], alpha=0.7)
    axes[0, 1].set_ylabel('Average Score', fontsize=11)
    axes[0, 1].set_title('Average Food Collected', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Subplot 3: Avg Steps
    axes[1, 0].bar(encodings, avg_steps, color=[COLORS[e] for e in encodings], alpha=0.7)
    axes[1, 0].set_ylabel('Average Steps', fontsize=11)
    axes[1, 0].set_title('Average Steps per Episode', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Subplot 4: Total Wins
    axes[1, 1].bar(encodings, total_wins, color=[COLORS[e] for e in encodings], alpha=0.7)
    axes[1, 1].set_ylabel('Total Wins', fontsize=11)
    axes[1, 1].set_title('Total Episodes with Perfect Score (Win)', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    fig.tight_layout()
    filepath = output_dir / "plot_7_final_performance_summary.png"
    fig.savefig(filepath, dpi=150)
    print(f"    ✓ Saved: {filepath.name}")
    
    print("\n" + "="*80)
    print("ALL PLOTS GENERATED AND SAVED SUCCESSFULLY!".center(80))
    print("="*80)
    
    print("\nDisplaying plots...")
    plt.show()

# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    try:
        print("\n" + "="*80)
        print("Q-LEARNING SNAKE - TEST & PLOT GENERATOR".center(80))
        print("="*80)
        
        # Load pre-trained Q-tables
        print("\nLoading pre-trained Q-tables...")
        q_tables = load_all_q_tables()
        
        if not q_tables:
            print("Error: No Q-tables loaded!")
            exit(1)
        
        print(f"✓ Successfully loaded {len(q_tables)} Q-tables")
        print(f"  Encodings: {', '.join(sorted(q_tables.keys()))}")
        
        # Get number of episodes from user
        try:
            num_episodes = int(input("\nNumber of episodes to test per encoding (default: 100): ") or "100")
        except ValueError:
            num_episodes = 100
        
        # Test all encodings
        print(f"\nTesting {len(q_tables)} encodings with {num_episodes} episodes each...")
        test_data = test_all_encodings(q_tables, num_episodes=num_episodes, show_progress=True)
        
        # Print summary statistics
        print("\n" + "="*80)
        print("TEST RESULTS SUMMARY".center(80))
        print("="*80)
        for encoding in sorted(test_data.keys()):
            df = test_data[encoding]
            print(f"\n{encoding.upper()}:")
            print(f"  Episodes:       {len(df)}")
            print(f"  Avg Reward:     {df['reward'].mean():7.2f} ± {df['reward'].std():6.2f}")
            print(f"  Avg Score:      {df['score'].mean():7.2f} ± {df['score'].std():6.2f}")
            print(f"  Avg Steps:      {df['steps'].mean():7.2f} ± {df['steps'].std():6.2f}")
            print(f"  Win Rate:       {df['win'].mean()*100:6.1f}%")
            print(f"  Total Wins:     {df['win'].sum():.0f}/{len(df)}")
        
        # Generate and save plots
        generate_and_save_plots(test_data)
        
        # Save test results to CSV
        TEST_CSV_DIR.mkdir(parents=True, exist_ok=True)
        print("\nSaving test results to CSV files...")
        for encoding in test_data.keys():
            filename = TEST_CSV_DIR / f"test_results_{encoding}.csv"
            test_data[encoding].to_csv(filename, index=False)
            print(f"  ✓ Saved: {filename.name}")
        
        print("\n" + "="*80)
        print("COMPLETE!".center(80))
        print("="*80 + "\n")
        
        # Display all plots
        print("Displaying all plots...")
        plt.show()
    
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
