import sys
import csv
import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GAME_DIR = PROJECT_ROOT / "game"
CURRENT_DIR = Path(__file__).resolve().parent

for path in (PROJECT_ROOT, GAME_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dqn_agent import DQNAgent
from dqn_snake_env import DQNSnakeEnv
from dqn_state_encoding import DQNStateEncoding
from dqn_state_encoding_distance import DQNDistanceStateEncoding
from dqn_state_encoding_raycasting import DQNRayCastingStateEncoding
from dqn_state_encoding_localgrid import DQNLocalGridStateEncoding
from dqn_state_encoding_bodyawareness import DQNBodyAwarenessStateEncoding


# Feature size mapping for each encoding
ENCODING_FEATURE_SIZES = {
    'basic': 12,
    'distance': 13,
    'raycasting': 20,
    'localgrid': 24,
    'bodyaware': 14,
}


class DQNTrainingStats:
    def __init__(self):
        self.episode_rewards = []
        self.episode_scores = []
        self.episode_steps = []
        self.episode_wins = []
        self.episode_epsilons = []
    
    def add_episode(self, reward, score, steps, win, epsilon):
        self.episode_rewards.append(reward)
        self.episode_scores.append(score)
        self.episode_steps.append(steps)
        self.episode_wins.append(win)
        self.episode_epsilons.append(epsilon)
    
    def get_averages(self):
        if not self.episode_rewards:
            return 0, 0, 0, 0
        
        avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)
        avg_score = sum(self.episode_scores) / len(self.episode_scores)
        avg_steps = sum(self.episode_steps) / len(self.episode_steps)
        total_wins = sum(self.episode_wins)
        
        return avg_reward, avg_score, avg_steps, total_wins
    
    def save_to_csv(self, filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'reward', 'score', 'steps', 'win', 'epsilon'])
            for i, (r, s, st, w, e) in enumerate(zip(
                self.episode_rewards, 
                self.episode_scores, 
                self.episode_steps, 
                self.episode_wins,
                self.episode_epsilons
            )):
                writer.writerow([i + 1, r, s, st, int(w), e])


def train_dqn(
    num_episodes=1000,
    render=False,
    render_fps=100000,
    learning_rate=0.001,
    gamma=0.99,
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995,
    batch_size=32,
    memory_size=10000,
    hidden_size=256,
    food_reward=10,
    death_penalty=-10,
    per_step_reward=-0.1,
    reward_for_winning=1000,
    max_steps_per_episode=1000,
    update_frequency=100,
    encoding_name='basic',
    state_encoder=None,
):
    # Determine feature size based on encoding
    feature_size = ENCODING_FEATURE_SIZES.get(encoding_name, 12)
    
    # Create state encoder if not provided
    if state_encoder is None:
        encoding_map = {
            'basic': DQNStateEncoding,
            'distance': DQNDistanceStateEncoding,
            'raycasting': DQNRayCastingStateEncoding,
            'localgrid': DQNLocalGridStateEncoding,
            'bodyaware': DQNBodyAwarenessStateEncoding,
        }
        state_encoder = encoding_map.get(encoding_name, DQNStateEncoding)()
    
    # Create environment with state encoder
    env = DQNSnakeEnv(
        render_mode=render,
        max_steps_per_episode=max_steps_per_episode,
        food_reward=food_reward,
        death_penalty=death_penalty,
        per_step_reward=per_step_reward,
        reward_for_winning=reward_for_winning,
        state_encoder=state_encoder,
    )
    
    # Create agent with appropriate feature size
    agent = DQNAgent(
        state_size=feature_size,
        action_size=4,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size,
        memory_size=memory_size,
        hidden_size=hidden_size,
        update_frequency=update_frequency,
    )
    
    stats = DQNTrainingStats()
    training_csv_dir = CURRENT_DIR / "training_csv"
    training_csv_dir.mkdir(parents=True, exist_ok=True)
    history_file = training_csv_dir / f"dqn_training_stats_{encoding_name}.csv"
    
    try:
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0.0
            episode_done = False
            
            while not episode_done:
                # Choose action using epsilon-greedy policy
                action = agent.choose_action(state)
                
                # Take action in environment
                next_state, reward, done, info = env.step(action)
                
                # Store experience in replay memory
                agent.remember(state, action, reward, next_state, done)
                
                # Train on a batch from replay memory
                agent.replay()
                
                # Update target network periodically
                agent.step_count += 1
                if agent.step_count % update_frequency == 0:
                    agent.update_target_network()
                
                episode_reward += reward
                episode_done = done
                state = next_state
                
                # Render if enabled
                if render:
                    env.render(fps=render_fps)
            
            # Decay epsilon after each episode
            agent.decay_epsilon()
            
            # Calculate episode statistics
            total_cells = env.board.cols * env.board.rows
            episode_win = len(env.snake.snake) >= total_cells
            
            stats.add_episode(
                reward=episode_reward,
                score=info["score"],
                steps=info["steps"],
                win=episode_win,
                epsilon=agent.epsilon,
            )
            stats.save_to_csv(str(history_file))
            
            # Print progress
            if (episode + 1) % 100 == 0:
                avg_reward, avg_score, avg_steps, total_wins = stats.get_averages()
                print(f"Episode {episode + 1}/{num_episodes} - Avg Reward: {avg_reward:.2f}, Avg Score: {avg_score:.2f}, Wins: {total_wins}, Epsilon: {agent.epsilon:.4f}")
        
        # Print final statistics
        avg_reward, avg_score, avg_steps, total_wins = stats.get_averages()
        print(f"\n{'='*80}")
        print(f">>> {encoding_name.upper()} TRAINING COMPLETE <<<")
        print(f"{'='*80}")
        print(f"Total Episodes:        {num_episodes}")
        print(f"Average Reward:        {avg_reward:.2f}")
        print(f"Average Score:         {avg_score:.2f}")
        print(f"Average Steps:         {avg_steps:.1f}")
        print(f"Total Wins:            {total_wins}")
        print(f"Final Epsilon:         {agent.epsilon:.4f}")
        print(f"{'='*80}")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    
    finally:
        env.close()
    
    return agent, stats


if __name__ == "__main__":
    # Configuration
    config = {
        'num_episodes': 3000,
        'render': False,
        'render_fps': 100000,
        'learning_rate': 0.001,
        'gamma': 0.99,
        'epsilon': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995,
        'batch_size': 32,
        'memory_size': 10000,
        'hidden_size': 256,
        'food_reward': 100,
        'death_penalty': -1000,
        'per_step_reward': -0.1,
        'reward_for_winning': 10000,
        'max_steps_per_episode': 3000,
        'update_frequency': 10000,
    }
    
    # Define all state encodings
    encodings = {
        'basic': DQNStateEncoding(),
        'distance': DQNDistanceStateEncoding(),
        'raycasting': DQNRayCastingStateEncoding(),
        'localgrid': DQNLocalGridStateEncoding(),
        'bodyaware': DQNBodyAwarenessStateEncoding(),
    }
    
    # Train with each encoding
    training_wins = {}
    print(f"\n{'='*80}")
    print("DQN TRAINING - COMPARING STATE ENCODINGS")
    print(f"{'='*80}\n")
    
    for encoding_name, encoding in encodings.items():
        print(f"\n{'='*80}")
        print(f"Training with {encoding_name.upper()} encoding...")
        print(f"{'='*80}")
        agent, stats = train_dqn(
            state_encoder=encoding,
            encoding_name=encoding_name,
            **config
        )
        _, _, _, total_wins = stats.get_averages()
        training_wins[encoding_name] = int(total_wins)
        
        # Save model per encoding
        models_dir = CURRENT_DIR / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / f"dqn_trained_model_{encoding_name}.pkl"
        agent.save(str(model_path))
    
    print(f"\n\n{'='*80}")
    print("TRAINING SUMMARY - WINS BY STATE ENCODING")
    print(f"{'='*80}")
    for encoding_name, wins in training_wins.items():
        print(f"{encoding_name.upper():15} : {wins} wins")
    print(f"{'='*80}")
