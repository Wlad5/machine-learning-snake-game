import sys
import csv
import random
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
    'basic': 16,      # 4 direction + 2 norm food offsets + 4 binary food dir + 4 danger + 2 tail offset
    'distance': 17,   # 4 direction + 4 food dir + 4 wall dist + food dist + 4 danger flags
    'raycasting': 23, # 4 direction + 8 rays × 2 + 2 tail offset + 1 norm food dist
    'localgrid': 61,  # 4 direction + 4 food dir + 48 grid (24 cells × 2) + 2 norm food offset + 1 food dist + 2 tail offset
    'bodyaware': 15,  # 4 direction + 4 food dir + 4 danger + length + 2 tail offset
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
    food_reward=50,
    death_penalty=-50,
    per_step_reward=-0.01,
    reward_for_winning=10000,
    distance_bonus=1.0,
    distance_penalty=-0.5,
    length_bonus_multiplier=10,
    milestone_rewards=None,
    max_steps_per_episode=1000,
    update_frequency=100,
    encoding_name='basic',
    state_encoder=None,
    domain_randomization_grids=None,
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
        distance_bonus=distance_bonus,
        distance_penalty=distance_penalty,
        length_bonus_multiplier=length_bonus_multiplier,
        milestone_rewards=milestone_rewards,
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
    
    print(f"Training on: {agent.device} ({encoding_name} encoding)")

    stats = DQNTrainingStats()
    training_csv_dir = CURRENT_DIR / "training_csv"
    training_csv_dir.mkdir(parents=True, exist_ok=True)
    history_file = training_csv_dir / f"dqn_training_stats_{encoding_name}.csv"
    
    try:
        for episode in range(num_episodes):
            if domain_randomization_grids is not None:
                g = random.choice(domain_randomization_grids)
                state = env.reset(grid_cols=g, grid_rows=g)
            else:
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

                # Update target network and train every 4 steps
                agent.step_count += 1
                if agent.step_count % 4 == 0:
                    agent.replay()
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
            # Save CSV and print every 100 episodes
            if (episode + 1) % 100 == 0:
                stats.save_to_csv(str(history_file))
                avg_reward, avg_score, avg_steps, total_wins = stats.get_averages()
                print(f"Episode {episode + 1}/{num_episodes} - Avg Reward: {avg_reward:.2f}, Avg Score: {avg_score:.2f}, Wins: {total_wins}, Epsilon: {agent.epsilon:.4f}")
        
        # Final CSV save
        stats.save_to_csv(str(history_file))

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
    'num_episodes': 150000,          # more episodes for harder task
    'render': False,
    'render_fps': 100000,
    'learning_rate': 0.0005,         # lower LR for more stable learning on complex grids
    'gamma': 0.995,                  # higher discount — long-horizon planning matters more on big grids
    'epsilon': 1.0,
    'epsilon_min': 0.05,             # higher floor — keep some exploration on unseen large grids
    'epsilon_decay': 0.999955,       # reaches ~0.05 at ~65k episodes out of 150k
    'batch_size': 512,
    'memory_size': 200000,           # larger buffer for diverse experiences across grid sizes
    'hidden_size': 512,              # larger network for more complex spatial representations
    'food_reward': 100,
    'death_penalty': -300,
    'per_step_reward': -0.02,        # slightly higher step penalty to discourage circling on big grids
    'reward_for_winning': 5000,
    'distance_bonus': 2.0,           # stronger guidance on larger grids where food is farther away
    'distance_penalty': -1.0,
    'length_bonus_multiplier': 10,
    'milestone_rewards': {5: 100, 10: 300, 20: 600, 30: 1000, 50: 2000},  # extended milestones for big grids
    'max_steps_per_episode': 5000,   # 10x10=100 cells; needs much more headroom (was 1500 for 5x5=25 cells)
    'update_frequency': 10000,
    'domain_randomization_grids': [3, 4, 5, 6, 7, 8, 10],  # train on the target sizes, drop 3x3
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
