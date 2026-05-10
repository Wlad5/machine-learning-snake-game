import sys
import pickle
import csv
import random
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GAME_DIR = PROJECT_ROOT / "game"
CURRENT_DIR = Path(__file__).resolve().parent

for path in (PROJECT_ROOT, GAME_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from linear_q_learning_agent import LinearQLearningAgent
from linear_q_learning_snake_env import LinearQLearningEnvironment
from linear_q_learning_state_encoding import LinearQLearningStateEncoding
from linear_q_learning_state_encoding_distance import LinearQLearningDistanceEncoding
from linear_q_learning_state_encoding_raycasting import LinearQLearningRayCastingEncoding
from linear_q_learning_state_encoding_localgrid import LinearQLearningLocalGridEncoding
from linear_q_learning_state_encoding_bodyawareness import LinearQLearningBodyAwarenessEncoding


# Feature size mapping for each encoding
ENCODING_FEATURE_SIZES = {
    'basic': 16,
    'distance': 17,
    'raycasting': 23,
    'localgrid': 61,
    'bodyaware': 15,
}


class TrainingStats:
    """Tracks training statistics across episodes."""
    
    def __init__(self):
        self.episode_rewards = []
        self.episode_scores = []
        self.episode_steps = []
        self.episode_wins = []
        self.episode_epsilons = []
    
    def add_episode(self, reward, score, steps, win, epsilon):
        """Record statistics for an episode."""
        self.episode_rewards.append(reward)
        self.episode_scores.append(score)
        self.episode_steps.append(steps)
        self.episode_wins.append(win)
        self.episode_epsilons.append(epsilon)
    
    def get_averages(self):
        """Get average statistics."""
        if not self.episode_rewards:
            return 0, 0, 0, 0
        
        avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)
        avg_score = sum(self.episode_scores) / len(self.episode_scores)
        avg_steps = sum(self.episode_steps) / len(self.episode_steps)
        total_wins = sum(self.episode_wins)
        
        return avg_reward, avg_score, avg_steps, total_wins
    
    def save_to_csv(self, filename):
        """Save statistics to CSV file."""
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


def train(
    num_episodes=10000,
    render=False,
    render_fps=100000,
    learning_rate=0.1,
    gamma=0.9,
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995,
    food_reward=50,
    death_penalty=-50,
    per_step_reward=-0.01,
    reward_for_winning=10000,
    distance_bonus=1.0,
    distance_penalty=-0.5,
    length_bonus_multiplier=10,
    milestone_rewards=None,
    max_steps_per_episode=1000,
    encoding_name='basic',
    state_encoder=None,
    domain_randomization_grids=None,
):
    """
    Train a Linear Q-Learning agent with a specific state encoding.
    
    Args:
        num_episodes: Number of training episodes
        render: Whether to render the game
        render_fps: Frames per second for rendering
        learning_rate: Learning rate (alpha)
        gamma: Discount factor
        epsilon: Initial exploration rate
        epsilon_min: Minimum exploration rate
        epsilon_decay: Exploration decay rate
        food_reward: Reward for eating food
        death_penalty: Penalty for dying
        per_step_reward: Reward per step (usually negative)
        reward_for_winning: Reward for winning
        distance_bonus: Reward for reducing distance to food
        distance_penalty: Penalty for increasing distance to food
        length_bonus_multiplier: Multiplier for snake length growth bonus
        milestone_rewards: Score milestone bonus map
        max_steps_per_episode: Max steps per episode
        encoding_name: Name of the encoding (basic, distance, raycasting, localgrid, bodyaware)
        state_encoder: State encoder object (if None, creates one based on encoding_name)
    
    Returns:
        tuple: (agent, stats)
    """
    
    # Determine feature size based on encoding
    feature_size = ENCODING_FEATURE_SIZES.get(encoding_name, 12)
    
    # Create state encoder if not provided
    if state_encoder is None:
        encoding_map = {
            'basic': LinearQLearningStateEncoding,
            'distance': LinearQLearningDistanceEncoding,
            'raycasting': LinearQLearningRayCastingEncoding,
            'localgrid': LinearQLearningLocalGridEncoding,
            'bodyaware': LinearQLearningBodyAwarenessEncoding,
        }
        state_encoder = encoding_map.get(encoding_name, LinearQLearningStateEncoding)()
    
    env = LinearQLearningEnvironment(
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
    
    agent = LinearQLearningAgent(
        action_size=4,
        feature_size=feature_size,
        alfa=learning_rate,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
    )
    
    stats = TrainingStats()
    training_csv_dir = CURRENT_DIR / "training_csv"
    training_csv_dir.mkdir(parents=True, exist_ok=True)
    stats_file = training_csv_dir / f"linear_q_training_stats_{encoding_name}.csv"
    
    try:
        for episode in range(num_episodes):
            if domain_randomization_grids is not None:
                g = random.choice(domain_randomization_grids)
                state = env.reset(grid_cols=g, grid_rows=g)
            else:
                state = env.reset()
            episode_reward = 0.0
            episode_done = False
            
            if render:
                env.render(fps=render_fps)
            
            while not episode_done:
                action = agent.choose_action(state)
                next_state, reward, done, info = env.step(action)
                
                agent.learn(state, action, reward, next_state, done)
                
                episode_reward += reward
                episode_done = done
                state = next_state
                
                if render:
                    env.render(fps=render_fps)
            
            total_cells = env.board.cols * env.board.rows
            episode_win = len(env.snake.snake) >= total_cells
            
            stats.add_episode(
                reward=episode_reward,
                score=info["score"],
                steps=info["steps"],
                win=episode_win,
                epsilon=agent.epsilon,
            )
            
            agent.decay_epsilon()
            
            if (episode + 1) % 100 == 0:
                stats.save_to_csv(str(stats_file))
                avg_reward, avg_score, avg_steps, total_wins = stats.get_averages()
                print(f"Episode {episode + 1}/{num_episodes} - Avg Reward: {avg_reward:.2f}, Avg Score: {avg_score:.2f}, Wins: {total_wins}, Epsilon: {agent.epsilon:.4f}")
        
        # Final CSV save
        stats.save_to_csv(str(stats_file))

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
        
        # Save weights
        models_dir = CURRENT_DIR / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        weights_file = models_dir / f"linear_q_weights_{encoding_name}.pkl"
        
        with open(weights_file, "wb") as f:
            pickle.dump(agent.weights, f)
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    
    finally:
        env.close()
    
    return agent, stats


if __name__ == "__main__":
    # Define all state encodings
    encodings = {
        'basic': LinearQLearningStateEncoding(),
        'distance': LinearQLearningDistanceEncoding(),
        'raycasting': LinearQLearningRayCastingEncoding(),
        'localgrid': LinearQLearningLocalGridEncoding(),
        'bodyaware': LinearQLearningBodyAwarenessEncoding(),
    }
    
    # Training configuration
    num_episodes = 50000
    render = False
    render_fps = 100000
    learning_rate = 0.1
    gamma = 0.9
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.99985
    food_reward = 100
    death_penalty = -300
    per_step_reward = -0.05
    reward_for_winning = 2000
    distance_bonus = 1.0
    distance_penalty = -0.5
    length_bonus_multiplier = 10
    milestone_rewards = {5: 100, 10: 200, 15: 300, 20: 500}
    max_steps_per_episode = 1500
    domain_randomization_grids = [3, 4, 5, 6]
    
    # Train with each encoding
    training_results = {}
    for encoding_name, encoding in encodings.items():
        print(f"\n{'='*80}")
        print(f"Training with {encoding_name.upper()} encoding...")
        print(f"{'='*80}")
        agent, stats = train(
            num_episodes=num_episodes,
            render=render,
            render_fps=render_fps,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            food_reward=food_reward,
            death_penalty=death_penalty,
            per_step_reward=per_step_reward,
            reward_for_winning=reward_for_winning,
            distance_bonus=distance_bonus,
            distance_penalty=distance_penalty,
            length_bonus_multiplier=length_bonus_multiplier,
            milestone_rewards=milestone_rewards,
            max_steps_per_episode=max_steps_per_episode,
            encoding_name=encoding_name,
            state_encoder=encoding,
            domain_randomization_grids=domain_randomization_grids,
        )
        avg_reward, avg_score, avg_steps, total_wins = stats.get_averages()
        training_results[encoding_name] = total_wins
    
    print(f"\n\n{'='*80}")
    print("TRAINING SUMMARY - WINS BY STATE ENCODING")
    print(f"{'='*80}")
    for encoding_name, wins in training_results.items():
        print(f"{encoding_name.upper():15} : {wins} wins")
    print(f"{'='*80}")