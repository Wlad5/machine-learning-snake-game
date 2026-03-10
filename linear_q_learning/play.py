import sys
import pickle
import numpy as np
from collections import defaultdict
from pathlib import Path
from linear_q_learning_agent import LinearQLearningAgent
from linear_q_learning_snake_env import LinearQLearningEnvironment


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CURRENT_DIR = Path(__file__).resolve().parent
GAME_DIR = PROJECT_ROOT / "game"
weights_dir = CURRENT_DIR / "weights.pkl"

for path in (PROJECT_ROOT, GAME_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

def load_weights(filepath):
    try:
        with open(filepath, "rb") as f:
            weights = pickle.load(f)
        return weights
    except FileNotFoundError:
        print(f"Error: Weights file not found at {filepath}")
        print("Train the agent first by running linear_q_learning_training.py")
        return None
    
def play(num_episodes=5, fps=10):
    weights = load_weights(weights_dir)
    if weights is None:
        return
    
    env = LinearQLearningEnvironment(
        render_mode=True,
        max_steps_per_episode=1000,
        food_reward=10,
        death_penalty=-10,
        per_step_reward=-0.1,
        reward_for_winning=1000,
    )
    
    agent = LinearQLearningAgent(
        action_size=4,
        feature_size=12,
        alfa=0.1,
        gamma=0.9,
        epsilon=0.0,  # No exploration, only exploitation
        epsilon_min=0.0,
        epsilon_decay=1.0,
    )
    
    agent.weights = weights
    
    total_wins = 0
    total_cells = env.board.cols * env.board.rows
    
    try:
        for episode in range(num_episodes):
            print(f"Episode {episode + 1}/{num_episodes}")
            state = env.reset()
            episode_reward = 0.0
            episode_done = False
        
            while not episode_done:
                action = agent.choose_action(state)
                next_state, reward, done, info = env.step(action)
                episode_reward += reward
                state = next_state
                episode_done = done
                env.render(fps=fps)
            
            # Check if episode was a win (filled entire board)
            if info["score"] >= total_cells:
                total_wins += 1
                print(f"Total wins: {total_wins}/{episode + 1}")
            else:
                print(f"Episode Reward: {episode_reward:.2f} - Score: {info['score']}")
        
        print(f"\n{'='*50}")
        print(f"Total Wins: {total_wins}/{num_episodes}")
        print(f"{'='*50}")
    except KeyboardInterrupt:
        print("\n\nGame interrupted by user")

    finally:
        env.close() if hasattr(env, "close") else None

if __name__ == "__main__":
    play(num_episodes=30, fps=10)