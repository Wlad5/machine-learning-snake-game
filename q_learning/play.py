import sys
import pickle
import numpy as np
from pathlib import Path
from snake_env import Snake_Env
from collections import defaultdict
from q_learning_agent import Q_learning_Agent

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CURRENT_DIR = Path(__file__).resolve().parent
GAME_DIR = PROJECT_ROOT / "game"
q_table_file = CURRENT_DIR / "q_learning_q_table.pkl"

for path in (PROJECT_ROOT, GAME_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def load_q_table(filepath):
    try:
        with open(filepath, 'rb') as f:
            q_table_dict = pickle.load(f)
        q_table = defaultdict(lambda: np.zeros(4, dtype=np.float32))
        q_table.update(q_table_dict)
        print(f"Learned states: {len(q_table)}")
        return q_table
    except FileNotFoundError:
        print(f"Error: Q-table file not found at {filepath}")
        print("Train the agent first by running train.py")
        return None

def play(num_episodes=5, fps=10):
    q_table = load_q_table(q_table_file)
    if q_table is None:
        return
    
    env = Snake_Env(
        render_mode=True,
        max_steps_per_episode=1000,
        food_reward=10,
        death_penalty=-10,
        per_step_reward=-0.1,
        reward_for_winning=1000,
    )
    
    agent = Q_learning_Agent(
        action_size=4,
        learning_rate=0.1,
        gamma=0.9,
        epsilon=0.0,  # No exploration, only exploitation
        epsilon_min=0.0,
        epsilon_decay=1.0,
    )
    
    agent.q_table = q_table
    
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
                episode_done = done
                state = next_state
                
                env.render(fps=fps)
            
            print(f"  Score: {info['score']}, Reward: {episode_reward:.2f}, Steps: {info['steps']}")
        
        print("\nGame finished!")
    
    except KeyboardInterrupt:
        print("\nGame interrupted by user")
    
    finally:
        env.close() if hasattr(env, 'close') else None

if __name__ == "__main__":
    play(num_episodes=20, fps=60)