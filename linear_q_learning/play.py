import sys
import pickle
import numpy as np
from collections import defaultdict
from pathlib import Path
from linear_q_learning_agent import LinearQLearningAgent
from linear_q_learning_snake_env import LinearQLearningEnvironment
from linear_q_learning_state_encoding import LinearQLearningStateEncoding
from linear_q_learning_state_encoding_distance import LinearQLearningDistanceEncoding
from linear_q_learning_state_encoding_raycasting import LinearQLearningRayCastingEncoding
from linear_q_learning_state_encoding_localgrid import LinearQLearningLocalGridEncoding
from linear_q_learning_state_encoding_bodyawareness import LinearQLearningBodyAwarenessEncoding


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CURRENT_DIR = Path(__file__).resolve().parent
GAME_DIR = PROJECT_ROOT / "game"

# Feature size mapping for each encoding
ENCODING_FEATURE_SIZES = {
    'basic': 12,
    'distance': 13,
    'raycasting': 20,
    'localgrid': 24,
    'bodyaware': 14,
}

# Encoding class mapping
ENCODING_CLASSES = {
    'basic': LinearQLearningStateEncoding,
    'distance': LinearQLearningDistanceEncoding,
    'raycasting': LinearQLearningRayCastingEncoding,
    'localgrid': LinearQLearningLocalGridEncoding,
    'bodyaware': LinearQLearningBodyAwarenessEncoding,
}

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
    
def play_single_encoding(num_episodes=5, fps=10, encoding_name='basic'):
    """
    Play the snake game using a trained agent with a specific encoding.
    
    Args:
        num_episodes: Number of games to play
        fps: Frames per second for rendering
        encoding_name: Name of the state encoding (basic, distance, raycasting, localgrid, bodyaware)
    """
    weights_file = CURRENT_DIR / f"linear_q_weights_{encoding_name}.pkl"
    weights = load_weights(weights_file)
    if weights is None:
        return
    
    # Get feature size and encoding class for this encoding
    feature_size = ENCODING_FEATURE_SIZES.get(encoding_name, 12)
    encoding_class = ENCODING_CLASSES.get(encoding_name, LinearQLearningStateEncoding)
    state_encoder = encoding_class()
    
    env = LinearQLearningEnvironment(
        render_mode=True,
        max_steps_per_episode=1000,
        food_reward=10,
        death_penalty=-10,
        per_step_reward=-0.1,
        reward_for_winning=1000,
        state_encoder=state_encoder,
    )
    
    agent = LinearQLearningAgent(
        action_size=4,
        feature_size=feature_size,
        alfa=0.1,
        gamma=0.9,
        epsilon=0.0,  # No exploration, only exploitation
        epsilon_min=0.0,
        epsilon_decay=1.0,
    )
    
    agent.weights = weights
    
    total_wins = 0
    total_cells = env.board.cols * env.board.rows
    
    print(f"Playing with {encoding_name.upper()} encoding...")
    print(f"Feature size: {feature_size}")
    print("=" * 50)
    
    try:
        for episode in range(num_episodes):
            print(f"\nEpisode {episode + 1}/{num_episodes}")
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
                print(f"WIN! Score: {info['score']} | Total wins: {total_wins}/{episode + 1}")
            else:
                print(f"Episode Reward: {episode_reward:.2f} | Score: {info['score']}")
        
        print(f"\n{'='*50}")
        print(f"Total Wins: {total_wins}/{num_episodes}")
        print(f"{'='*50}")
    except KeyboardInterrupt:
        print("\n\nGame interrupted by user")

    finally:
        env.close() if hasattr(env, "close") else None

def play(num_episodes=5, fps=10, encoding_name='basic', play_all=True):
    """
    Play the snake game using a trained agent.
    
    Args:
        num_episodes: Number of games to play per encoding
        fps: Frames per second for rendering
        encoding_name: Name of the state encoding (basic, distance, raycasting, localgrid, bodyaware)
        play_all: If True, play with all available encodings
    """
    if play_all:
        encoding_list = ['basic', 'distance', 'raycasting', 'localgrid', 'bodyaware']
        print("\n" + "=" * 70)
        print("PLAYING WITH ALL STATE ENCODINGS")
        print("=" * 70)
        for encoding in encoding_list:
            print(f"\n\n{'#' * 70}")
            print(f"# {encoding.upper()} ENCODING")
            print(f"{'#' * 70}\n")
            play_single_encoding(num_episodes=num_episodes, fps=fps, encoding_name=encoding)
        print("\n" + "=" * 70)
        print("FINISHED PLAYING WITH ALL ENCODINGS")
        print("=" * 70)
    else:
        play_single_encoding(num_episodes=num_episodes, fps=fps, encoding_name=encoding_name)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Play Snake with a trained Linear Q-Learning agent')
    parser.add_argument('--episodes', type=int, default=20, help='Number of episodes to play per encoding')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second')
    parser.add_argument('--encoding', type=str, default='basic', 
                       choices=['basic', 'distance', 'raycasting', 'localgrid', 'bodyaware'],
                       help='State encoding to use (ignored if --all is set)')
    parser.add_argument('--all', action='store_true', default=True,
                       help='Play with all available state encodings (default: True)')
    parser.add_argument('--single', action='store_true', 
                       help='Play with only a single encoding specified by --encoding')
    
    args = parser.parse_args()
    
    # If --single is used, play only the specified encoding; otherwise play all
    play(num_episodes=args.episodes, fps=args.fps, encoding_name=args.encoding, play_all=not args.single)