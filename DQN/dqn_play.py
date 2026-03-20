import sys
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GAME_DIR = PROJECT_ROOT / "game"
dqn_dir = Path(__file__).resolve().parent
default_model_path = dqn_dir / "dqn_trained_model.pkl"

for path in (PROJECT_ROOT, GAME_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dqn_agent import DQNAgent
from dqn_snake_env import DQNSnakeEnv


def play_dqn(
    model_path="dqn_trained_model.pkl",
    num_episodes=5,
    render=True,
    fps=30,
    max_steps_per_episode=3000,
):
    try:
        agent = DQNAgent.load(model_path, epsilon=0.0)  # Set epsilon to 0 for greedy play
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        print("Please train the model first using dqn_train.py")
        return
    
    # Create environment
    env = DQNSnakeEnv(
        render_mode=render,
        max_steps_per_episode=max_steps_per_episode,
    )
    
    episode_scores = []
    episode_rewards = []
    episode_wins = 0
    
    try:
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0.0
            episode_done = False
            
            while not episode_done:
                action = agent.choose_action(state)
                
                next_state, reward, done, info = env.step(action)
                
                episode_reward += reward
                episode_done = done
                state = next_state
                
                if render:
                    env.render(fps=fps)
            
            total_cells = env.board.cols * env.board.rows
            episode_win = len(env.snake.snake) >= total_cells
            
            episode_scores.append(info["score"])
            episode_rewards.append(episode_reward)
            if episode_win:
                episode_wins += 1
            
            print(f"Episode {episode + 1}/{num_episodes} - Score: {info['score']}, Reward: {episode_reward:.2f}")
        
        print("\n" + "=" * 80)
        print("GAME PLAY COMPLETED")
        print("=" * 80)
        print(f"Total Episodes:        {num_episodes}")
        print(f"Average Score:         {sum(episode_scores) / len(episode_scores):.2f}")
        print(f"Best Score:            {max(episode_scores)}")
        print(f"Worst Score:           {min(episode_scores)}")
        print(f"Total Wins:            {episode_wins}")
        print(f"Average Reward:        {sum(episode_rewards) / len(episode_rewards):.2f}")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\nGame interrupted by user.")
    finally:
        env.close()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Play Snake game with trained DQN model")
    parser.add_argument(
        "--model",
        type=str,
        default=str(default_model_path),
        help=f"Path to the trained model (default: {default_model_path})",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of episodes to play (default: 5)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for rendering (default: 30)",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rendering",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=3000,
        help="Maximum steps per episode (default: 3000)",
    )
    
    args = parser.parse_args()
    
    play_dqn(
        model_path=args.model,
        num_episodes=args.episodes,
        render=not args.no_render,
        fps=args.fps,
        max_steps_per_episode=args.max_steps,
    )
