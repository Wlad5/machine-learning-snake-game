import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GAME_DIR = PROJECT_ROOT / "game"

for path in (PROJECT_ROOT, GAME_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dqn_agent import DQNAgent
from dqn_snake_env import DQNSnakeEnv


def train_dqn(
    num_episodes=500,
    render=True,
    fps=30,
    learning_rate=0.001,
    gamma=0.99,
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995,
    batch_size=32,
    memory_size=10000,
    hidden_size=256,
    max_steps_per_episode=1000,
    food_reward=10,
    death_penalty=-10,
    per_step_reward=-0.1,
    reward_for_winning=1000,
    update_frequency=100,
    save_model_path=None,
):
    """
    Train DQN agent on Snake Game with rendering support.
    
    Args:
        num_episodes: Number of training episodes
        render: Whether to render the game during training
        fps: Frames per second for rendering
        learning_rate: Learning rate for optimizer
        gamma: Discount factor
        epsilon: Initial exploration rate
        epsilon_min: Minimum exploration rate
        epsilon_decay: Decay rate per episode
        batch_size: Batch size for experience replay
        memory_size: Size of replay memory
        hidden_size: Size of hidden layers in DQN
        max_steps_per_episode: Maximum steps per episode
        food_reward: Reward for eating food
        death_penalty: Penalty for dying
        per_step_reward: Reward per step (usually negative to encourage shorter games)
        reward_for_winning: Reward for winning
        update_frequency: Update target network every N steps
        save_model_path: Path to save the trained model (uses pickle format)
    """
    
    env = DQNSnakeEnv(
        render_mode=render,
        max_steps_per_episode=max_steps_per_episode,
        food_reward=food_reward,
        death_penalty=death_penalty,
        per_step_reward=per_step_reward,
        reward_for_winning=reward_for_winning,
    )
    
    agent = DQNAgent(
        state_size=12,
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
    
    episode_rewards = []
    episode_scores = []
    episode_wins = 0
    
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
                    env.render(fps=fps)
            
            # Decay epsilon after each episode
            agent.decay_epsilon()
            
            # Calculate episode statistics
            total_cells = env.board.cols * env.board.rows
            episode_win = len(env.snake.snake) >= total_cells
            
            episode_rewards.append(episode_reward)
            episode_scores.append(info["score"])
            if episode_win:
                episode_wins += 1
            
            # Print progress every 50 episodes
            if (episode + 1) % 50 == 0:
                avg_reward = sum(episode_rewards[-50:]) / 50
                avg_score = sum(episode_scores[-50:]) / 50
                print(f"\nEpisode {episode + 1}/{num_episodes}")
                print(f"  Avg Reward (last 50):  {avg_reward:.2f}")
                print(f"  Avg Score (last 50):   {avg_score:.2f}")
                print(f"  Epsilon:               {agent.epsilon:.4f}")
                print(f"  Total Wins:            {episode_wins}")
                print(f"  Memory Size:           {len(agent.memory)}")
            else:
                print(f"Episode {episode + 1}/{num_episodes} - Reward: {episode_reward:.2f}, Score: {info['score']}, Epsilon: {agent.epsilon:.4f}")
        
        # Print final statistics
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED")
        print("=" * 80)
        print(f"Total Episodes:        {num_episodes}")
        print(f"Average Reward:        {sum(episode_rewards) / len(episode_rewards):.2f}")
        print(f"Average Score:         {sum(episode_scores) / len(episode_scores):.2f}")
        print(f"Best Score:            {max(episode_scores)}")
        print(f"Total Wins:            {episode_wins}")
        print(f"Final Epsilon:         {agent.epsilon:.4f}")
        print("=" * 80)
        # Save model if path is provided
        if save_model_path:
            agent.save(save_model_path)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        # Save model even if interrupted
        if save_model_path:
            agent.save(save_model_path)
        print("\nTraining interrupted by user.")
    finally:
        env.close()


if __name__ == "__main__":
    # Setup model save path in DQN directory
    dqn_dir = Path(__file__).resolve().parent
    model_save_path = dqn_dir / "dqn_trained_model.pkl"
    
    # Basic configuration
    train_dqn(
        num_episodes=1000,
        render=True,
        fps=1000,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        save_model_path=str(model_save_path),
        epsilon_min=0.01,
        epsilon_decay=0.995,
        batch_size=32,
        memory_size=10000,
        hidden_size=256,
        max_steps_per_episode=3000,
        update_frequency=1000,
    )