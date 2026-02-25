import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GAME_DIR = PROJECT_ROOT / "game"

for path in (PROJECT_ROOT, GAME_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from q_learning_agent import Q_learning_Agent
from snake_env import Snake_Env


def train(
    num_episodes=100,
    render=True,
    fps=10,
    learning_rate=0.4,
    gamma=0.6,
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995,
    food_reward=100,
    reward_for_winning=1000,
    death_penalty=-100,
    per_step_reward=-1,
    max_steps_per_episode=5000,
):
    print("=" * 80)
    print(f"Training Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Render: {render}")
    print(f"  FPS: {fps}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Gamma: {gamma}")
    print(f"  Epsilon: {epsilon}")
    print(f"  Epsilon Decay: {epsilon_decay}")
    print(f"  Food Reward: {food_reward}")
    print(f"  Death Penalty: {death_penalty}")
    print(f"  Per Step Reward: {per_step_reward}")
    print(f"  Max Steps per Episode: {max_steps_per_episode}")
    print("=" * 80)
    
    env = Snake_Env(
        render_mode=render,
        max_steps_per_episode=max_steps_per_episode,
        food_reward=food_reward,
        death_penalty=death_penalty,
        per_step_reward=per_step_reward,
        reward_for_winning=reward_for_winning,
    )
    
    agent = Q_learning_Agent(
        action_size=4,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
    )
    
    # Training loop
    episode_rewards = []
    episode_scores = []
    episode_steps = []
    episode_wins = []
    
    try:
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0.0
            episode_done = False
            
            if render:
                print(f"[Episode {episode}] Rendering initial state...")
                env.render(fps=fps)
                print(f"[Episode {episode}] Initial render complete")
            
            while not episode_done:
                action = agent.choose_action(state)
                
                next_state, reward, done, info = env.step(action)
                
                agent.learn(state, action, reward, next_state, done)
                
                episode_reward += reward
                episode_done = done
                state = next_state
                
                if render:
                    env.render(fps=fps)
                    if episode == 0 and env.step_count <= 5:
                        print(f"  -> Frame rendered")
            
            total_cells = env.board.cols * env.board.rows
            episode_win = len(env.snake.snake) >= total_cells
            episode_wins.append(int(episode_win))
            agent.decay_epsilon()
            
            # Summary for first episode
            if episode == 0:
                print(f"\n[Episode {episode + 1} Summary]")
                print(f"Episode completed with {info['steps']} steps and score {info['score']}")
                print(f"Final Snake Head Position: {env.snake.snake_head}")
            
            # Log episode results
            episode_rewards.append(episode_reward)
            episode_scores.append(info["score"])
            episode_steps.append(info["steps"])
            
            # Print progress every 10 episodes
            if (episode + 1) % 10 == 0:
                avg_reward = sum(episode_rewards[-10:]) / 10
                avg_score = sum(episode_scores[-10:]) / 10
                avg_steps = sum(episode_steps[-10:]) / 10
                
                print(f"\nEpisode {episode + 1}/{num_episodes}")
                print(f"  Epsilon: {agent.epsilon:.4f}")
                print(f"  Avg Reward (last 10): {avg_reward:.2f}")
                print(f"  Avg Score (last 10): {avg_score:.2f}")
                print(f"  Avg Steps (last 10): {avg_steps:.1f}")
                print(f"  Last Episode - Score: {info['score']}, Steps: {info['steps']}, Reward: {episode_reward:.2f}")
        
        # Final statistics
        print("\n" + "=" * 80)
        print(f"Training Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        print(f"Final Statistics:")
        print(f"  Total Episodes: {num_episodes}")
        print(f"  Average Reward: {sum(episode_rewards) / num_episodes:.2f}")
        print(f"  Average Score: {sum(episode_scores) / num_episodes:.2f}")
        print(f"  Average Steps: {sum(episode_steps) / num_episodes:.1f}")
        print(f"  Best Score: {max(episode_scores)}")
        print(f"  Total Wins: {sum(episode_wins)}")
        print(f"  Final Epsilon: {agent.epsilon:.4f}")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print(f"Episodes Completed: {len(episode_rewards)}")
        print(f"Average Score (so far): {sum(episode_scores) / len(episode_scores):.2f}")
    
    finally:
        env.close()
    
    return agent, episode_rewards, episode_scores, episode_steps

if __name__ == "__main__":
    agent, rewards, scores, steps = train(
        num_episodes=30000,
        render=True,
        fps=5000,
        learning_rate=0.1,
        gamma=0.9,  
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        food_reward=100,
        reward_for_winning=2000,
        death_penalty=-300,
        per_step_reward=-0.1,
        max_steps_per_episode=5000,
        )
