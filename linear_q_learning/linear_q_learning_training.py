import sys
import pickle
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GAME_DIR = PROJECT_ROOT / "game"
CURRENT_DIR = Path(__file__).resolve().parent
weights_dir = CURRENT_DIR / "weights.pkl"

for path in (PROJECT_ROOT, GAME_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from linear_q_learning_agent import LinearQLearningAgent
from linear_q_learning_snake_env import LinearQLearningEnvironment


class TrainingStats:
    """Tracks training statistics across episodes."""
    
    def __init__(self):
        self.episode_rewards = []
        self.episode_scores = []
        self.episode_steps = []
        self.episode_wins = []
    
    def add_episode(self, reward, score, steps, win):
        """Record statistics for an episode."""
        self.episode_rewards.append(reward)
        self.episode_scores.append(score)
        self.episode_steps.append(steps)
        self.episode_wins.append(win)
    
    def get_averages(self):
        """Get average statistics."""
        if not self.episode_rewards:
            return 0, 0, 0, 0
        
        avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)
        avg_score = sum(self.episode_scores) / len(self.episode_scores)
        avg_steps = sum(self.episode_steps) / len(self.episode_steps)
        total_wins = sum(self.episode_wins)
        
        return avg_reward, avg_score, avg_steps, total_wins


def train(
    num_episodes=500,
    render=True,
    render_fps=100,
    learning_rate=0.1,
    gamma=0.9,
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995,
    food_reward=100,
    death_penalty=-300,
    per_step_reward=-0.1,
    reward_for_winning=1000,
    max_steps_per_episode=1000,
):
    
    env = LinearQLearningEnvironment(
        render_mode=render,
        max_steps_per_episode=max_steps_per_episode,
        food_reward=food_reward,
        death_penalty=death_penalty,
        per_step_reward=per_step_reward,
        reward_for_winning=reward_for_winning,
    )
    
    agent = LinearQLearningAgent(
        action_size=4,
        feature_size=12,
        alfa=learning_rate,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
    )
    
    stats = TrainingStats()
    
    print("=" * 80)
    print("LINEAR Q-LEARNING SNAKE GAME TRAINING")
    print("=" * 80)
    print(f"Episodes: {num_episodes}")
    print(f"Render: {render} (FPS: {render_fps})")
    print(f"Learning Rate: {learning_rate}")
    print(f"Gamma: {gamma}")
    print(f"Initial Epsilon: {epsilon}")
    print("=" * 80)
    
    try:
        for episode in range(num_episodes):
            print(f"\nEpisode {episode + 1}/{num_episodes}")
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
            )
            
            agent.decay_epsilon()
            
            if (episode + 1) % 50 == 0 or episode == 0:
                avg_reward, avg_score, avg_steps, total_wins = stats.get_averages()
        
        avg_reward, avg_score, avg_steps, total_wins = stats.get_averages()
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"Total Episodes:  {num_episodes}")
        print(f"Average Reward:  {avg_reward:.2f}")
        print(f"Average Score:   {avg_score:.2f}")
        print(f"Average Steps:   {avg_steps:.1f}")
        print(f"Total Wins:      {total_wins}")
        print(f"Final Epsilon:   {agent.epsilon:.4f}")
        print("=" * 80)
        print(agent.weights)

        with open(weights_dir, "wb") as f:
            pickle.dump(agent.weights, f)
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    
    finally:
        env.close()
    
    return agent, stats

if __name__ == "__main__":
    agent, stats = train(
        num_episodes=5000,
        render=True,
        render_fps=5000,\
        learning_rate=0.1,
        gamma=0.9,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        food_reward=100,
        death_penalty=-300,
        per_step_reward=-0.1,
        reward_for_winning=1000,
        max_steps_per_episode=3000,
    )