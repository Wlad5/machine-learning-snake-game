import sys
from pathlib import Path
from training_config import TrainingConfig, AgentConfig, RewardConfig, EnvironmentConfig
from training_stats import TrainingStatistics, EpisodeResult

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GAME_DIR = PROJECT_ROOT / "game"

for path in (PROJECT_ROOT, GAME_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from q_learning_agent import Q_learning_Agent
from snake_env import Snake_Env


def train(config: TrainingConfig):
    num_episodes                = config.num_episodes

    render                      = config.environment_config.render
    fps                         = config.environment_config.fps

    action_size                 = config.agent_config.action_size
    learning_rate               = config.agent_config.learning_rate
    gamma                       = config.agent_config.gamma
    epsilon                     = config.agent_config.epsilon
    epsilon_min                 = config.agent_config.epsilon_min
    epsilon_decay               = config.agent_config.epsilon_decay
    
    food_reward                 = config.reward_config.food_reward
    reward_for_winning          = config.reward_config.reward_for_winning
    death_penalty               = config.reward_config.death_penalty
    per_step_reward             = config.reward_config.per_step_reward
    
    max_steps_per_episode       = config.environment_config.max_steps_per_episode
    
    env = Snake_Env(
        render_mode             = render,
        max_steps_per_episode   = max_steps_per_episode,
        food_reward             = food_reward,
        death_penalty           = death_penalty,
        per_step_reward         = per_step_reward,
        reward_for_winning      = reward_for_winning,
    )
    
    agent = Q_learning_Agent(
        action_size             = action_size,
        learning_rate           = learning_rate,
        gamma                   = gamma,
        epsilon                 = epsilon,
        epsilon_min             = epsilon_min,
        epsilon_decay           = epsilon_decay,
    )
    
    stats = TrainingStatistics()
    
    try:
        for episode in range(num_episodes):
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            state = env.reset()
            episode_reward = 0.0
            episode_done = False
            
            if render:
                env.render(fps=fps)
            
            while not episode_done:
                action = agent.choose_action(state)
                
                next_state, reward, done, info = env.step(action)
                
                agent.learn(state, action, reward, next_state, done)
                
                episode_reward += reward
                episode_done = done
                state = next_state
                
                if render:
                    env.render(fps=fps)
            
            total_cells = env.board.cols * env.board.rows
            episode_win = len(env.snake.snake) >= total_cells
            agent.decay_epsilon()
            result = EpisodeResult(
                reward=episode_reward,
                score=info["score"],
                steps=info["steps"],
                win=episode_win
            )
            stats.add_episode(result)
                    
        final_stats = stats.get_final_stats()
        print(f"  Total Episodes:   {final_stats['total_episodes']}")
        print(f"  Average Reward:   {final_stats['avg_reward']:.2f}")
        print(f"  Average Score:    {final_stats['avg_score']:.2f}")
        print(f"  Average Steps:    {final_stats['avg_steps']:.1f}")
        print(f"  Best Score:       {final_stats['best_score']}")
        print(f"  Total Wins:       {final_stats['total_wins']}")
        print(f"  Final Epsilon:    {agent.epsilon:.4f}")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    
    finally:
        env.close()
    
    return agent, stats

if __name__ == "__main__":
    config = TrainingConfig(
        num_episodes=2000,
        agent_config = AgentConfig(
            learning_rate   = 0.1,
            gamma           = 0.9,
            epsilon         = 1.0,
            epsilon_min     = 0.01,
            epsilon_decay   = 0.995,
            action_size     = 4,
        ),
        reward_config = RewardConfig(
            food_reward         = 100,
            reward_for_winning  = 2000,
            death_penalty       = -300,
            per_step_reward     = -0.1,
        ),
        environment_config = EnvironmentConfig(
            render                  =True,
            fps                     =5000,
            max_steps_per_episode   =5000,
            )
    )
    agent, stats = train(config)
