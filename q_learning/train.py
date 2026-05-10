import sys
import random
from pathlib import Path
from training_config import TrainingConfig, AgentConfig, RewardConfig, EnvironmentConfig
from training_stats import TrainingStatistics, EpisodeResult
import pickle

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GAME_DIR = PROJECT_ROOT / "game"
CURRENT_DIR = Path(__file__).resolve().parent

for path in (PROJECT_ROOT, GAME_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from q_learning_agent import Q_learning_Agent
from snake_env import Snake_Env
from state_encoding import State_Encoding
from state_encoding_distance import DistanceStateEncoding
from state_encoding_raycasting import RayCastingStateEncoding
from state_encoding_localgrid import LocalGridStateEncoding
from state_encoding_bodyawareness import BodyAwarenessStateEncoding


def train(config: TrainingConfig, encoding_name: str, state_encoder, domain_randomization_grids=None):
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
    length_bonus_multiplier     = config.reward_config.length_bonus_multiplier
    milestone_rewards           = config.reward_config.milestone_rewards
    stagnation_scale            = config.reward_config.stagnation_scale
    revisit_penalty             = config.reward_config.revisit_penalty
    distance_shaping_scale      = config.reward_config.distance_shaping_scale
    scale_death_by_score        = config.reward_config.scale_death_by_score
    use_dynamic_step_budget     = config.reward_config.use_dynamic_step_budget
    
    max_steps_per_episode       = config.environment_config.max_steps_per_episode
    
    training_csv_dir = CURRENT_DIR / "training_csv"
    training_csv_dir.mkdir(parents=True, exist_ok=True)
    history_file = training_csv_dir / f"training_stats_{encoding_name}.csv"
    models_dir = CURRENT_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    q_table_file = models_dir / f"q_learning_q_table_{encoding_name}.pkl"
    
    env = Snake_Env(
        render_mode             = render,
        max_steps_per_episode   = max_steps_per_episode,
        food_reward             = food_reward,
        death_penalty           = death_penalty,
        per_step_reward         = per_step_reward,
        reward_for_winning      = reward_for_winning,
        length_bonus_multiplier = length_bonus_multiplier,
        milestone_rewards       = milestone_rewards,
        stagnation_scale        = stagnation_scale,
        revisit_penalty         = revisit_penalty,
        distance_shaping_scale  = distance_shaping_scale,
        scale_death_by_score    = scale_death_by_score,
        use_dynamic_step_budget = use_dynamic_step_budget,
        state_encoder           = state_encoder,
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
            if domain_randomization_grids is not None:
                g = random.choice(domain_randomization_grids)
                state = env.reset(grid_cols=g, grid_rows=g)
            else:
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
                win=episode_win,
                epsilon=agent.epsilon,
            )
            stats.add_episode(result, episode)
            if (episode + 1) % 100 == 0:
                stats.save_to_csv(str(history_file))
                final_stats = stats.get_final_stats()
                print(f"Episode {episode + 1}/{num_episodes} - Avg Reward: {final_stats['avg_reward']:.2f}, Avg Score: {final_stats['avg_score']:.2f}, Wins: {final_stats['total_wins']}, Epsilon: {agent.epsilon:.4f}")

        # Final CSV save
        stats.save_to_csv(str(history_file))

        # Print final statistics
        final_stats = stats.get_final_stats()
        print(f"\n{'='*80}")
        print(f">>> {encoding_name.upper()} TRAINING COMPLETE <<<")
        print(f"{'='*80}")
        print(f"Total Episodes:        {final_stats['total_episodes']}")
        print(f"Average Reward:        {final_stats['avg_reward']:.2f}")
        print(f"Average Score:         {final_stats['avg_score']:.2f}")
        print(f"Average Steps:         {final_stats['avg_steps']:.1f}")
        print(f"Total Wins:            {final_stats['total_wins']}")
        print(f"Final Epsilon:         {agent.epsilon:.4f}")
        print(f"{'='*80}")
        with open(q_table_file, "wb") as f:
            pickle.dump(dict(agent.q_table), f)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    
    finally:
        env.close()
    
    return agent, stats

if __name__ == "__main__":
    config = TrainingConfig(
        num_episodes=50000,
        agent_config = AgentConfig(
            learning_rate   = 0.1,
            gamma           = 0.9,
            epsilon         = 1.0,
            epsilon_min     = 0.01,
            epsilon_decay   = 0.99985,
            action_size     = 4,
        ),
        reward_config = RewardConfig(
            food_reward             = 100,
            reward_for_winning      = 2000,
            death_penalty           = -300,
            per_step_reward         = -0.05,
            stagnation_scale        = 10.0,
            revisit_penalty         = 2.0,
            distance_shaping_scale  = 0.3,
            scale_death_by_score    = True,
            use_dynamic_step_budget = True,
        ),
        environment_config = EnvironmentConfig(
            render                  =False,
            fps                     =100000,
            max_steps_per_episode   =1500,
            )
    )
    domain_randomization_grids = [3, 4, 5, 6, 7]
    
    # Define all state encodings
    encodings = {
        'basic': State_Encoding(),
        'distance': DistanceStateEncoding(),
        'raycasting': RayCastingStateEncoding(),
        'localgrid': LocalGridStateEncoding(),
        'bodyaware': BodyAwarenessStateEncoding(),
    }
    
    # Train with each encoding
    training_wins = {}
    for encoding_name, encoding in encodings.items():
        print(f"\n{'='*80}")
        print(f"Training with {encoding_name.upper()} encoding...")
        print(f"{'='*80}")
        agent, stats = train(config, encoding_name, encoding, domain_randomization_grids=domain_randomization_grids)
        final_stats = stats.get_final_stats()
        training_wins[encoding_name] = final_stats['total_wins']
    
    print(f"\n\n{'='*80}")
    print("TRAINING SUMMARY - WINS BY STATE ENCODING")
    print(f"{'='*80}")
    for encoding_name, wins in training_wins.items():
        print(f"{encoding_name.upper():15} : {wins} wins")
    print(f"{'='*80}")