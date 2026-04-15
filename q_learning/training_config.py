from dataclasses import dataclass

@dataclass
class AgentConfig:
    learning_rate: float = 0.15           # Increased from 0.1 for faster convergence
    gamma: float = 0.95                   # Increased from 0.9 to value future rewards more
    epsilon: float = 1.0
    epsilon_min: float = 0.02             # Increased from 0.01 to maintain exploration
    epsilon_decay: float = 0.998          # Slower decay for better exploration
    action_size: int = 4

@dataclass
class RewardConfig:
    # Core rewards - higher magnitudes for clarity
    food_reward: float = 50               # Increased from 10 (5x boost)
    death_penalty: float = -50            # Increased from -10 (now matches food reward)
    per_step_reward: float = -0.01        # Reduced from -0.1 (less time pressure)
    
    # Winning and progress rewards
    reward_for_winning: float = 10000     # Increased from 1000 (10x boost)
    length_bonus_multiplier: float = 10   # Multiplier for snake length reward
    
    # Distance-based reward shaping
    distance_bonus: float = 1.0           # Reward for moving closer to food
    distance_penalty: float = -0.5        # Penalty for moving away from food
    
    # Milestone bonuses (NEW)
    milestone_rewards: dict = None        # Set in __post_init__
    
    def __post_init__(self):
        if self.milestone_rewards is None:
            self.milestone_rewards = {
                5: 100,                   # Bonus for collecting 5 foods
                10: 200,                  # Bonus for collecting 10 foods
                15: 300,                  # Bonus for collecting 15 foods
                20: 500,                  # Bonus for collecting 20 foods
            }

@dataclass
class EnvironmentConfig:
    max_steps_per_episode: int = 5000
    render: bool = True
    fps: int = 10

@dataclass
class TrainingConfig:
    num_episodes: int = 100
    agent_config: AgentConfig = None
    reward_config: RewardConfig = None
    environment_config: EnvironmentConfig = None

    def __post_init__(self):
        if self.agent_config is None:
            self.agent_config = AgentConfig()
        if self.reward_config is None:
            self.reward_config = RewardConfig()
        if self.environment_config is None:
            self.environment_config = EnvironmentConfig()