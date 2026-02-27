from dataclasses import dataclass

@dataclass
class AgentConfig:
    """Q-learning agent hyperparameters"""
    learning_rate: float = 0.1
    gamma: float = 0.9
    epsilon: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    action_size: int = 4

@dataclass
class RewardConfig:
    """Reward structure for the environment"""
    food_reward: float = 100
    reward_for_winning: float = 1000
    death_penalty: float = -100
    per_step_reward: float = -1

@dataclass
class EnvironmentConfig:
    """Environment settings"""
    max_steps_per_episode: int = 5000
    render: bool = True
    fps: int = 10

@dataclass
class TrainingConfig:
    """Top-level training configuration"""
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