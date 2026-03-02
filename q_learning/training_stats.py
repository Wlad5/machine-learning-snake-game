from dataclasses import dataclass
from typing import List, Dict, Any
import pandas as pd

@dataclass
class EpisodeResult:
    reward  : float
    score   : int
    steps   : int
    win     : bool
    epsilon : float

class TrainingStatistics:
    
    def __init__(self):
        self.results: List[EpisodeResult] = []
        self.df = pd.DataFrame(columns=['episode', 'reward'])
    
    def add_episode(self, result: EpisodeResult, episode_num: int) -> None:
        self.results.append(result)
        new_row = pd.DataFrame({
            'episode': [episode_num],
            'steps': [result.steps],
            'reward': [round(result.reward, 3)],
            'score': [result.score],
            'win': [result.win],
            'epsilon': [round(result.epsilon, 3)],
        })
        self.df = pd.concat([self.df, new_row], ignore_index=True)
    
    def save_to_csv(self, filepath: str) -> None:
        self.df.to_csv(filepath, index=False)

    def get_final_stats(self) -> Dict[str, Any]:
        if not self.results:
            return {
                "total_episodes"    : 0,
                "avg_reward"        : 0.0,
                "avg_score"         : 0.0,
                "avg_steps"         : 0.0,
                "best_score"        : 0,
                "total_wins"        : 0,
            }
        
        rewards = [r.reward for r in self.results]
        scores  = [r.score for r in self.results]
        steps   = [r.steps for r in self.results]
        wins    = [r.win for r in self.results]
        
        return {
            "total_episodes"    : len(self.results),
            "avg_reward"        : sum(rewards) / len(rewards),
            "avg_score"         : sum(scores) / len(scores),
            "avg_steps"         : sum(steps) / len(steps),
            "best_score"        : max(scores),
            "total_wins"        : sum(wins),
        }
    

# DURING TRAINING (Per Episode)
# Episode return (cumulative reward)
# Final score (snake length achieved)
# Steps taken in episode
# Win flag (boolean - did snake fill board)
# Food collected count (number of food pieces eaten)
# Current epsilon (exploration rate at that episode)\

# DURING TRAINING (Calculated/Aggregated)
# Rolling average return (last 50 episodes)
# Rolling average score (last 50 episodes)
# Rolling average steps (last 50 episodes)
# Running max score (best score achieved so far in training)
# Variance of returns (for stability - computed across all episodes so far)
# Success rate (% of episodes that resulted in win - last 50 episodes)
# Timestamp per episode (wall-clock time for training duration calculation)

# AFTER TRAINING COMPLETES (Summary Statistics)
# Total episodes completed
# Average return across all episodes
# Average score across all episodes
# Average steps across all episodes
# Best score achieved during training
# Total wins during training
# Total training time (wall-clock)
# Final epsilon value
# Training seed used

# AT TEST TIME (Per Environment Variant)
# For each test configuration (10x10, 12x12, clustered food, etc.):
# 23. Per-episode return (test)
# 24. Per-episode score (test)
# 25. Per-episode steps (test)
# 26. Per-episode win flag (test)
# 27. Average return across N test episodes
# 28. Average score across N test episodes
# 29. Variance of returns (for stability comparison vs training)
# 30. Performance drop % (test avg score vs training avg score)

# EXPERIMENT METADATA (Saved Once Per Run)
# Training seed
# Test seed
# State representation complexity level
# Learning rate (α)
# Discount factor (γ)
# Epsilon initial, min, decay
# Reward structure (food_reward, food_penalty, death_penalty, per_step_reward)
# Grid size (training)
# Max steps per episode
# Number of training episodes