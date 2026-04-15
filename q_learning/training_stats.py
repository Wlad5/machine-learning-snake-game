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