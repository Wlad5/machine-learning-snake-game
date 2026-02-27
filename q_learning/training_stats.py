from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class EpisodeResult:
    """Container for a single episode's outcome."""
    reward  : float
    score   : int
    steps   : int
    win     : bool


class TrainingStatistics:
    """Manages and aggregates training statistics across episodes."""
    
    def __init__(self):
        self.results: List[EpisodeResult] = []
    
    def add_episode(self, result: EpisodeResult) -> None:
        """Add an episode result to the statistics."""
        self.results.append(result)
    
    def get_final_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics for all completed episodes.
        
        Returns:
            Dictionary with aggregated statistics
        """
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