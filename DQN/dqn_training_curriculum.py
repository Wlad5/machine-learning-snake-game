"""
DQN Curriculum Training for Snake Generalization
=================================================
Trains a single DQN agent across multiple randomized grid sizes so it learns
a grid-size-invariant policy instead of memorizing one board layout.

Key ideas vs single-grid training:
  - Each episode randomly selects a grid from the curriculum list.
  - Progressive unlocking: starts with small grids, gradually introduces larger ones.
  - Larger replay buffer and slower epsilon decay for more diverse experience.
  - Saved model uses "_curriculum" suffix so it does not overwrite single-grid models.

Usage (train all encodings):
    python dqn_training_curriculum.py

Usage (single encoding, custom settings):
    python dqn_training_curriculum.py --encoding raycasting --episodes 40000 --grids 3x3 4x4 5x5 6x6 7x7
"""

import sys
import csv
import random
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GAME_DIR = PROJECT_ROOT / "game"
CURRENT_DIR = Path(__file__).resolve().parent

for path in (PROJECT_ROOT, GAME_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dqn_agent import DQNAgent
from dqn_snake_env import DQNSnakeEnv
from dqn_state_encoding import DQNStateEncoding
from dqn_state_encoding_distance import DQNDistanceStateEncoding
from dqn_state_encoding_raycasting import DQNRayCastingStateEncoding
from dqn_state_encoding_localgrid import DQNLocalGridStateEncoding
from dqn_state_encoding_bodyawareness import DQNBodyAwarenessStateEncoding


# Fixed feature-vector size for each encoding (does NOT change with grid size).
ENCODING_FEATURE_SIZES = {
    "basic": 16,      # 4 direction + 2 norm food offsets + 4 binary food dir + 4 danger + 2 tail offset
    "distance": 17,   # 4 direction + 4 food dir + 4 wall dist + food dist + 4 danger flags
    "raycasting": 23, # 4 direction + 8 rays × 2 + 2 tail offset + 1 norm food dist
    "localgrid": 43,  # 4 direction + 4 food dir + 30 grid (15 cells × 2) + 2 norm food offset + 1 food dist + 2 tail offset
    "bodyaware": 15,  # 4 direction + 4 food dir + 4 danger + length + 2 tail offset
}

ENCODING_MAP = {
    "basic": DQNStateEncoding,
    "distance": DQNDistanceStateEncoding,
    "raycasting": DQNRayCastingStateEncoding,
    "localgrid": DQNLocalGridStateEncoding,
    "bodyaware": DQNBodyAwarenessStateEncoding,
}

# Default curriculum: ordered from easiest to hardest.
DEFAULT_CURRICULUM_GRIDS = [(3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)]

CELL_SIZE = 33  # pixel size — irrelevant for headless training


def _max_steps_for_grid(cols: int, rows: int) -> int:
    """Scale the per-episode step budget with board area."""
    return max(600, cols * rows * 60)


def train_dqn_curriculum(
    num_episodes: int = 100_000,
    curriculum_grids=None,
    progressive: bool = True,
    progressive_unlock_frac: float = 0.40,
    encoding_name: str = "raycasting",
    learning_rate: float = 0.0005,
    gamma: float = 0.99,
    epsilon: float = 1.0,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.99993,
    batch_size: int = 64,
    memory_size: int = 100_000,
    hidden_size: int = 256,
    num_layers: int = 3,
    food_reward: float = 50.0,
    death_penalty: float = -50.0,
    per_step_reward: float = -0.01,
    base_reward_for_winning: float = 10_000.0,
    scale_win_reward_with_grid: bool = False,
    update_frequency: int = 500,
    save_path: str = None,
    save_as_standard: bool = True,
    print_every: int = 500,
):
    """
    Train a single DQN agent on multiple grid sizes (curriculum).

    Parameters
    ----------
    num_episodes : total training episodes across all grids.
    curriculum_grids : list of (cols, rows) tuples, ordered easy → hard.
    progressive : if True, use weighted sampling so smaller grids dominate early
        and larger grids ramp up to equal weight by progressive_unlock_frac.
    progressive_unlock_frac : fraction of training by which ALL grids have
        equal weight.  Smaller than 0.5 = large grids unlocked sooner.
    scale_win_reward_with_grid : scale reward_for_winning by (grid_area / 9)
        so winning a 6x6 grid is proportionally more rewarded than a 3x3.
    save_as_standard : also save the model under the standard filename
        (dqn_trained_model_{encoding}.pkl) so the generalization experiment
        will automatically pick it up.
    num_layers : number of hidden layers in the DQN network (default 3).
    """
    if curriculum_grids is None:
        curriculum_grids = DEFAULT_CURRICULUM_GRIDS

    EncoderClass = ENCODING_MAP.get(encoding_name, DQNStateEncoding)
    feature_size = ENCODING_FEATURE_SIZES.get(encoding_name, 12)

    # ------------------------------------------------------------------
    # Create one headless environment per grid size.
    # Each env gets its own encoder instance so internal state is isolated.
    # ------------------------------------------------------------------
    BASE_GRID_AREA = 9  # 3x3 — win reward is scaled relative to this

    envs: dict = {}
    for cols, rows in curriculum_grids:
        if scale_win_reward_with_grid:
            win_reward = base_reward_for_winning * ((cols * rows) / BASE_GRID_AREA)
        else:
            win_reward = base_reward_for_winning
        envs[(cols, rows)] = DQNSnakeEnv(
            render_mode=False,
            grid_cols=cols,
            grid_rows=rows,
            cell_size=CELL_SIZE,
            max_steps_per_episode=_max_steps_for_grid(cols, rows),
            food_reward=food_reward,
            death_penalty=death_penalty,
            per_step_reward=per_step_reward,
            reward_for_winning=win_reward,
            state_encoder=EncoderClass(),
        )

    # ------------------------------------------------------------------
    # Single shared agent — trained on all grid sizes simultaneously.
    # ------------------------------------------------------------------
    agent = DQNAgent(
        state_size=feature_size,
        action_size=4,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size,
        memory_size=memory_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        update_frequency=update_frequency,
    )

    training_csv_dir = CURRENT_DIR / "training_csv"
    training_csv_dir.mkdir(parents=True, exist_ok=True)
    history_file = training_csv_dir / f"dqn_training_stats_{encoding_name}_curriculum.csv"

    fieldnames = ["episode", "grid", "reward", "score", "steps", "win", "epsilon"]
    csv_file = open(history_file, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    records_window: list = []  # rolling window for console logging

    try:
        for episode in range(num_episodes):
            # ----------------------------------------------------------
            # Progressive curriculum: weighted sampling so the agent sees
            # ALL grid sizes from the start, but smaller grids dominate
            # early. By progressive_unlock_frac all grids have equal weight.
            # This avoids the sharp distribution shift of hard cutoffs.
            # ----------------------------------------------------------
            if progressive and len(curriculum_grids) > 1:
                progress = episode / max(num_episodes - 1, 1)
                # unlock_frac goes 0 → 1 over the first progressive_unlock_frac
                # fraction of training, then stays at 1.
                unlock_frac = min(progress / max(progressive_unlock_frac, 1e-9), 1.0)
                n = len(curriculum_grids)
                weights = []
                for i in range(n):
                    # Grid i's weight linearly ramps from min_w to 1.0 as
                    # unlock_frac goes from i/(n-1) to 1.0.  Grids below the
                    # current unlock cursor always have weight ≥ min_w so
                    # the agent is never completely blind to harder grids.
                    min_w = 0.05
                    threshold = i / max(n - 1, 1)
                    w = min_w + (1.0 - min_w) * min(unlock_frac / max(threshold + 1e-9, 1e-9), 1.0)
                    weights.append(min(w, 1.0))
                cols, rows = random.choices(curriculum_grids, weights=weights)[0]
            else:
                cols, rows = random.choice(curriculum_grids)

            env = envs[(cols, rows)]

            state = env.reset()
            episode_reward = 0.0
            done = False

            while not done:
                action = agent.choose_action(state)
                next_state, reward, done, info = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                agent.replay()
                state = next_state
                episode_reward += reward

            agent.decay_epsilon()

            if (episode + 1) % update_frequency == 0:
                agent.update_target_network()

            # Win: snake occupied every cell on the board.
            total_cells = cols * rows
            is_win = int(len(env.snake.snake) >= total_cells)

            row = {
                "episode": episode + 1,
                "grid": f"{cols}x{rows}",
                "reward": round(episode_reward, 2),
                "score": info.get("score", 0),
                "steps": info.get("steps", 0),
                "win": is_win,
                "epsilon": round(agent.epsilon, 6),
            }
            writer.writerow(row)
            records_window.append(row)

            # ----------------------------------------------------------
            # Console progress report every print_every episodes.
            # ----------------------------------------------------------
            if (episode + 1) % print_every == 0:
                window = records_window[-print_every:]
                avg_score = sum(r["score"] for r in window) / len(window)
                win_rate = sum(r["win"] for r in window) / len(window) * 100
                grids_seen = sorted({r["grid"] for r in window})
                print(
                    f"[{encoding_name}] ep {episode + 1:>6}/{num_episodes} | "
                    f"grids={grids_seen} | "
                    f"avg_score={avg_score:.2f} | "
                    f"win%={win_rate:.1f} | "
                    f"ε={agent.epsilon:.4f}"
                )

    finally:
        csv_file.close()
        for env in envs.values():
            env.close()

    # ------------------------------------------------------------------
    # Save the trained model.
    # ------------------------------------------------------------------
    if save_path is None:
        models_dir = CURRENT_DIR / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(
            models_dir / f"dqn_trained_model_{encoding_name}_curriculum.pkl"
        )

    agent.save(save_path)
    print(f"[{encoding_name}] Model saved → {save_path}")

    if save_as_standard:
        models_dir = CURRENT_DIR / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        standard_path = str(models_dir / f"dqn_trained_model_{encoding_name}.pkl")
        agent.save(standard_path)
        print(f"[{encoding_name}] Also saved as standard → {standard_path}")

    return agent


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DQN Curriculum Training — generalizes across grid sizes"
    )
    parser.add_argument(
        "--encoding",
        default=None,
        choices=list(ENCODING_MAP.keys()),
        help="Train a single encoding. Omit to train all.",
    )
    parser.add_argument(
        "--episodes", type=int, default=100_000, help="Total training episodes."
    )
    parser.add_argument(
        "--grids",
        nargs="+",
        default=None,
        metavar="WxH",
        help="Grid sizes e.g. --grids 3x3 4x4 5x5 6x6 (default: 3x3 4x4 5x5 6x6)",
    )
    parser.add_argument(
        "--no-progressive",
        action="store_true",
        help="Disable progressive unlocking (use all grids from episode 0).",
    )
    parser.add_argument(
        "--unlock-frac",
        type=float,
        default=0.40,
        help="Fraction of training by which all grids reach equal weight (default 0.40).",
    )
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--memory-size", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--epsilon-decay", type=float, default=0.99993)
    parser.add_argument(
        "--save-as-standard",
        action="store_true",
        help="Also overwrite the standard model file (dqn_trained_model_{enc}.pkl) "
             "so the generalization experiment picks it up automatically.",
    )
    args = parser.parse_args()

    grids = None
    if args.grids:
        grids = [tuple(int(v) for v in g.split("x")) for g in args.grids]

    encodings_to_train = (
        [args.encoding] if args.encoding else list(ENCODING_MAP.keys())
    )

    for enc in encodings_to_train:
        print(f"\n{'=' * 60}")
        print(f"  Curriculum training: {enc.upper()}")
        print(f"  Episodes   : {args.episodes}")
        print(f"  Grids      : {grids or DEFAULT_CURRICULUM_GRIDS}")
        print(f"  Progressive: {not args.no_progressive} (equal weight @{args.unlock_frac*100:.0f}%)")
        print(f"  Net        : {args.num_layers} layers × {args.hidden_size} units")
        print(f"{'=' * 60}")
        train_dqn_curriculum(
            num_episodes=args.episodes,
            curriculum_grids=grids,
            progressive=not args.no_progressive,
            progressive_unlock_frac=args.unlock_frac,
            encoding_name=enc,
            learning_rate=args.lr,
            epsilon_decay=args.epsilon_decay,
            batch_size=args.batch_size,
            memory_size=args.memory_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            save_as_standard=True,
        )
