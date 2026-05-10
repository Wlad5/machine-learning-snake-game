"""
Q-Learning Multi-Food Pattern Experiment
==========================================
Evaluates trained Q-Learning agents on boards that spawn multiple food items
simultaneously, placed according to different spatial patterns.

Configuration lives at the top of this file (section "EXPERIMENT
CONFIGURATION").  Just edit GRID_COLS / GRID_ROWS / FOOD_COUNTS /
FOOD_PATTERNS / EPISODES_PER_CONFIG and run:

    python multi_food_experiment.py

CSVs  → q_learning/multi_food_csv/
Plots → q_learning/multi_food_plots/
"""

import sys
import pickle
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pygame as pg

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GAME_DIR = PROJECT_ROOT / "game"
CURRENT_DIR = Path(__file__).resolve().parent

for _p in (PROJECT_ROOT, GAME_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from q_learning_agent import Q_learning_Agent
from state_encoding import State_Encoding
from state_encoding_distance import DistanceStateEncoding
from state_encoding_raycasting import RayCastingStateEncoding
from state_encoding_localgrid import LocalGridStateEncoding
from state_encoding_bodyawareness import BodyAwarenessStateEncoding
from game.board import Board
from game.enums import Direction, CellType
from game.snake import Snake

# ==============================================================================
#  EXPERIMENT CONFIGURATION  ← edit these values
# ==============================================================================

GRID_COLS = 4      # Board columns
GRID_ROWS = 4      # Board rows
CELL_SIZE = 33      # Pixels per cell (irrelevant for headless runs)

# Number of simultaneous food items to test (each value is a separate condition)
FOOD_COUNTS = [1, 2, 3, 5, 8, 14]

# Spatial patterns to test.  Remove any you don't want.
#   "uniform"       – completely random placement
#   "center_biased" – food weighted toward the board centre
#   "edge_biased"   – food weighted toward the board edges
#   "corners"       – food weighted toward the four corners
FOOD_PATTERNS = ["uniform", "center_biased", "edge_biased", "corners"]

# Evaluation episodes per (encoding × food_count × pattern) combination
EPISODES_PER_CONFIG = 100

# ── Watch mode ─────────────────────────────────────────────────────────────────
WATCH_MODE       = False
WATCH_FOOD_COUNT = 5      # simultaneous food items while watching
WATCH_FPS        = 1000   # playback speed (frames per second)
# ──────────────────────────────────────────────────────────────────────────────

MODELS_DIR     = CURRENT_DIR / "models"
CSV_OUTPUT_DIR = CURRENT_DIR / "multi_food_csv"
PLOTS_DIR      = CURRENT_DIR / "multi_food_plots"

# ==============================================================================
#  CONSTANTS  (no need to edit below this line for basic use)
# ==============================================================================

ENCODINGS = {
    "basic":      State_Encoding,
    "distance":   DistanceStateEncoding,
    "raycasting": RayCastingStateEncoding,
    "localgrid":  LocalGridStateEncoding,
    "bodyaware":  BodyAwarenessStateEncoding,
}

COLORS = {
    "basic":      "#1f77b4",
    "bodyaware":  "#ff7f0e",
    "distance":   "#2ca02c",
    "localgrid":  "#d62728",
    "raycasting": "#9467bd",
}

ENCODING_LABELS = {
    "basic":      "Basic",
    "distance":   "Distance",
    "raycasting": "Raycasting",
    "localgrid":  "Local Grid",
    "bodyaware":  "Body Aware",
}

PATTERN_LABELS = {
    "uniform":       "Uniform",
    "center_biased": "Center Biased",
    "edge_biased":   "Edge Biased",
    "corners":       "Corners",
}

PATTERN_COLORS = {
    "uniform":       "#5b9bd5",
    "center_biased": "#ed7d31",
    "edge_biased":   "#70ad47",
    "corners":       "#ffc000",
}

# ==============================================================================
#  FOOD PLACEMENT PATTERNS
# ==============================================================================

def _build_weight_matrix(cols: int, rows: int, pattern: str) -> np.ndarray:
    """Return a (rows, cols) probability-weight array for food spawning."""
    w = np.ones((rows, cols), dtype=np.float64)
    cx = (cols - 1) / 2.0
    cy = (rows - 1) / 2.0
    max_dist = np.sqrt(cx ** 2 + cy ** 2) + 1e-9

    if pattern == "uniform":
        pass  # uniform weights

    elif pattern == "center_biased":
        for y in range(rows):
            for x in range(cols):
                d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                w[y, x] = np.exp(-3.0 * (d / max_dist) ** 2)

    elif pattern == "edge_biased":
        for y in range(rows):
            for x in range(cols):
                d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                w[y, x] = 1.0 - np.exp(-3.0 * (d / max_dist) ** 2) + 0.05

    elif pattern == "corners":
        for y in range(rows):
            for x in range(cols):
                corner_dists = [
                    np.sqrt(x ** 2 + y ** 2),
                    np.sqrt((x - (cols - 1)) ** 2 + y ** 2),
                    np.sqrt(x ** 2 + (y - (rows - 1)) ** 2),
                    np.sqrt((x - (cols - 1)) ** 2 + (y - (rows - 1)) ** 2),
                ]
                d = min(corner_dists)
                w[y, x] = np.exp(-3.0 * (d / max_dist) ** 2)

    return np.maximum(w, 1e-9)


def _sample_position(cols: int, rows: int, forbidden: set, weights: np.ndarray):
    """Sample one grid cell not in *forbidden* using *weights* for probability.

    Returns ``None`` if every free cell has zero weight (e.g. board is full).
    """
    flat = weights.flatten().copy()
    for fx, fy in forbidden:
        if 0 <= fy < rows and 0 <= fx < cols:
            flat[fy * cols + fx] = 0.0
    total = flat.sum()
    if total <= 0.0:
        return None
    idx = np.random.choice(rows * cols, p=flat / total)
    y, x = divmod(idx, cols)
    return (x, y)


# ==============================================================================
#  MULTI-FOOD ENVIRONMENT
# ==============================================================================

class _FoodProxy:
    """Lightweight stand-in for a Food object; only exposes `.position`."""

    __slots__ = ("position",)

    def __init__(self, position):
        self.position = position


class MultiFoodEnv:
    """
    Snake environment with *food_count* simultaneous food items placed
    according to *food_pattern*.

    The state encoder always receives the food nearest to the snake head,
    so all existing single-food encoders work without modification.
    """

    def __init__(
        self,
        grid_cols: int,
        grid_rows: int,
        cell_size: int,
        food_count: int,
        food_pattern: str,
        state_encoder,
        food_reward: float = 50.0,
        death_penalty: float = -50.0,
        per_step_reward: float = -0.01,
        reward_for_winning: float = 10_000.0,
        length_bonus_multiplier: float = 10.0,
        milestone_rewards: dict = None,
        stagnation_scale: float = 10.0,
        revisit_penalty: float = 2.0,
        distance_shaping_scale: float = 0.3,
    ):
        self.cols = grid_cols
        self.rows = grid_rows
        self.cell_size = cell_size
        self.food_count = food_count
        self.food_pattern = food_pattern
        self.state_encoder = state_encoder

        # Dynamic step budget mirrors the single-food env
        self._max_steps_hard = max(800, grid_cols * grid_rows * 80)

        self.food_reward = food_reward
        self.death_penalty = death_penalty
        self.per_step_reward = per_step_reward
        self.reward_for_winning = reward_for_winning
        self.length_bonus_multiplier = length_bonus_multiplier
        self.milestone_rewards = milestone_rewards or {5: 100, 10: 200, 15: 300, 20: 500}
        self.stagnation_scale = stagnation_scale
        self.revisit_penalty = revisit_penalty
        self.distance_shaping_scale = distance_shaping_scale

        pg.init()
        self.board = Board(grid_cols * cell_size, grid_rows * cell_size, cell_size)

        # Pre-compute spatial weights once (static per pattern + grid size)
        self._weights = _build_weight_matrix(grid_cols, grid_rows, food_pattern)

        self.snake: Snake = None
        self.food_positions: list = []
        self._nearest_proxy = _FoodProxy((0, 0))

        self.done = False
        self.score = 0
        self.step_count = 0
        self.steps_since_last_food = 0
        self.visited_recently: dict = {}
        self._prev_dist_to_nearest = 0

    # ------------------------------------------------------------------
    def reset(self):
        self.snake = Snake(cols=self.cols, rows=self.rows, cell_size=self.cell_size)
        self.snake.set_direction(Direction.RIGHT)

        self.food_positions = []
        for _ in range(self.food_count):
            forbidden = self.snake.snake_positions | set(self.food_positions)
            pos = _sample_position(self.cols, self.rows, forbidden, self._weights)
            if pos is not None:
                self.food_positions.append(pos)

        self.done = False
        self.score = 0
        self.step_count = 0
        self.steps_since_last_food = 0
        self.visited_recently = {}

        self._update_nearest()
        self._prev_dist_to_nearest = self._manhattan(self._nearest_proxy.position)
        self._refresh_board()

        return self._get_state()

    # ------------------------------------------------------------------
    def step(self, action: int):
        if self.done:
            return self._get_state(), 0.0, True, {"score": self.score, "steps": self.step_count}

        _dir_map = {
            0: Direction.UP,
            1: Direction.DOWN,
            2: Direction.LEFT,
            3: Direction.RIGHT,
        }
        self.snake.set_direction(_dir_map.get(action, self.snake.get_direction()))
        self.snake.move()
        self.step_count += 1

        head = self.snake.snake_head
        body_no_head = list(self.snake.snake)[1:]
        reward = 0.0

        hit_wall = not self.board.in_bounds_cell(head)
        hit_self = head in body_no_head

        if hit_wall or hit_self:
            # Scale death penalty with score progress
            progress_mul = 1.0 + (self.score / 10.0) * 2.0
            reward += self.death_penalty * progress_mul
            self.done = True

        elif head in self.food_positions:
            self.score += 1
            reward += self.food_reward
            reward += len(self.snake.snake) ** 1.2 * self.length_bonus_multiplier
            if self.score in self.milestone_rewards:
                reward += self.milestone_rewards[self.score]

            self.steps_since_last_food = 0
            self.visited_recently = {}

            self.food_positions.remove(head)
            self.snake.grow()

            total_cells = self.cols * self.rows
            if len(self.snake.snake) >= total_cells:
                reward += self.reward_for_winning
                self.done = True
            else:
                # Respawn exactly one food to maintain food_count
                forbidden = self.snake.snake_positions | set(self.food_positions)
                new_pos = _sample_position(self.cols, self.rows, forbidden, self._weights)
                if new_pos is not None:
                    self.food_positions.append(new_pos)

        else:
            # Step penalty + stagnation
            self.steps_since_last_food += 1
            stag = self.steps_since_last_food / (self.cols * self.rows * 2)
            reward += self.per_step_reward * (1.0 + stag * self.stagnation_scale)

            # Distance shaping toward nearest food
            self._update_nearest()
            if self.food_positions:
                cur_dist = self._manhattan(self._nearest_proxy.position)
                norm_imp = (self._prev_dist_to_nearest - cur_dist) / (self.cols + self.rows)
                reward += norm_imp * self.food_reward * self.distance_shaping_scale
                self._prev_dist_to_nearest = cur_dist

            # Revisit penalty (discourages tight loops)
            cell = tuple(head)
            last_visit = self.visited_recently.get(cell, -999)
            gap = self.step_count - last_visit
            loop_threshold = self.cols * self.rows
            if gap < loop_threshold:
                reward -= self.revisit_penalty * (1.0 - gap / loop_threshold)
            self.visited_recently[cell] = self.step_count

        # Dynamic step budget
        if not self.done:
            dynamic_max = self.cols * self.rows * max(len(self.snake.snake), 3) * 2
            if self.step_count >= min(dynamic_max, self._max_steps_hard):
                reward += self.death_penalty * 0.5
                self.done = True

        self._update_nearest()
        self._refresh_board()
        return self._get_state(), reward, self.done, {"score": self.score, "steps": self.step_count}

    # ------------------------------------------------------------------
    def _manhattan(self, pos) -> float:
        hx, hy = self.snake.snake_head
        return abs(hx - pos[0]) + abs(hy - pos[1])

    def _update_nearest(self):
        if not self.food_positions:
            self._nearest_proxy.position = self.snake.snake_head
        else:
            self._nearest_proxy.position = min(self.food_positions, key=self._manhattan)

    def _refresh_board(self):
        self.board.reset_grid()
        for cell in self.snake.snake:
            self.board.set_cell(cell, CellType.SNAKE)
        for fp in self.food_positions:
            if self.board.in_bounds_cell(fp):
                self.board.set_cell(fp, CellType.FOOD)

    def _get_state(self):
        self._update_nearest()
        return self.state_encoder.encode(self.board, self.snake, self._nearest_proxy)

    def close(self):
        pass


# ==============================================================================
#  EXPERIMENT RUNNER
# ==============================================================================

def _load_agent(encoding_name: str):
    path = MODELS_DIR / f"q_learning_q_table_{encoding_name}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        q_table_dict = pickle.load(f)
    agent = Q_learning_Agent(action_size=4, epsilon=0.0)
    agent.q_table.update(q_table_dict)
    return agent


def _evaluate_config(
    encoding_name: str,
    agent: Q_learning_Agent,
    food_count: int,
    food_pattern: str,
    episodes: int,
) -> pd.DataFrame:
    env = MultiFoodEnv(
        grid_cols=GRID_COLS,
        grid_rows=GRID_ROWS,
        cell_size=CELL_SIZE,
        food_count=food_count,
        food_pattern=food_pattern,
        state_encoder=ENCODINGS[encoding_name](),
    )
    agent.epsilon = 0.0
    records = []

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = agent.choose_action(state)
            state, reward, done, info = env.step(action)
            total_reward += reward

        total_cells = GRID_COLS * GRID_ROWS
        is_win = int(len(env.snake.snake) >= total_cells)

        records.append(
            {
                "encoding":     encoding_name,
                "food_count":   food_count,
                "food_pattern": food_pattern,
                "grid":         f"{GRID_COLS}x{GRID_ROWS}",
                "episode":      ep,
                "reward":       total_reward,
                "score":        info["score"],
                "steps":        info["steps"],
                "win":          is_win,
            }
        )

    env.close()
    return pd.DataFrame(records)


def run_experiment(episodes_per_config: int = EPISODES_PER_CONFIG):
    """Run all (encoding × food_count × pattern) combinations and return DataFrames."""
    all_records = []

    for encoding_name in ENCODINGS:
        agent = _load_agent(encoding_name)
        if agent is None:
            print(f"[WARN] No model found for '{encoding_name}' — skipping.")
            continue

        print(f"\n=== {encoding_name.upper()} ===")
        for food_count in FOOD_COUNTS:
            for pattern in FOOD_PATTERNS:
                label = f"food={food_count}  pattern={pattern:<14}"
                print(f"  {label} ...", end="", flush=True)
                df = _evaluate_config(encoding_name, agent, food_count, pattern, episodes_per_config)
                all_records.append(df)
                mean_score = df["score"].mean()
                win_rate   = df["win"].mean() * 100
                print(f"  avg_score={mean_score:.2f}  win%={win_rate:.1f}")

    if not all_records:
        raise RuntimeError("No data generated — check that model files exist in q_learning/models/.")

    episode_df = pd.concat(all_records, ignore_index=True)

    summary = (
        episode_df
        .groupby(["encoding", "food_count", "food_pattern", "grid"], as_index=False)
        .agg(
            episodes    =("episode", "count"),
            avg_reward  =("reward",  "mean"),
            std_reward  =("reward",  "std"),
            avg_score   =("score",   "mean"),
            std_score   =("score",   "std"),
            avg_steps   =("steps",   "mean"),
            win_rate    =("win",     "mean"),
            total_wins  =("win",     "sum"),
        )
    )
    summary["win_rate"] = summary["win_rate"] * 100.0
    return episode_df, summary


# ==============================================================================
#  CSV OUTPUT
# ==============================================================================

def save_csv(episode_df: pd.DataFrame, summary: pd.DataFrame):
    CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    ep_path  = CSV_OUTPUT_DIR / f"multifood_episodes_{ts}.csv"
    sum_path = CSV_OUTPUT_DIR / f"multifood_summary_{ts}.csv"

    episode_df.to_csv(ep_path,  index=False)
    summary.to_csv(sum_path,    index=False)

    # Stable latest copies
    episode_df.to_csv(CSV_OUTPUT_DIR / "multifood_episodes_latest.csv",  index=False)
    summary.to_csv(CSV_OUTPUT_DIR    / "multifood_summary_latest.csv",   index=False)

    print(f"\nCSVs saved → {ep_path.name}")
    print(f"             {sum_path.name}")
    return ep_path, sum_path


# ==============================================================================
#  PLOTTING
# ==============================================================================

def _configure_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor":  "white",
            "axes.facecolor":    "#f7f8fa",
            "axes.edgecolor":    "#d0d7de",
            "axes.titleweight":  "bold",
            "axes.labelsize":    11,
            "axes.titlesize":    13,
            "legend.frameon":    True,
            "legend.facecolor":  "white",
            "legend.edgecolor":  "#d0d7de",
            "grid.alpha":        0.35,
            "grid.linestyle":    "--",
        }
    )


def _present_encodings(summary: pd.DataFrame) -> list:
    return [e for e in ENCODINGS if (summary["encoding"] == e).any()]


# ── Plot 1 ─────────────────────────────────────────────────────────────────────
def plot_score_vs_food_count(summary: pd.DataFrame) -> plt.Figure:
    """Average score vs food count — one subplot per pattern, one line per encoding."""
    n = len(FOOD_PATTERNS)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, pattern in zip(axes, FOOD_PATTERNS):
        sub = summary[summary["food_pattern"] == pattern]
        for enc in _present_encodings(summary):
            enc_sub = sub[sub["encoding"] == enc].sort_values("food_count")
            if enc_sub.empty:
                continue
            ax.plot(
                enc_sub["food_count"],
                enc_sub["avg_score"],
                marker="o",
                linewidth=2,
                label=ENCODING_LABELS.get(enc, enc),
                color=COLORS.get(enc),
            )
        ax.set_title(PATTERN_LABELS.get(pattern, pattern))
        ax.set_xlabel("Food Count")
        ax.set_xticks(FOOD_COUNTS)
        if ax is axes[0]:
            ax.set_ylabel("Average Score")
        ax.legend(fontsize=8)

    fig.suptitle("Average Score vs Food Count by Pattern", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


# ── Plot 2 ─────────────────────────────────────────────────────────────────────
def plot_win_rate_vs_food_count(summary: pd.DataFrame) -> plt.Figure:
    """Win rate (%) vs food count — one subplot per pattern, one line per encoding."""
    n = len(FOOD_PATTERNS)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, pattern in zip(axes, FOOD_PATTERNS):
        sub = summary[summary["food_pattern"] == pattern]
        for enc in _present_encodings(summary):
            enc_sub = sub[sub["encoding"] == enc].sort_values("food_count")
            if enc_sub.empty:
                continue
            ax.plot(
                enc_sub["food_count"],
                enc_sub["win_rate"],
                marker="o",
                linewidth=2,
                label=ENCODING_LABELS.get(enc, enc),
                color=COLORS.get(enc),
            )
        ax.set_title(PATTERN_LABELS.get(pattern, pattern))
        ax.set_xlabel("Food Count")
        ax.set_xticks(FOOD_COUNTS)
        ax.set_ylim(0, 100)
        if ax is axes[0]:
            ax.set_ylabel("Win Rate (%)")
        ax.legend(fontsize=8)

    fig.suptitle("Win Rate vs Food Count by Pattern", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


# ── Plot 3 ─────────────────────────────────────────────────────────────────────
def plot_steps_vs_food_count(summary: pd.DataFrame) -> plt.Figure:
    """Average survival steps vs food count (averaged over patterns)."""
    agg = (
        summary
        .groupby(["encoding", "food_count"], as_index=False)["avg_steps"]
        .mean()
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    for enc in _present_encodings(summary):
        enc_sub = agg[agg["encoding"] == enc].sort_values("food_count")
        if enc_sub.empty:
            continue
        ax.plot(
            enc_sub["food_count"],
            enc_sub["avg_steps"],
            marker="o",
            linewidth=2,
            label=ENCODING_LABELS.get(enc, enc),
            color=COLORS.get(enc),
        )

    ax.set_title("Average Survival Steps vs Food Count (averaged over patterns)")
    ax.set_xlabel("Food Count")
    ax.set_ylabel("Average Steps")
    ax.set_xticks(FOOD_COUNTS)
    ax.legend(title="State Encoding")
    fig.tight_layout()
    return fig


# ── Plot 4 ─────────────────────────────────────────────────────────────────────
def plot_pattern_comparison(summary: pd.DataFrame, food_count: int = None) -> plt.Figure:
    """Grouped bar chart comparing patterns for a specific food count."""
    if food_count is None:
        food_count = max(FOOD_COUNTS)

    sub  = summary[summary["food_count"] == food_count]
    encs = _present_encodings(sub)
    n_enc     = len(encs)
    n_pat     = len(FOOD_PATTERNS)
    bar_width = 0.8 / n_pat
    x = np.arange(n_enc)

    fig, axes = plt.subplots(1, 2, figsize=(7 * 2, 5))

    metrics = [
        ("avg_score", "Average Score",  f"Avg Score by Pattern  (food count = {food_count})"),
        ("win_rate",  "Win Rate (%)",   f"Win Rate by Pattern  (food count = {food_count})"),
    ]

    for ax, (metric, ylabel, title) in zip(axes, metrics):
        for pi, pattern in enumerate(FOOD_PATTERNS):
            pat_sub = sub[sub["food_pattern"] == pattern]
            values  = []
            for enc in encs:
                row = pat_sub[pat_sub["encoding"] == enc]
                values.append(float(row[metric].values[0]) if not row.empty else 0.0)

            offsets = x + (pi - (n_pat - 1) / 2) * bar_width
            ax.bar(
                offsets, values, width=bar_width,
                label=PATTERN_LABELS.get(pattern, pattern),
                color=PATTERN_COLORS.get(pattern),
                alpha=0.88,
            )

        ax.set_title(title)
        ax.set_xlabel("State Encoding")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([ENCODING_LABELS.get(e, e) for e in encs], rotation=15)
        ax.legend(title="Pattern")
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.35)

    fig.suptitle(f"Pattern Impact at {food_count} Food Items", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


# ── Plot 5 ─────────────────────────────────────────────────────────────────────
def plot_heatmap(summary: pd.DataFrame, food_count: int = None) -> plt.Figure:
    """Heatmap of avg score: encoding (rows) × pattern (columns)."""
    if food_count is None:
        food_count = max(FOOD_COUNTS)

    sub  = summary[summary["food_count"] == food_count]
    encs = _present_encodings(sub)

    matrix = np.zeros((len(encs), len(FOOD_PATTERNS)))
    for i, enc in enumerate(encs):
        for j, pat in enumerate(FOOD_PATTERNS):
            row = sub[(sub["encoding"] == enc) & (sub["food_pattern"] == pat)]
            if not row.empty:
                matrix[i, j] = float(row["avg_score"].values[0])

    fig, ax = plt.subplots(figsize=(8, max(3, len(encs) + 1)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="Average Score")

    ax.set_xticks(range(len(FOOD_PATTERNS)))
    ax.set_xticklabels([PATTERN_LABELS.get(p, p) for p in FOOD_PATTERNS])
    ax.set_yticks(range(len(encs)))
    ax.set_yticklabels([ENCODING_LABELS.get(e, e) for e in encs])

    vmax = matrix.max() if matrix.max() > 0 else 1.0
    for i in range(len(encs)):
        for j in range(len(FOOD_PATTERNS)):
            text_color = "white" if matrix[i, j] > vmax * 0.6 else "black"
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center",
                    fontsize=10, color=text_color)

    ax.set_title(
        f"Avg Score Heatmap: Encoding × Pattern  (food count = {food_count})",
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


# ── Plot 6 ─────────────────────────────────────────────────────────────────────
def plot_score_vs_food_count_combined(summary: pd.DataFrame) -> plt.Figure:
    """Avg score vs food count — all patterns overlaid, one subplot per encoding."""
    encs = _present_encodings(summary)
    n = len(encs)
    cols_n = min(n, 3)
    rows_n = (n + cols_n - 1) // cols_n

    fig, axes = plt.subplots(rows_n, cols_n, figsize=(5 * cols_n, 4 * rows_n), sharey=True)
    axes_flat = np.array(axes).flatten()

    for ax_idx, enc in enumerate(encs):
        ax = axes_flat[ax_idx]
        enc_sub = summary[summary["encoding"] == enc]
        for pattern in FOOD_PATTERNS:
            pat_sub = enc_sub[enc_sub["food_pattern"] == pattern].sort_values("food_count")
            if pat_sub.empty:
                continue
            ax.plot(
                pat_sub["food_count"],
                pat_sub["avg_score"],
                marker="o",
                linewidth=2,
                label=PATTERN_LABELS.get(pattern, pattern),
                color=PATTERN_COLORS.get(pattern),
            )
        ax.set_title(ENCODING_LABELS.get(enc, enc))
        ax.set_xlabel("Food Count")
        ax.set_xticks(FOOD_COUNTS)
        if ax_idx % cols_n == 0:
            ax.set_ylabel("Average Score")
        ax.legend(fontsize=8)

    # Hide unused subplot slots
    for idx in range(len(encs), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Avg Score vs Food Count — All Patterns per Encoding",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


# ── Save all plots ──────────────────────────────────────────────────────────────
def save_plots(summary: pd.DataFrame):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    _configure_style()

    plots = {
        "1_score_vs_food_count_by_pattern.png":    plot_score_vs_food_count(summary),
        "2_win_rate_vs_food_count_by_pattern.png": plot_win_rate_vs_food_count(summary),
        "3_steps_vs_food_count.png":               plot_steps_vs_food_count(summary),
        "4_score_per_encoding_all_patterns.png":   plot_score_vs_food_count_combined(summary),
    }

    for name, fig in plots.items():
        path = PLOTS_DIR / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {name}")


# ==============================================================================
#  WATCH MODE  (visual playback)
# ==============================================================================

# Each food item gets its own colour so they are easy to distinguish.
_FOOD_COLORS = [
    (255,  80,  80),   # red
    ( 80, 220,  80),   # green
    ( 80, 160, 255),   # blue
    (255, 200,  60),   # yellow
    (200,  80, 220),   # purple
    (255, 140,   0),   # orange
    (  0, 210, 210),   # cyan
    (255, 105, 180),   # pink
]


def watch_mode(
    encoding_names: list,
    food_count: int,
    food_patterns: list,
    fps: int = 10,
    episodes: int = 10,
):
    """Open a single pygame window and cycle through all encoding × pattern combos."""
    cell  = CELL_SIZE
    bw    = GRID_COLS * cell
    bh    = GRID_ROWS * cell
    panel = 90

    pg.init()
    screen     = pg.display.set_mode((bw, bh + panel))
    clock_font = pg.font.SysFont(None, 28)
    small      = pg.font.SysFont(None, 21)
    tick       = pg.time.Clock()

    total_combos = len(encoding_names) * len(food_patterns)
    combo_idx    = 0
    running      = True

    for encoding_name in encoding_names:
        if not running:
            break

        agent = _load_agent(encoding_name)
        if agent is None:
            print(f"[WARN] No model for '{encoding_name}' — skipping.")
            combo_idx += len(food_patterns)
            continue
        agent.epsilon = 0.0

        for food_pattern in food_patterns:
            if not running:
                break

            combo_idx += 1
            pg.display.set_caption(
                f"Multi-Food Watch  [{combo_idx}/{total_combos}]  |  "
                f"{ENCODING_LABELS.get(encoding_name, encoding_name)}  |  "
                f"{food_count} food  |  {PATTERN_LABELS.get(food_pattern, food_pattern)}"
            )
            print(
                f"  [{combo_idx}/{total_combos}]  "
                f"{encoding_name:<10}  pattern={food_pattern}"
            )

            env = MultiFoodEnv(
                grid_cols=GRID_COLS,
                grid_rows=GRID_ROWS,
                cell_size=cell,
                food_count=food_count,
                food_pattern=food_pattern,
                state_encoder=ENCODINGS[encoding_name](),
            )

            ep = 0
            while running and ep < episodes:
                state = env.reset()
                done  = False
                ep   += 1

                while not done and running:
                    # ── event handling ──────────────────────────────────────
                    for event in pg.event.get():
                        if event.type == pg.QUIT:
                            running = False
                        elif event.type == pg.KEYDOWN and event.key in (pg.K_ESCAPE, pg.K_q):
                            running = False

                    if not running:
                        break

                    action = agent.choose_action(state)
                    state, _, done, info = env.step(action)

                    # ── draw board background + grid ────────────────────────
                    screen.fill((10, 10, 10))
                    for x in range(0, bw + 1, cell):
                        pg.draw.line(screen, (35, 35, 35), (x, 0), (x, bh))
                    for y in range(0, bh + 1, cell):
                        pg.draw.line(screen, (35, 35, 35), (0, y), (bw, y))

                    # ── food items ──────────────────────────────────────────
                    for i, fp in enumerate(env.food_positions):
                        color = _FOOD_COLORS[i % len(_FOOD_COLORS)]
                        px, py = fp[0] * cell, fp[1] * cell
                        margin = max(3, cell // 8)
                        pg.draw.rect(
                            screen, color,
                            pg.Rect(px + margin, py + margin, cell - 2 * margin, cell - 2 * margin),
                        )
                        pg.draw.rect(
                            screen, (240, 240, 240),
                            pg.Rect(px + margin, py + margin, cell - 2 * margin, cell - 2 * margin),
                            1,
                        )

                    # ── snake body ──────────────────────────────────────────
                    body = list(env.snake.snake)
                    n    = max(len(body), 1)
                    for idx, seg in enumerate(reversed(body[1:])):
                        t  = idx / n
                        g  = int(100 + 120 * t)
                        px, py = seg[0] * cell, seg[1] * cell
                        pg.draw.rect(
                            screen, (30, g, 30),
                            pg.Rect(px + 1, py + 1, cell - 2, cell - 2),
                        )

                    # ── snake head ──────────────────────────────────────────
                    if body:
                        hx, hy = body[0]
                        px, py = hx * cell, hy * cell
                        pg.draw.rect(
                            screen, (255, 0, 220),
                            pg.Rect(px + 1, py + 1, cell - 2, cell - 2),
                        )

                    # ── nearest-food highlight ───────────────────────────────
                    if env.food_positions:
                        nx, ny = env._nearest_proxy.position
                        px, py = nx * cell, ny * cell
                        surf = pg.Surface((cell, cell), pg.SRCALPHA)
                        surf.fill((255, 255, 255, 40))
                        screen.blit(surf, (px, py))

                    # ── status panel ────────────────────────────────────────
                    pg.draw.rect(screen, (25, 25, 25), pg.Rect(0, bh, bw, panel))
                    pg.draw.line(screen, (70, 70, 70), (0, bh), (bw, bh), 2)

                    line1 = clock_font.render(
                        f"Ep {ep}/{episodes}   Score: {info['score']}   Steps: {info['steps']}   "
                        f"[{combo_idx}/{total_combos}]",
                        True, (230, 230, 230),
                    )
                    line2 = small.render(
                        f"Encoding: {ENCODING_LABELS.get(encoding_name, encoding_name)}    "
                        f"Food: {food_count}    "
                        f"Pattern: {PATTERN_LABELS.get(food_pattern, food_pattern)}",
                        True, (170, 170, 170),
                    )
                    line3 = small.render(
                        f"Grid: {GRID_COLS}×{GRID_ROWS}    ESC / Q to quit",
                        True, (100, 100, 100),
                    )
                    screen.blit(line1, (10, bh + 8))
                    screen.blit(line2, (10, bh + 36))
                    screen.blit(line3, (10, bh + 60))

                    pg.display.flip()
                    tick.tick(fps)

                if running:
                    pg.time.wait(400)

            env.close()

    pg.quit()


# ==============================================================================
#  ENTRY POINT
# ==============================================================================

def _print_summary(summary: pd.DataFrame):
    print("\n" + "=" * 72)
    print("RESULTS  (averaged over patterns)")
    print("=" * 72)
    agg = (
        summary
        .groupby(["encoding", "food_count"], as_index=False)
        .agg(
            avg_score =("avg_score", "mean"),
            win_rate  =("win_rate",  "mean"),
            avg_steps =("avg_steps", "mean"),
        )
    )
    agg["avg_score"] = agg["avg_score"].map(lambda v: f"{v:.2f}")
    agg["win_rate"]  = agg["win_rate"].map(lambda v: f"{v:.1f}%")
    agg["avg_steps"] = agg["avg_steps"].map(lambda v: f"{v:.0f}")
    print(agg.to_string(index=False))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Q-Learning Multi-Food Pattern Experiment")

    # ── watch mode ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Open a pygame window and watch the agent play instead of running the full experiment.",
    )
    parser.add_argument(
        "--encoding",
        default="localgrid",
        choices=list(ENCODINGS.keys()),
        help="State encoding to use in watch mode (default: localgrid).",
    )
    parser.add_argument(
        "--food-count",
        type=int,
        default=FOOD_COUNTS[1] if len(FOOD_COUNTS) > 1 else FOOD_COUNTS[0],
        help="Number of simultaneous food items for watch mode (default: second value in FOOD_COUNTS).",
    )
    parser.add_argument(
        "--pattern",
        default="uniform",
        choices=list(FOOD_PATTERNS) if isinstance(FOOD_PATTERNS[0], str) else ["uniform"],
        help="Food placement pattern for watch mode (default: uniform).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second in watch mode (default: 10).",
    )
    # ── experiment mode ─────────────────────────────────────────────────────
    parser.add_argument(
        "--episodes",
        type=int,
        default=EPISODES_PER_CONFIG,
        help=f"Episodes per (encoding × food_count × pattern) config (default {EPISODES_PER_CONFIG}).",
    )
    args = parser.parse_args()

    if args.watch or WATCH_MODE:
        fcount  = args.food_count if args.watch else WATCH_FOOD_COUNT
        fps_val = args.fps        if args.watch else WATCH_FPS
        encs    = [args.encoding] if args.watch else list(ENCODINGS.keys())
        pats    = [args.pattern]  if args.watch else list(FOOD_PATTERNS)
        print(
            f"\nWatch mode: encodings={encs}  food_count={fcount}"
            f"  patterns={pats}  fps={fps_val}  episodes={args.episodes}"
        )
        watch_mode(
            encoding_names=encs,
            food_count=fcount,
            food_patterns=pats,
            fps=fps_val,
            episodes=args.episodes,
        )

    total_configs = len(ENCODINGS) * len(FOOD_COUNTS) * len(FOOD_PATTERNS)
    print(f"\nQ-Learning Multi-Food Pattern Experiment")
    print(f"  Grid        : {GRID_COLS}x{GRID_ROWS}")
    print(f"  Food counts : {FOOD_COUNTS}")
    print(f"  Patterns    : {FOOD_PATTERNS}")
    print(f"  Episodes    : {args.episodes} per config")
    print(f"  Total runs  : {total_configs} configs × {args.episodes} episodes")

    episode_df, summary = run_experiment(args.episodes)
    save_csv(episode_df, summary)

    print("\nGenerating plots ...")
    save_plots(summary)

    _print_summary(summary)
    print("\nDone.")


if __name__ == "__main__":
    main()
