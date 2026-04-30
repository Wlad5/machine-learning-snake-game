import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

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


# Add or remove grid sizes here.
GRID_SIZES = [
    (3, 3),
    (4, 4),
    (5, 5),
    (6, 6),
]

EPISODES_PER_GRID = 2000
CELL_SIZE = 33

MODELS_DIR = CURRENT_DIR / "models"
CSV_OUTPUT_DIR = CURRENT_DIR / "generalization_csv"
PLOTS_OUTPUT_DIR = CURRENT_DIR / "generalization_plots"

ENCODINGS = {
    "basic": DQNStateEncoding,
    "distance": DQNDistanceStateEncoding,
    "raycasting": DQNRayCastingStateEncoding,
    "localgrid": DQNLocalGridStateEncoding,
    "bodyaware": DQNBodyAwarenessStateEncoding,
}

COLORS = {
    "basic": "#1f77b4",
    "bodyaware": "#ff7f0e",
    "distance": "#2ca02c",
    "localgrid": "#d62728",
    "raycasting": "#9467bd",
}

ENCODING_LABELS = {
    "basic": "Basic",
    "distance": "Distance",
    "raycasting": "Raycasting",
    "localgrid": "Local Grid",
    "bodyaware": "Body Aware",
}


def _grid_sort_key(label: str):
    cols, rows = label.split("x")
    return int(cols) * int(rows), int(cols), int(rows)


def _ordered_grid_labels(summary: pd.DataFrame):
    labels = summary["grid_label"].drop_duplicates().tolist()
    return sorted(labels, key=_grid_sort_key)


def _encoding_label(encoding_name: str) -> str:
    return ENCODING_LABELS.get(encoding_name, encoding_name)


def configure_plot_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#f7f8fa",
            "axes.edgecolor": "#d0d7de",
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.titlesize": 14,
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "#d0d7de",
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
        }
    )


def load_dqn_agent(filepath: Path):
    if not filepath.exists():
        return None
    return DQNAgent.load(str(filepath), epsilon=0.0)


def max_steps_for_grid(cols: int, rows: int) -> int:
    # Scale episode horizon with board area so larger boards have enough time.
    return max(800, cols * rows * 60)


def evaluate_single_grid(encoding_name: str, agent: DQNAgent, cols: int, rows: int, episodes: int) -> pd.DataFrame:
    env = DQNSnakeEnv(
        render_mode=False,
        grid_cols=cols,
        grid_rows=rows,
        cell_size=CELL_SIZE,
        max_steps_per_episode=max_steps_for_grid(cols, rows),
        food_reward=100,
        death_penalty=-300,
        per_step_reward=-0.05,
        reward_for_winning=2000,
        state_encoder=ENCODINGS[encoding_name](),
    )

    # Ensure deterministic evaluation mode.
    agent.epsilon = 0.0

    records = []

    try:
        for episode in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0.0

            while not done:
                action = agent.choose_action(state)
                next_state, reward, done, info = env.step(action)
                total_reward += reward
                state = next_state

            total_cells = env.board.cols * env.board.rows
            is_win = int(len(env.snake.snake) >= total_cells)

            records.append(
                {
                    "encoding": encoding_name,
                    "grid_cols": cols,
                    "grid_rows": rows,
                    "grid_label": f"{cols}x{rows}",
                    "grid_area": cols * rows,
                    "episode": episode,
                    "reward": total_reward,
                    "score": info["score"],
                    "steps": info["steps"],
                    "win": is_win,
                }
            )
    finally:
        env.close()

    return pd.DataFrame(records)


def run_generalization_experiment(grid_sizes, episodes_per_grid: int):
    all_episode_results = []

    for encoding_name in ENCODINGS:
        model_file = MODELS_DIR / f"dqn_trained_model_{encoding_name}.pkl"
        agent = load_dqn_agent(model_file)

        if agent is None:
            print(f"[WARN] Missing model for {encoding_name}: {model_file}")
            continue

        print(f"\n=== {encoding_name.upper()} ===")
        for cols, rows in grid_sizes:
            print(f"Evaluating grid {cols}x{rows} ({episodes_per_grid} episodes)...")
            grid_df = evaluate_single_grid(encoding_name, agent, cols, rows, episodes_per_grid)
            all_episode_results.append(grid_df)

    if not all_episode_results:
        raise RuntimeError("No experiment data was generated. Check model files in DQN/models.")

    episode_results = pd.concat(all_episode_results, ignore_index=True)

    summary = (
        episode_results.groupby(["encoding", "grid_cols", "grid_rows", "grid_label", "grid_area"], as_index=False)
        .agg(
            episodes=("episode", "count"),
            avg_reward=("reward", "mean"),
            std_reward=("reward", "std"),
            avg_score=("score", "mean"),
            std_score=("score", "std"),
            avg_steps=("steps", "mean"),
            win_rate=("win", "mean"),
            total_wins=("win", "sum"),
        )
        .sort_values(["encoding", "grid_area", "grid_cols", "grid_rows"])
    )

    summary["win_rate"] = summary["win_rate"] * 100.0
    return episode_results, summary


def save_csv_results(episode_results: pd.DataFrame, summary: pd.DataFrame):
    CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    episode_file = CSV_OUTPUT_DIR / f"generalization_episode_results_{timestamp}.csv"
    summary_file = CSV_OUTPUT_DIR / f"generalization_summary_{timestamp}.csv"

    episode_results.to_csv(episode_file, index=False)
    summary.to_csv(summary_file, index=False)

    # Convenience copies with stable names.
    episode_results.to_csv(CSV_OUTPUT_DIR / "generalization_episode_results_latest.csv", index=False)
    summary.to_csv(CSV_OUTPUT_DIR / "generalization_summary_latest.csv", index=False)

    return episode_file, summary_file


def plot_avg_score_vs_grid(summary: pd.DataFrame):
    grid_labels = _ordered_grid_labels(summary)
    x = range(len(grid_labels))

    fig, ax = plt.subplots(figsize=(11, 6))

    for encoding_name in ENCODINGS:
        group = summary[summary["encoding"] == encoding_name]
        if group.empty:
            continue

        sorted_group = group.set_index("grid_label").reindex(grid_labels).reset_index()

        mean_values = sorted_group["avg_score"].to_numpy(dtype=float)
        std_values = sorted_group["std_score"].fillna(0.0).to_numpy(dtype=float)
        color = COLORS.get(encoding_name, None)

        ax.plot(
            x,
            mean_values,
            marker="o",
            linewidth=2,
            label=_encoding_label(encoding_name),
            color=color,
        )
        ax.fill_between(x, mean_values - std_values, mean_values + std_values, color=color, alpha=0.15)

    ax.set_title("Generalization: Average Score Across Grid Sizes")
    ax.set_xlabel("Grid Size")
    ax.set_ylabel("Average Score")
    ax.set_xticks(list(x))
    ax.set_xticklabels(grid_labels)
    ax.legend(loc="best", ncols=2, title="State Encoding")
    fig.tight_layout()
    return fig


def plot_win_rate_vs_grid(summary: pd.DataFrame):
    grid_labels = _ordered_grid_labels(summary)
    x = pd.Series(range(len(grid_labels)), dtype=float)
    present_encodings = [name for name in ENCODINGS if (summary["encoding"] == name).any()]
    bar_width = 0.78 / max(1, len(present_encodings))

    fig, ax = plt.subplots(figsize=(11, 6))

    for idx, encoding_name in enumerate(present_encodings):
        group = summary[summary["encoding"] == encoding_name]
        sorted_group = group.set_index("grid_label").reindex(grid_labels).reset_index()
        win_values = sorted_group["win_rate"].to_numpy(dtype=float)
        bar_positions = x + (idx - (len(present_encodings) - 1) / 2) * bar_width

        ax.bar(
            bar_positions,
            win_values,
            width=bar_width,
            label=_encoding_label(encoding_name),
            color=COLORS.get(encoding_name, None),
            alpha=0.9,
        )

    ax.set_title("Generalization: Win Rate by Grid Size")
    ax.set_xlabel("Grid Size")
    ax.set_ylabel("Win Rate (%)")
    ax.set_xticks(list(range(len(grid_labels))))
    ax.set_xticklabels(grid_labels)
    ax.legend(loc="best", ncols=2, title="State Encoding")
    ax.set_ylim(0, 100)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.35)
    fig.tight_layout()
    return fig


def plot_avg_steps_vs_grid(summary: pd.DataFrame):
    grid_labels = _ordered_grid_labels(summary)
    x = range(len(grid_labels))

    fig, ax = plt.subplots(figsize=(11, 6))

    for encoding_name in ENCODINGS:
        group = summary[summary["encoding"] == encoding_name]
        if group.empty:
            continue

        sorted_group = group.set_index("grid_label").reindex(grid_labels).reset_index()

        ax.plot(
            x,
            sorted_group["avg_steps"],
            marker="o",
            linewidth=2,
            label=_encoding_label(encoding_name),
            color=COLORS.get(encoding_name, None),
        )

    ax.set_title("Generalization: Average Survival Steps")
    ax.set_xlabel("Grid Size")
    ax.set_ylabel("Average Steps")
    ax.set_xticks(list(x))
    ax.set_xticklabels(grid_labels)
    ax.legend(loc="best", ncols=2, title="State Encoding")
    fig.tight_layout()
    return fig


def save_plots(summary: pd.DataFrame):
    PLOTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_plot_style()

    figures = {
        "generalization_1_avg_score_vs_grid.png": plot_avg_score_vs_grid(summary),
        "generalization_2_win_rate_vs_grid.png": plot_win_rate_vs_grid(summary),
        "generalization_3_avg_steps_vs_grid.png": plot_avg_steps_vs_grid(summary),
    }

    for filename, fig in figures.items():
        fig.savefig(PLOTS_OUTPUT_DIR / filename, dpi=150)
        plt.close(fig)


def print_compact_summary(summary: pd.DataFrame):
    print("\n" + "=" * 80)
    print("GENERALIZATION SUMMARY")
    print("=" * 80)

    display_cols = [
        "encoding",
        "grid_label",
        "avg_score",
        "win_rate",
        "avg_steps",
        "avg_reward",
        "total_wins",
        "episodes",
    ]

    formatted = summary[display_cols].copy()
    formatted["avg_score"] = formatted["avg_score"].map(lambda x: f"{x:.2f}")
    formatted["win_rate"] = formatted["win_rate"].map(lambda x: f"{x:.1f}%")
    formatted["avg_steps"] = formatted["avg_steps"].map(lambda x: f"{x:.1f}")
    formatted["avg_reward"] = formatted["avg_reward"].map(lambda x: f"{x:.2f}")

    print(formatted.to_string(index=False))


def main():
    print("\nRunning DQN generalization experiment...")
    print(f"Grid sizes: {GRID_SIZES}")
    print(f"Episodes per grid: {EPISODES_PER_GRID}")

    episode_results, summary = run_generalization_experiment(
        grid_sizes=GRID_SIZES,
        episodes_per_grid=EPISODES_PER_GRID,
    )

    episode_file, summary_file = save_csv_results(episode_results, summary)
    save_plots(summary)
    print_compact_summary(summary)

    print("\nSaved CSV files:")
    print(f"- {episode_file}")
    print(f"- {summary_file}")
    print("\nSaved plot files:")
    print(f"- {PLOTS_OUTPUT_DIR / 'generalization_1_avg_score_vs_grid.png'}")
    print(f"- {PLOTS_OUTPUT_DIR / 'generalization_2_win_rate_vs_grid.png'}")
    print(f"- {PLOTS_OUTPUT_DIR / 'generalization_3_avg_steps_vs_grid.png'}")


if __name__ == "__main__":
    main()
