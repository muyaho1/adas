import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def plot_training_curve(rewards: list[float], output_path: str, window: int = 50) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(rewards, alpha=0.3, label="Raw")
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(rewards)), smoothed, label=f"Rolling avg ({window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Training Curve")
    ax.legend()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_action_heatmap(action_counts: dict, action_names: list[str], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    counts = [action_counts.get(i, 0) for i in range(len(action_names))]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.barh(action_names, counts)
    ax.set_xlabel("Frequency")
    ax.set_title("Action Distribution")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_policy_heatmap(
    distress_bins: np.ndarray,
    compliance_bins: np.ndarray,
    action_grid: np.ndarray,
    action_names: list[str],
    output_path: str,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        action_grid,
        xticklabels=[f"{c:.1f}" for c in compliance_bins],
        yticklabels=[f"{d:.1f}" for d in distress_bins],
        annot=True,
        fmt="d",
        cmap="viridis",
        ax=ax,
    )
    ax.set_xlabel("Compliance")
    ax.set_ylabel("Distress")
    ax.set_title("Policy: Preferred Action by State")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
