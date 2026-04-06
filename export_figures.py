#!/usr/bin/env python3
"""CLI: Export results to publication-ready figures."""
import argparse
import json
import os
import numpy as np
from src.eval.visualize import plot_training_curve, plot_action_heatmap, plot_policy_heatmap
from src.environment.child_env import ACTIONS


def main():
    parser = argparse.ArgumentParser(description="Export results to publication figures")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="results/figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Training curve
    rewards_path = os.path.join(args.results_dir, "training", "episode_rewards.json")
    if os.path.exists(rewards_path):
        with open(rewards_path) as f:
            rewards = json.load(f)
        plot_training_curve(rewards, os.path.join(args.output_dir, "training_curve.png"))
        print("Exported training_curve.png")

    # Comparison results
    comparison_path = os.path.join(args.results_dir, "evaluation", "comparison.json")
    if os.path.exists(comparison_path):
        with open(comparison_path) as f:
            results = json.load(f)
        print(f"Comparison results loaded: {list(results.keys())}")

    print("Figure export complete.")


if __name__ == "__main__":
    main()
