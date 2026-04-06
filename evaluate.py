#!/usr/bin/env python3
"""CLI: Evaluate the trained agent against baselines."""
import argparse
import json
import os
import numpy as np
from src.eval.metrics import EvaluationMetrics
from src.eval.visualize import plot_training_curve, plot_action_heatmap
from src.environment.child_env import ADHDChildEnv, ACTIONS
from src.agent.ppo_agent import PPOAgent
from src.agent.baselines import RandomAgent, RuleBasedAgent, SingleActionAgent
from train import load_config, build_env


def evaluate_agent(agent, env, n_episodes: int = 100, deterministic: bool = True) -> dict:
    metrics = EvaluationMetrics()

    for ep in range(n_episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        episode_actions = []

        while not terminated and not truncated:
            if hasattr(agent, "model"):
                action = agent.predict(obs, deterministic=deterministic)
            else:
                action = agent.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_actions.append(action)

        success = terminated and env.current_state["compliance"] > env.success_threshold
        distress_peak = max(env.distress_history) if env.distress_history else 0.0
        safety_violations = sum(
            1 for h in env.history if h["state"].get("escalation_risk", 0) > 0.8
        )
        distress_monotonic = all(
            env.distress_history[i] >= env.distress_history[i + 1]
            for i in range(len(env.distress_history) - 1)
        ) if len(env.distress_history) > 1 else False

        metrics.record_episode(
            success=success,
            turns=len(episode_actions),
            distress_peak=distress_peak,
            safety_violations=safety_violations,
            distress_monotonic=distress_monotonic,
        )
        metrics.record_actions(episode_actions)

    return metrics.summary()


def run_baseline_comparison(config: dict, n_episodes: int = 100):
    results = {}
    baselines = {
        "random": RandomAgent(n_actions=12),
        "rule_based": RuleBasedAgent(),
        "single_action_best": SingleActionAgent(action=0),
    }

    for name, agent in baselines.items():
        env = build_env(config)
        summary = evaluate_agent(agent, env, n_episodes=n_episodes)
        results[name] = summary
        print(f"{name}: {summary}")

    model_path = "results/training/ppo_adhd_agent"
    if os.path.exists(model_path + ".zip"):
        env = build_env(config)
        ppo = PPOAgent.load(model_path, env=env)
        summary = evaluate_agent(ppo, env, n_episodes=n_episodes)
        results["ppo_trained"] = summary
        print(f"ppo_trained: {summary}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained agent vs baselines")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()

    config = load_config(args.config)
    results = run_baseline_comparison(config, n_episodes=args.episodes)

    os.makedirs("results/evaluation", exist_ok=True)
    with open("results/evaluation/comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results/evaluation/comparison.json")


if __name__ == "__main__":
    main()
