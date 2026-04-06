#!/usr/bin/env python3
"""CLI: Train the ADHD behavioral intervention RL agent."""
import argparse
import os

import yaml

from src.agent.ppo_agent import PPOAgent
from src.environment.child_env import ADHDChildEnv
from src.environment.child_profiles import load_profiles
from src.environment.scenarios import load_scenarios
from src.llm.backend import LLMBackend
from src.llm.claude_code_backend import ClaudeCodeBackend
from src.llm.codex_cli_backend import CodexCLIBackend
from src.reward.reward_function import RewardFunction
from src.simulation.memory import MemoryStore


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_reward(config: dict) -> RewardFunction:
    rc = config["reward"]
    return RewardFunction(
        w_compliance=rc["w_compliance"],
        w_attention=rc["w_attention"],
        w_distress=rc["w_distress"],
        w_efficiency=rc["w_efficiency"],
        safety_threshold=rc["safety_threshold"],
        safety_penalty=rc["safety_penalty"],
        bonus_success=rc["bonus_success"],
        bonus_fast=rc["bonus_fast"],
        bonus_gentle=rc["bonus_gentle"],
        bonus_monotonic=rc["bonus_monotonic"],
        penalty_failure=rc["penalty_failure"],
        gentle_threshold=rc["gentle_threshold"],
        fast_threshold=rc["fast_threshold"],
    )


def build_backend(config: dict) -> LLMBackend:
    lc = config["llm"]
    backend_name = lc.get("backend", "codex_cli")
    backend_kwargs = {
        "cache_dir": lc["cache_dir"],
        "cache_enabled": lc["cache_enabled"],
        "retry_attempts": lc["retry_attempts"],
        "retry_delay": lc["retry_delay"],
    }

    if backend_name == "codex_cli":
        return CodexCLIBackend(
            model=lc.get("model"),
            command=lc.get("command", "codex"),
            timeout=lc.get("timeout", 180),
            cwd=lc.get("cwd"),
            **backend_kwargs,
        )
    if backend_name == "claude_code":
        return ClaudeCodeBackend(**backend_kwargs)

    raise ValueError(f"Unsupported LLM backend: {backend_name}")


def build_env(
    config: dict,
    backend: LLMBackend | None = None,
    memory_store: MemoryStore | None = None,
) -> ADHDChildEnv:
    if backend is None:
        backend = build_backend(config)
    ec = config["environment"]
    profiles = load_profiles("data/profiles/adhd_profiles.yaml")
    scenarios = load_scenarios("data/scenarios/task_transitions.yaml")
    reward_fn = build_reward(config)
    return ADHDChildEnv(
        backend=backend,
        profiles=profiles,
        scenarios=scenarios,
        max_turns=ec["max_turns"],
        success_threshold=ec["success_threshold"],
        failure_distress=ec["failure_distress"],
        failure_consecutive=ec["failure_consecutive"],
        reward_fn=reward_fn,
        memory_store=memory_store or MemoryStore(),
        use_constrained_transitions=ec.get("use_constrained_transitions", True),
        use_memory_adjusted_reset=ec.get("use_memory_adjusted_reset", True),
    )


def run_training(config: dict, env: ADHDChildEnv) -> PPOAgent:
    ac = config["agent"]
    agent = PPOAgent(
        env=env,
        hidden_sizes=ac["hidden_sizes"],
        learning_rate=ac["learning_rate"],
        n_steps=ac["n_steps"],
        batch_size=ac["batch_size"],
        n_epochs=ac["n_epochs"],
        gamma=ac["gamma"],
    )
    agent.train(total_timesteps=ac["total_timesteps"])

    os.makedirs("results/training", exist_ok=True)
    try:
        agent.save("results/training/ppo_adhd_agent")
        print("Model saved to results/training/ppo_adhd_agent")
    except PermissionError as exc:
        print(f"Warning: could not overwrite results/training/ppo_adhd_agent.zip ({exc})")
    return agent


def main():
    parser = argparse.ArgumentParser(description="Train ADHD behavioral intervention agent")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    args = parser.parse_args()

    config = load_config(args.config)
    env = build_env(config)
    run_training(config, env)

    cache_stats = env.backend.cache.stats() if hasattr(env.backend, "cache") else {}
    print(f"Cache stats: {cache_stats}")
    print("Training complete.")


if __name__ == "__main__":
    main()

