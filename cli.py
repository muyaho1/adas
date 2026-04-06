#!/usr/bin/env python3
"""Unified CLI for the ADHD Behavioral AI Simulation project.

Usage:
    python cli.py train [--config CONFIG]
    python cli.py evaluate [--config CONFIG] [--episodes N]
    python cli.py export [--results-dir DIR] [--output-dir DIR]
    python cli.py demo [--config CONFIG] [--episodes N]
    python cli.py classroom-demo [--config CONFIG] [--sessions N] [--json-output FILE] [--html-output FILE] [--preview-output FILE] [--mock-backend]
    python cli.py cache-stats
"""
import argparse
import json
import os
import sys


def cmd_train(args):
    """Train the RL agent."""
    from train import build_env, load_config, run_training

    config = load_config(args.config)
    env = build_env(config)
    run_training(config, env)
    cache_stats = env.backend.cache.stats() if hasattr(env.backend, "cache") else {}
    print(f"Cache stats: {cache_stats}")
    print("Training complete.")



def cmd_evaluate(args):
    """Evaluate trained agent vs baselines."""
    from evaluate import run_baseline_comparison
    from train import load_config

    config = load_config(args.config)
    results = run_baseline_comparison(config, n_episodes=args.episodes)
    os.makedirs("results/evaluation", exist_ok=True)
    with open("results/evaluation/comparison.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results/evaluation/comparison.json")



def cmd_export(args):
    """Export results to figures."""
    from export_figures import main as export_main

    sys.argv = [
        "export_figures.py",
        "--results-dir",
        args.results_dir,
        "--output-dir",
        args.output_dir,
    ]
    export_main()



def cmd_demo(args):
    """Run a single demo episode and print the interaction."""
    from src.environment.child_env import ACTIONS
    from train import build_env, load_config

    config = load_config(args.config)
    env = build_env(config)

    model_path = "results/training/ppo_adhd_agent"
    if os.path.exists(model_path + ".zip"):
        from src.agent.ppo_agent import PPOAgent

        agent = PPOAgent.load(model_path, env=env)
        agent_name = "Trained PPO Agent"
    else:
        from src.agent.baselines import RuleBasedAgent

        agent = RuleBasedAgent()
        agent_name = "Rule-Based Agent (no trained model found)"

    print("=== Demo Episode ===")
    print(f"Agent: {agent_name}\n")

    for ep in range(args.episodes):
        obs, info = env.reset()
        print(f"--- Episode {ep + 1} ---")
        print(f"Profile: {env.current_profile.name} ({env.current_profile.severity})")
        print(f"Scenario: {env.current_scenario.name}")
        print(f"Initial: {info['narrative']}")
        if info.get("memory"):
            print(f"Memory: {info['memory']}")
        print("")

        terminated = False
        truncated = False
        total_reward = 0.0

        while not terminated and not truncated:
            if hasattr(agent, "model"):
                action = agent.predict(obs, deterministic=True)
            else:
                action = agent.predict(obs)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            print(f"  Turn {info['turn']}: [{ACTIONS[action]}]")
            print(f"    Child: {info['narrative']}")
            print(
                "    State: "
                f"distress={env.current_state['distress_level']:.2f}, "
                f"compliance={env.current_state['compliance']:.2f}, "
                f"attention={env.current_state['attention']:.2f}, "
                f"escalation={env.current_state['escalation_risk']:.2f}"
            )
            print(f"    Reward: {reward:.3f}")
            if info.get("memory"):
                print(f"    Memory: {info['memory']}")
            print("")

        outcome = (
            "SUCCESS"
            if terminated and env.current_state["compliance"] > env.success_threshold
            else "TIMEOUT/FAILURE"
        )
        print(f"  Result: {outcome} | Total reward: {total_reward:.3f}\n")



def cmd_classroom_demo(args):
    """Run classroom sessions and export a replay log + HTML visualization."""
    from src.agent.baselines import RuleBasedAgent
    from src.agent.ppo_agent import PPOAgent
    from src.simulation.classroom_world import ClassroomWorld
    from src.simulation.mock_demo import ScriptedStudentBackend, export_preview_png
    from src.ui.classroom_replay import export_replay_html
    from train import build_env, load_config

    config = load_config(args.config)

    if args.mock_backend:
        env = build_env(config, backend=ScriptedStudentBackend())
        teacher_policy = RuleBasedAgent()
    else:
        env = build_env(config)
        model_path = "results/training/ppo_adhd_agent"
        if os.path.exists(model_path + ".zip"):
            teacher_policy = PPOAgent.load(model_path, env=env)
        else:
            teacher_policy = RuleBasedAgent()

    world = ClassroomWorld(env=env, teacher_policy=teacher_policy)
    log_data = world.save_sessions(args.json_output, n_sessions=args.sessions)
    export_replay_html(log_data, args.html_output)
    export_preview_png(log_data, args.preview_output)

    print(f"Classroom simulation log saved to {args.json_output}")
    print(f"Replay HTML saved to {args.html_output}")
    print(f"Preview PNG saved to {args.preview_output}")
    if args.mock_backend:
        print("Mode: mock backend")



def cmd_cache_stats(args):
    """Show cache statistics."""
    cache_dir = ".cache/responses"
    if not os.path.exists(cache_dir):
        print("No cache directory found.")
        return
    files = [f for f in os.listdir(cache_dir) if f.endswith(".json")]
    total_size = sum(os.path.getsize(os.path.join(cache_dir, f)) for f in files)
    print(f"Cache entries: {len(files)}")
    print(f"Total size: {total_size / 1024:.1f} KB")
    print(f"Location: {os.path.abspath(cache_dir)}")



def main():
    parser = argparse.ArgumentParser(
        description="ADHD Behavioral AI Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  train          Train the RL agent
  evaluate       Evaluate trained agent vs baselines
  export         Export results to publication figures
  demo           Run demo episodes with interaction logs
  classroom-demo Run Agent-Hospital-style classroom replay sessions
  cache-stats    Show response cache statistics
        """,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_train = subparsers.add_parser("train", help="Train the RL agent")
    p_train.add_argument("--config", default="configs/default.yaml")

    p_eval = subparsers.add_parser("evaluate", help="Evaluate agent vs baselines")
    p_eval.add_argument("--config", default="configs/default.yaml")
    p_eval.add_argument("--episodes", type=int, default=100)

    p_export = subparsers.add_parser("export", help="Export figures")
    p_export.add_argument("--results-dir", default="results")
    p_export.add_argument("--output-dir", default="results/figures")

    p_demo = subparsers.add_parser("demo", help="Run demo episodes")
    p_demo.add_argument("--config", default="configs/default.yaml")
    p_demo.add_argument("--episodes", type=int, default=1)

    p_classroom = subparsers.add_parser(
        "classroom-demo",
        help="Run Agent-Hospital-style classroom replay",
    )
    p_classroom.add_argument("--config", default="configs/default.yaml")
    p_classroom.add_argument("--sessions", type=int, default=3)
    p_classroom.add_argument(
        "--json-output",
        default="results/classroom/classroom_simulation.json",
    )
    p_classroom.add_argument(
        "--html-output",
        default="results/classroom/classroom_replay.html",
    )
    p_classroom.add_argument(
        "--preview-output",
        default="results/classroom/classroom_preview.png",
    )
    p_classroom.add_argument(
        "--mock-backend",
        action="store_true",
        help="Use a deterministic mock student backend instead of live codex exec",
    )

    subparsers.add_parser("cache-stats", help="Show cache stats")

    args = parser.parse_args()
    commands = {
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "export": cmd_export,
        "demo": cmd_demo,
        "classroom-demo": cmd_classroom_demo,
        "cache-stats": cmd_cache_stats,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
