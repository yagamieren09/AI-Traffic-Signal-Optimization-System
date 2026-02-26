"""
Evaluation script: compares trained RL agent vs fixed-time baseline.
Uses identical seeds and episode lengths for fairness.
"""

import numpy as np
import os, sys, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.traffic_env import TrafficEnv
from agent.q_agent import QLearningAgent
from agent.fixed_time import FixedTimeController

ARRIVAL_RATE    = 0.8
EVAL_EPISODES   = 10
EPISODE_STEPS   = 200
SEED_BASE       = 42   # same base as training for fair comparison


def run_episode(env: TrafficEnv, controller, greedy: bool = True) -> dict:
    obs = env.reset()
    if hasattr(controller, "reset"):
        controller.reset()
    total_reward = 0.0

    for _ in range(EPISODE_STEPS):
        if isinstance(controller, QLearningAgent):
            action = controller.select_action(obs, greedy=greedy)
        else:
            action = controller.select_action(obs)
        obs, reward, _ = env.step(action)
        total_reward += reward

    metrics = env.get_metrics()
    metrics["total_reward"] = total_reward
    return metrics


def evaluate(agent: QLearningAgent = None, verbose: bool = True) -> dict:
    if agent is None:
        # standalone: retrain quickly for demo
        from training.train import train
        _, agent = train(verbose=False)

    results = {"rl": [], "fixed": []}

    for ep in range(EVAL_EPISODES):
        seed = SEED_BASE + ep

        # RL agent
        env_rl = TrafficEnv(arrival_rate=ARRIVAL_RATE, seed=seed)
        m_rl   = run_episode(env_rl, agent, greedy=True)
        results["rl"].append(m_rl)

        # Fixed-time baseline (same seed)
        env_fx = TrafficEnv(arrival_rate=ARRIVAL_RATE, seed=seed)
        m_fx   = run_episode(env_fx, FixedTimeController(cycle_half=10))
        results["fixed"].append(m_fx)

    def summarize(runs: list[dict], key: str):
        vals = [r[key] for r in runs]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    summary = {}
    for method in ["rl", "fixed"]:
        summary[method] = {
            "reward":     summarize(results[method], "total_reward"),
            "wait":       summarize(results[method], "total_wait"),
            "queue":      summarize(results[method], "avg_queue_per_intersection"),
            "throughput": summarize(results[method], "total_throughput"),
        }

    # percentage improvement
    def pct_improve(key, higher_better=True):
        rl_m  = summary["rl"][key]["mean"]
        fx_m  = summary["fixed"][key]["mean"]
        if fx_m == 0:
            return 0.0
        diff  = (rl_m - fx_m) / abs(fx_m) * 100
        return diff if higher_better else -diff

    summary["improvement"] = {
        "wait_reduction_pct":       pct_improve("wait",       higher_better=False),
        "queue_reduction_pct":      pct_improve("queue",      higher_better=False),
        "throughput_increase_pct":  pct_improve("throughput", higher_better=True),
    }

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print("\n===== EVALUATION RESULTS =====")
        for method in ["rl", "fixed"]:
            label = "RL Agent  " if method == "rl" else "Fixed-Time"
            s = summary[method]
            print(f"\n{label}:")
            print(f"  Reward      : {s['reward']['mean']:8.2f}  ±{s['reward']['std']:.2f}")
            print(f"  Total Wait  : {s['wait']['mean']:8.1f}  ±{s['wait']['std']:.1f}")
            print(f"  Avg Queue   : {s['queue']['mean']:8.3f}  ±{s['queue']['std']:.3f}")
            print(f"  Throughput  : {s['throughput']['mean']:8.1f}  ±{s['throughput']['std']:.1f}")

        imp = summary["improvement"]
        print("\n===== IMPROVEMENT (RL vs Fixed) =====")
        print(f"  Wait reduction   : {imp['wait_reduction_pct']:+.1f}%")
        print(f"  Queue reduction  : {imp['queue_reduction_pct']:+.1f}%")
        print(f"  Throughput gain  : {imp['throughput_increase_pct']:+.1f}%")

    return summary, results


if __name__ == "__main__":
    evaluate()
