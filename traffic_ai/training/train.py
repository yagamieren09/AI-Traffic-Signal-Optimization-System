"""
Training loop for Q-learning agent on the traffic environment.
"""

import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.traffic_env import TrafficEnv
from agent.q_agent import QLearningAgent

# ─── Hyper-parameters ─────────────────────────────────────────────────────────
ARRIVAL_RATE   = 0.8
TRAIN_EPISODES = 600
EPISODE_STEPS  = 200
SEED           = 42

np.random.seed(SEED)


def train(arrival_rate: float = ARRIVAL_RATE, verbose: bool = True) -> dict:
    env   = TrafficEnv(arrival_rate=arrival_rate, seed=SEED)
    agent = QLearningAgent(seed=SEED)

    history = {
        "episode_reward":  [],
        "avg_wait":        [],
        "avg_queue":       [],
        "throughput":      [],
        "epsilon":         [],
        "states_visited":  [],
    }

    for ep in range(TRAIN_EPISODES):
        obs = env.reset()
        agent.update_lr(ep, TRAIN_EPISODES)
        ep_reward = 0.0

        for _ in range(EPISODE_STEPS):
            action   = agent.select_action(obs)
            next_obs, reward, _ = env.step(action)
            agent.update(obs, action, reward, next_obs)
            obs       = next_obs
            ep_reward += reward

        agent.decay_epsilon()
        metrics = env.get_metrics()

        history["episode_reward"].append(ep_reward)
        history["avg_wait"].append(metrics["total_wait"] / max(EPISODE_STEPS, 1))
        history["avg_queue"].append(metrics["avg_queue_per_intersection"])
        history["throughput"].append(metrics["total_throughput"])
        history["epsilon"].append(agent.eps)
        history["states_visited"].append(agent.num_states_visited())

        if verbose and (ep + 1) % 50 == 0:
            print(
                f"Ep {ep+1:4d}/{TRAIN_EPISODES}  "
                f"reward={ep_reward:7.2f}  "
                f"wait={history['avg_wait'][-1]:.1f}  "
                f"queue={history['avg_queue'][-1]:.2f}  "
                f"eps={agent.eps:.3f}  "
                f"states={agent.num_states_visited()}"
            )

    # Save Q-table metadata
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. States visited: {agent.num_states_visited()}")
    return history, agent


if __name__ == "__main__":
    train()
