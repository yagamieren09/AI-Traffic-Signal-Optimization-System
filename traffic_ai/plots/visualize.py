"""
Plot training history and evaluation comparison charts.
"""

import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


def smooth(arr, window=10):
    if len(arr) < window:
        return arr
    return np.convolve(arr, np.ones(window)/window, mode="valid")


def plot_training(history: dict):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Training History – RL Traffic Agent", fontsize=14, fontweight="bold")

    # Reward
    ax = axes[0, 0]
    raw = history["episode_reward"]
    ax.plot(raw, alpha=0.3, color="steelblue", label="Raw")
    ax.plot(np.arange(len(smooth(raw))), smooth(raw), color="steelblue", linewidth=2, label="Smoothed")
    ax.set_title("Episode Reward")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.legend()
    ax.grid(alpha=0.3)

    # Average Wait
    ax = axes[0, 1]
    aw = history["avg_wait"]
    ax.plot(aw, alpha=0.3, color="tomato")
    ax.plot(np.arange(len(smooth(aw))), smooth(aw), color="tomato", linewidth=2)
    ax.set_title("Average Wait Time per Step")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Wait")
    ax.grid(alpha=0.3)

    # Queue
    ax = axes[1, 0]
    aq = history["avg_queue"]
    ax.plot(aq, alpha=0.3, color="darkorange")
    ax.plot(np.arange(len(smooth(aq))), smooth(aq), color="darkorange", linewidth=2)
    ax.set_title("Avg Queue per Intersection")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Queue Length")
    ax.grid(alpha=0.3)

    # Epsilon
    ax = axes[1, 1]
    ax.plot(history["epsilon"], color="purple", linewidth=2)
    ax.set_title("Exploration (Epsilon) Decay")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "training_history.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_evaluation(summary: dict):
    metrics   = ["wait", "queue", "throughput"]
    labels    = ["Total Wait Time", "Avg Queue Length", "Throughput (vehicles)"]
    colors_rl = "#2196F3"
    colors_fx = "#FF5722"

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("RL Agent vs Fixed-Time Baseline", fontsize=14, fontweight="bold")

    for ax, metric, label in zip(axes, metrics, labels):
        rl_mean  = summary["rl"][metric]["mean"]
        rl_std   = summary["rl"][metric]["std"]
        fx_mean  = summary["fixed"][metric]["mean"]
        fx_std   = summary["fixed"][metric]["std"]

        x      = [0, 1]
        means  = [rl_mean, fx_mean]
        stds   = [rl_std,  fx_std]
        colors = [colors_rl, colors_fx]

        bars = ax.bar(x, means, yerr=stds, capsize=8, color=colors,
                      edgecolor="black", linewidth=0.8, width=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(["RL Agent", "Fixed-Time"])
        ax.set_title(label)
        ax.set_ylabel("Value")
        ax.grid(axis="y", alpha=0.3)

        # annotate values
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stds)*0.05,
                    f"{mean:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "evaluation_comparison.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_improvement(summary: dict):
    imp = summary["improvement"]
    keys   = list(imp.keys())
    labels = ["Wait\nReduction %", "Queue\nReduction %", "Throughput\nIncrease %"]
    values = [imp[k] for k in keys]
    colors = ["green" if v > 0 else "red" for v in values]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.8, width=0.5)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("RL Improvement over Fixed-Time Baseline (%)", fontsize=13, fontweight="bold")
    ax.set_ylabel("% Improvement")
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, values):
        ypos = val + (1 if val >= 0 else -3)
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                f"{val:+.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "improvement.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def generate_all(history: dict, summary: dict):
    plot_training(history)
    plot_evaluation(summary)
    plot_improvement(summary)
    print("All plots saved to ./plots/")


if __name__ == "__main__":
    with open("outputs/training_history.json") as f:
        history = json.load(f)
    with open("outputs/eval_summary.json") as f:
        summary = json.load(f)
    generate_all(history, summary)
