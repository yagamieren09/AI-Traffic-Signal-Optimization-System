"""
main.py – Run full pipeline: train → evaluate → visualize → summarize
"""

import os
import sys
import numpy as np

# ensure project root is on path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
os.makedirs(os.path.join(ROOT, "plots"),   exist_ok=True)


def main():
    print("=" * 60)
    print("  AI Traffic Signal Optimization — Full Pipeline")
    print("=" * 60)

    # ── 1. Train ──────────────────────────────────────────────────
    print("\n[1/3] Training Q-Learning Agent...")
    os.chdir(ROOT)
    from training.train import train
    history, agent = train(verbose=True)

    # ── 2. Evaluate ───────────────────────────────────────────────
    print("\n[2/3] Evaluating Agent vs Fixed-Time Baseline...")
    from evaluation.evaluate import evaluate
    summary, _ = evaluate(agent=agent, verbose=True)

    # ── 3. Visualize ──────────────────────────────────────────────
    print("\n[3/3] Generating Plots...")
    from plots.visualize import generate_all
    generate_all(history, summary)

    # ── 4. Final Summary ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL PERFORMANCE SUMMARY")
    print("=" * 60)
    imp = summary["improvement"]
    print(f"  ✓ Wait Time Reduction   : {imp['wait_reduction_pct']:+.1f}%")
    print(f"  ✓ Queue Reduction       : {imp['queue_reduction_pct']:+.1f}%")
    print(f"  ✓ Throughput Increase   : {imp['throughput_increase_pct']:+.1f}%")
    print(f"  ✓ Q-table states learned: {agent.num_states_visited()}")
    print(f"  ✓ Final epsilon         : {agent.eps:.4f}")
    print("\nOutputs saved to: outputs/ and plots/")
    print("=" * 60)


if __name__ == "__main__":
    main()
