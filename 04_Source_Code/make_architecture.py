"""Draws the system-architecture and agent-interaction diagrams used in the report and slides."""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def box(ax, x, y, w, h, text, fc, fs=9, ec="#1c2b4a"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="#111")


def arrow(ax, p1, p2, text="", rad=0.0, fs=8):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=13, lw=1.4, color="#333",
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2 + 0.12, text, ha="center", fontsize=fs, color="#333")


def architecture(out):
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 11); ax.set_ylim(0, 4.6); ax.axis("off")
    box(ax, 0.1, 1.6, 2.3, 1.5, "Demand generator\n(5 scenarios, seeded,\nidentical for all controllers)", "#dfe9f7")
    box(ax, 3.0, 0.5, 3.0, 3.6, "", "#f4f4f4")
    ax.text(4.5, 3.85, "Traffic simulator (traffic_env.py)", ha="center", fontsize=10, fontweight="bold")
    box(ax, 3.2, 2.3, 2.6, 1.2, "3x3 grid, 9 signals\nqueues + link travel\nspill-back, lost time", "#fff2cc")
    box(ax, 3.2, 0.7, 2.6, 1.2, "Metrics: waiting, queue,\nthroughput, travel time,\ncongestion", "#fff2cc")
    box(ax, 6.7, 2.5, 2.2, 1.4, "9 Q-learning agents\n(agents.py)\none Q-table each", "#f8d7da")
    box(ax, 6.7, 0.5, 2.2, 1.4, "Baselines: Random,\nFixed-time,\nLongest-queue-first", "#e2f0d9")
    box(ax, 9.3, 1.5, 1.6, 1.6, "train.py\nevaluate.py\nmake_results.py\nvisualize.py", "#e8e0f5", fs=8.5)
    arrow(ax, (2.4, 2.35), (3.2, 2.85), "arrivals")
    arrow(ax, (5.8, 3.25), (6.7, 3.25), "state s_i, reward r_i")
    arrow(ax, (6.7, 2.75), (5.8, 2.75), "action a_i (phase)")
    arrow(ax, (6.7, 1.2), (5.8, 2.3), "")
    arrow(ax, (8.9, 3.1), (9.3, 2.8)); arrow(ax, (8.9, 1.2), (9.3, 1.8))
    arrow(ax, (4.5, 2.3), (4.5, 1.9))
    plt.savefig(out, dpi=170, bbox_inches="tight"); plt.close()


def interaction(out):
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.set_xlim(0, 6.2); ax.set_ylim(0, 3.6); ax.axis("off")
    pos = {"A": (0.4, 1.3), "B": (2.5, 1.3), "C": (4.6, 1.3)}
    for k, (x, y) in pos.items():
        box(ax, x, y, 1.3, 1.0, f"Agent {k}\nQ-table {k}", "#f8d7da")
    arrow(ax, (1.7, 1.95), (2.5, 1.95), "downstream\noccupancy", fs=7)
    arrow(ax, (2.5, 1.65), (1.7, 1.65), "", fs=7)
    arrow(ax, (3.8, 1.95), (4.6, 1.95), "", fs=7); arrow(ax, (4.6, 1.65), (3.8, 1.65))
    ax.text(3.1, 0.75, "shared reward: r~_i = 0.7 r_i + 0.3 * mean(r_neighbours)", ha="center", fontsize=8)
    ax.text(3.1, 3.0, "Neighbouring agents exchange information;\nlearning stays decentralised (no central Q-function)", ha="center", fontsize=8.5)
    plt.savefig(out, dpi=170, bbox_inches="tight"); plt.close()


if __name__ == "__main__":
    rd = sys.argv[1] if len(sys.argv) > 1 else os.path.join("..", "05_Experimental_Results")
    architecture(os.path.join(rd, "architecture_diagram.png"))
    interaction(os.path.join(rd, "agent_interaction_diagram.png"))
