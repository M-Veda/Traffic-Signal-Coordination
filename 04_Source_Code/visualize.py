"""Animation of vehicles and traffic lights (matplotlib). Produces a GIF (and optionally an MP4 / live window).

  python visualize.py --controller coop --scenario asym --start 600 --duration 120
Controllers: fixed | random | lqf | iql | coop   (iql/coop load ../05_Experimental_Results/trained_q_tables.pkl)
"""
import argparse, os, pickle
import numpy as np
import matplotlib
import config as C
from traffic_env import TrafficNetwork, generate_arrivals, DELTA, PHASE_DIRS
from agents import FixedTimeController, RandomController, LongestQueueController, MultiAgentQ

L, STOP, SPACING, LANE = 4.0, 0.7, 0.28, 0.22
DIR_COLORS = {0: "#1f77b4", 1: "#17becf", 2: "#ff7f0e", 3: "#d62728"}


def make_controller(kind, results_dir, env):
    if kind == "fixed": return FixedTimeController()
    if kind == "random": return RandomController(0)
    if kind == "lqf": return LongestQueueController()
    coop = kind == "coop"
    with open(os.path.join(results_dir, "trained_q_tables.pkl"), "rb") as f:
        Q = pickle.load(f)["Coop-IQL (proposed)" if coop else "IQL"][0]
    a = MultiAgentQ(env, coop, coop_info="downstream"); a.Q = Q; a.eps = 0.0
    return a


def record(kind, scenario, seed, start, duration, results_dir):
    env = TrafficNetwork()
    env.reset(generate_arrivals(scenario, seed))
    ctrl = make_controller(kind, results_dir, env)
    ctrl.reset(env)
    frames = []
    done = False
    while not done and env.t < start + duration:
        acts = ctrl.act(env)
        _, done = env.step(acts, per_second_callback=(lambda e: frames.append(e.snapshot()) if e.t >= start else None))
    return frames, env


def node_xy(i):
    r, c = divmod(i, C.COLS)
    return np.array([c * L, -r * L])


def veh_xy(i, d, prog, k):
    dr, dc = DELTA[d]
    mv = np.array([dc, -dr], float)               # movement vector in plot coordinates
    perp = np.array([-mv[1], mv[0]]) * -LANE      # keep to the right-hand side
    stop = node_xy(i) - mv * STOP
    if k >= 0:
        return stop - mv * SPACING * k + perp
    start = node_xy(i) - mv * L
    return start + (stop - start) * prog + perp


def draw(frames, title, out_path, fps=8, step=2):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    fig, ax = plt.subplots(figsize=(7, 7))
    pad = L * 0.9
    ax.set_xlim(-pad, (C.COLS - 1) * L + pad); ax.set_ylim(-(C.ROWS - 1) * L - pad, pad)
    ax.set_aspect("equal"); ax.axis("off")
    for i in range(C.ROWS * C.COLS):
        x, y = node_xy(i)
        ax.plot([x - L * .85, x + L * .85], [y, y], color="#bbbbbb", lw=9, solid_capstyle="butt", zorder=1)
        ax.plot([x, x], [y - L * .85, y + L * .85], color="#bbbbbb", lw=9, solid_capstyle="butt", zorder=1)
    veh_sc = ax.scatter([], [], s=26, zorder=3)
    lights = {}
    for i in range(C.ROWS * C.COLS):
        for d in range(4):
            dr, dc = DELTA[d]; mv = np.array([dc, -dr], float)
            p = node_xy(i) - mv * (STOP + 0.15) + np.array([-mv[1], mv[0]]) * -(LANE + 0.5)
            lights[(i, d)] = ax.scatter([p[0]], [p[1]], s=55, c="red", edgecolors="k", zorder=4)
    txt = ax.text(0.5, 1.0, "", transform=ax.transAxes, ha="center", va="bottom", fontsize=11)
    ax.set_title(title, fontsize=12, pad=22)
    frs = frames[::step]

    def upd(k):
        f = frs[k]
        pts, cols = [], []
        for (i, d, prog, q) in f["veh"]:
            pts.append(veh_xy(i, d, prog, q)); cols.append(DIR_COLORS[d])
        veh_sc.set_offsets(np.array(pts) if pts else np.empty((0, 2))); veh_sc.set_color(cols)
        for (i, d), sc in lights.items():
            if f["lost"][i] > 0: sc.set_color("orange")
            elif d in PHASE_DIRS[f["phase"][i]]: sc.set_color("limegreen")
            else: sc.set_color("red")
        txt.set_text(f"t = {f['t']} s   vehicles on road: {len(f['veh'])}   (green=go, red=stop, orange=changeover)")
        return []
    anim = FuncAnimation(fig, upd, frames=len(frs), interval=1000 / fps, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    upd(len(frs) // 2); fig.savefig(out_path.replace(".gif", "_snapshot.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    matplotlib.use("Agg")
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", default="coop", choices=["fixed", "random", "lqf", "iql", "coop"])
    ap.add_argument("--scenario", default="asym", choices=C.SCENARIOS)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--start", type=int, default=600)
    ap.add_argument("--duration", type=int, default=120)
    ap.add_argument("--results_dir", default=os.path.join("..", "05_Experimental_Results"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    frames, _ = record(a.controller, a.scenario, a.seed, a.start, a.duration, a.results_dir)
    out = a.out or os.path.join(a.results_dir, f"simulation_{a.controller}_{a.scenario}.gif")
    draw(frames, f"3x3 signalised grid - {a.controller} control, {a.scenario} demand", out)
    print("saved", out)
