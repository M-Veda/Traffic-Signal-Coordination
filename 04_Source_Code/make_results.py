"""Turns raw training logs + evaluation results into charts and tables (all numbers come from the experiments)."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C
from traffic_env import TrafficNetwork, generate_arrivals
from agents import FixedTimeController, LongestQueueController, RandomController
from evaluate import run_episode
from train import VAL_SEEDS

ORDER = ["Random", "Fixed-time", "Longest-queue-first", "IQL", "Coop-IQL (proposed)"]
COLORS = {"Random": "#999999", "Fixed-time": "#1f77b4", "Longest-queue-first": "#2ca02c",
          "IQL": "#ff7f0e", "Coop-IQL (proposed)": "#d62728"}
LABELS = {"avg_wait_s": "Average vehicle waiting time (s)", "avg_queue_len": "Average queue length (vehicles / approach)",
          "throughput": "Throughput (vehicles completed in 30 min)", "avg_travel_time_s": "Average travel time (s)",
          "congestion_index": "Congestion index (share of approaches with >= 10 queued)",
          "spillback_events": "Spill-back events (blocked discharges)"}


def smooth(x, k=20):
    x = np.asarray(x, float)
    out = np.convolve(x, np.ones(k) / k, mode="valid")
    return np.concatenate([np.full(k - 1, np.nan), out])


def learning_curves(logs, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for name in ("IQL", "Coop-IQL (proposed)"):
        R = np.array([l["ep_return"] for l in logs[name]])        # seeds x episodes
        S = np.array([smooth(r) for r in R])
        m, sd = np.nanmean(S, 0), np.nanstd(S, 0)
        x = np.arange(1, R.shape[1] + 1)
        axes[0].plot(x, m, color=COLORS[name], label=name); axes[0].fill_between(x, m - sd, m + sd, color=COLORS[name], alpha=.2)
    axes[0].set_title("Learning curve: episode return during training\n(mean of 5 seeds +/- 1 std, 20-episode moving average)")
    axes[0].set_xlabel("Training episode"); axes[0].set_ylabel("Episode return (sum over time of mean per-agent reward)")
    axes[0].legend(); axes[0].grid(alpha=.3)
    for name in ("IQL", "Coop-IQL (proposed)"):
        V = np.array([l["val_wait"] for l in logs[name]]); x = logs[name][0]["val_ep"]
        m, sd = V.mean(0), V.std(0)
        axes[1].plot(x, m, "o-", color=COLORS[name], label=name); axes[1].fill_between(x, m - sd, m + sd, color=COLORS[name], alpha=.2)
    env = TrafficNetwork()
    for bname, ctrl in (("Fixed-time", FixedTimeController()), ("Longest-queue-first", LongestQueueController())):
        w = np.mean([run_episode(env, ctrl, generate_arrivals(sc, s))["avg_wait_s"] for sc in C.SCENARIOS for s in VAL_SEEDS])
        axes[1].axhline(w, color=COLORS[bname], ls="--", label=f"{bname} (baseline)")
    axes[1].set_ylim(28, 70)
    axes[1].set_title("Greedy validation waiting time during training\n(5 scenarios x 2 held-out seeds; mean of 5 runs +/- 1 std)")
    axes[1].set_xlabel("Training episode"); axes[1].set_ylabel("Average vehicle waiting time (s)")
    axes[1].legend(); axes[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


def aggregate(res):
    """overall[name][metric] = (mean, std) over all scenario x seed x instance runs; per[name][sc][metric] likewise."""
    metrics = ["avg_wait_s", "avg_queue_len", "throughput", "avg_travel_time_s", "congestion_index",
               "spillback_events", "vehicles_left_in_network"]
    overall, per = {}, {}
    for n in ORDER:
        overall[n], per[n] = {}, {}
        for m in metrics:
            per_sc = []
            for sc in C.SCENARIOS:
                v = np.array([r[m] for r in res[n][sc]], float)
                per[n].setdefault(sc, {})[m] = (v.mean(), v.std())
                per_sc.append(v)
            allv = np.concatenate(per_sc)
            overall[n][m] = (allv.mean(), allv.std())
    return overall, per


def bar_charts(overall, per, outdir):
    # 5 overall metric panels
    ms = ["avg_wait_s", "avg_queue_len", "throughput", "avg_travel_time_s", "congestion_index", "spillback_events"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, m in zip(axes.ravel(), ms):
        vals = [overall[n][m][0] for n in ORDER]; errs = [overall[n][m][1] for n in ORDER]
        bars = ax.bar(range(len(ORDER)), vals, color=[COLORS[n] for n in ORDER])
        for b_, v_ in zip(bars, vals):
            ax.text(b_.get_x() + b_.get_width() / 2, v_, f"{v_:.3g}" if v_ < 10 else f"{v_:.0f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(ORDER))); ax.set_xticklabels([n.replace(" (proposed)", "\n(proposed)").replace("Longest-queue-first", "Longest-\nqueue-first") for n in ORDER], fontsize=8)
        ax.set_ylabel(LABELS[m], fontsize=8); ax.set_title(LABELS[m].split(" (")[0], fontsize=10); ax.grid(axis="y", alpha=.3)
    fig.suptitle("Controller comparison averaged over 5 demand scenarios x 10 unseen seeds (bars = mean over all test runs)")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "comparison_overall.png"), dpi=150); plt.close()
    # per-scenario grouped bars for the 3 headline metrics
    for m, fn in (("avg_wait_s", "waiting_time_by_scenario.png"), ("avg_queue_len", "queue_length_by_scenario.png"),
                  ("throughput", "throughput_by_scenario.png"), ("avg_travel_time_s", "travel_time_by_scenario.png")):
        fig, ax = plt.subplots(figsize=(10, 4.6)); w = 0.16
        for k, n in enumerate(ORDER):
            v = [per[n][sc][m][0] for sc in C.SCENARIOS]; e = [per[n][sc][m][1] for sc in C.SCENARIOS]
            ax.bar(np.arange(5) + (k - 2) * w, v, w, yerr=e, capsize=2, color=COLORS[n], label=n)
        ax.set_xticks(range(5)); ax.set_xticklabels([f"{s}" for s in C.SCENARIOS]); ax.set_xlabel("Demand scenario")
        ax.set_ylabel(LABELS[m]); ax.set_title(LABELS[m].split(" (")[0] + " by demand scenario (mean +/- std over test runs)")
        if m == "throughput":
            lo = min(per[n][sc][m][0] for n in ORDER for sc in C.SCENARIOS); ax.set_ylim(lo * 0.8, None)
        ax.legend(fontsize=8, ncol=2); ax.grid(axis="y", alpha=.3)
        plt.tight_layout(); plt.savefig(os.path.join(outdir, fn), dpi=150); plt.close()


def tables(overall, per, res, outdir):
    lines = ["| Controller | Avg wait (s) | Avg queue (veh/approach) | Throughput (veh) | Avg travel time (s) | Congestion index | Spill-back events |",
             "|---|---|---|---|---|---|---|"]
    csv = ["controller,scenario,avg_wait_s,avg_queue_len,throughput,avg_travel_time_s,congestion_index,spillback_events"]
    for n in ORDER:
        o = overall[n]
        lines.append(f"| {n} | {o['avg_wait_s'][0]:.1f} ± {o['avg_wait_s'][1]:.1f} | {o['avg_queue_len'][0]:.2f} ± {o['avg_queue_len'][1]:.2f} | "
                     f"{o['throughput'][0]:.0f} ± {o['throughput'][1]:.0f} | {o['avg_travel_time_s'][0]:.1f} ± {o['avg_travel_time_s'][1]:.1f} | "
                     f"{o['congestion_index'][0]:.3f} ± {o['congestion_index'][1]:.3f} | {o['spillback_events'][0]:.0f} ± {o['spillback_events'][1]:.0f} |")
        for sc in C.SCENARIOS:
            p = per[n][sc]
            csv.append(f"{n},{sc}," + ",".join(f"{p[m][0]:.4f}" for m in ["avg_wait_s", "avg_queue_len", "throughput", "avg_travel_time_s", "congestion_index", "spillback_events"]))
    open(os.path.join(outdir, "results_overall.md"), "w").write("\n".join(lines) + "\n")
    open(os.path.join(outdir, "results_by_scenario.csv"), "w").write("\n".join(csv) + "\n")
    # per-scenario wait table
    w = ["| Controller | " + " | ".join(C.SCENARIOS) + " |", "|---|" + "---|" * 5]
    for n in ORDER:
        w.append(f"| {n} | " + " | ".join(f"{per[n][sc]['avg_wait_s'][0]:.1f}" for sc in C.SCENARIOS) + " |")
    open(os.path.join(outdir, "results_wait_by_scenario.md"), "w").write("\n".join(w) + "\n")
    # paired comparison: per (scenario, seed) mean over instances; Coop vs others
    def paired(a, b, m="avg_wait_s"):
        diffs = []
        for sc in C.SCENARIOS:
            for s in C.TEST_SEEDS:
                va = np.mean([r[m] for r in res[a][sc] if r["seed"] == s]); vb = np.mean([r[m] for r in res[b][sc] if r["seed"] == s])
                diffs.append(va - vb)
        d = np.array(diffs); rng = np.random.default_rng(0)
        boots = [rng.choice(d, len(d)).mean() for _ in range(5000)]
        return d.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5), float((d < 0).mean())
    out = {}
    for b in ("Random", "Fixed-time", "Longest-queue-first", "IQL"):
        out[b] = paired("Coop-IQL (proposed)", b)
    json.dump({f"Coop-IQL minus {k}": {"mean_diff_wait_s": v[0], "ci95_low": v[1], "ci95_high": v[2], "share_of_runs_coop_better": v[3]} for k, v in out.items()},
              open(os.path.join(outdir, "paired_comparison_wait.json"), "w"), indent=2)
    return out


def main(rd):
    logs = json.load(open(os.path.join(rd, "training_logs.json")))
    res = json.load(open(os.path.join(rd, "evaluation_raw.json")))
    learning_curves(logs, os.path.join(rd, "learning_curves.png"))
    overall, per = aggregate(res)
    bar_charts(overall, per, rd)
    p = tables(overall, per, res, rd)
    print(open(os.path.join(rd, "results_overall.md")).read())
    print(open(os.path.join(rd, "results_wait_by_scenario.md")).read())
    for k, v in p.items():
        print(f"Coop-IQL - {k}: mean diff wait {v[0]:.2f}s  95% CI [{v[1]:.2f}, {v[2]:.2f}]  better in {v[3]*100:.0f}% of paired runs")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join("..", "05_Experimental_Results"))
