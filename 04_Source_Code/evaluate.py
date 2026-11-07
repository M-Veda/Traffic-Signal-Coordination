"""Evaluation of all controllers on identical demand realisations."""
import json, os, pickle, sys
import numpy as np
import config as C
from traffic_env import TrafficNetwork, generate_arrivals
from agents import FixedTimeController, RandomController, LongestQueueController, MultiAgentQ

METRICS = ["avg_wait_s", "avg_queue_len", "throughput", "avg_travel_time_s", "congestion_index",
           "spillback_events", "vehicles_left_in_network"]


def run_episode(env, controller, arrivals, record=False):
    env.reset(arrivals)
    controller.reset(env)
    frames, done = [], False
    if record:
        frames.append(env.snapshot())
    while not done:
        acts = controller.act(env, explore=False)
        _, done = env.step(acts)
        if record:
            frames.append(env.snapshot())
    return (env.metrics(), frames) if record else env.metrics()


def evaluate_all(controllers, scenarios=C.SCENARIOS, seeds=C.TEST_SEEDS):
    """controllers: dict name -> object (or list of objects for multiple training seeds).
    Returns results[name][scenario] = list of metric dicts (one per test seed x controller instance)."""
    env = TrafficNetwork()
    res = {}
    for sc in scenarios:
        arr_cache = {s: generate_arrivals(sc, s) for s in seeds}
        for name, ctrls in controllers.items():
            ctrls = ctrls if isinstance(ctrls, list) else [ctrls]
            for ci, ctrl in enumerate(ctrls):
                for s in seeds:
                    if isinstance(ctrl, RandomController):
                        ctrl.rng = np.random.default_rng(s)
                    m = run_episode(env, ctrl, arr_cache[s])
                    m.update({"seed": s, "ctrl_instance": ci})
                    res.setdefault(name, {}).setdefault(sc, []).append(m)
    return res


def load_agents(results_dir):
    with open(os.path.join(results_dir, "trained_q_tables.pkl"), "rb") as f:
        d = pickle.load(f)
    env = TrafficNetwork()
    out = {}
    for name, coop in (("IQL", False), ("Coop-IQL (proposed)", True)):
        lst = []
        for Q in d[name]:
            a = MultiAgentQ(env, coop, coop_info="downstream")
            a.Q = Q
            a.eps = 0.0
            lst.append(a)
        out[name] = lst
    return out


def main(results_dir):
    agents = load_agents(results_dir)
    controllers = {"Random": RandomController(), "Fixed-time": FixedTimeController(),
                   "Longest-queue-first": LongestQueueController(), **agents}
    res = evaluate_all(controllers)
    with open(os.path.join(results_dir, "evaluation_raw.json"), "w") as f:
        json.dump(res, f)
    print("Evaluation finished.")
    return res


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join("..", "05_Experimental_Results"))
