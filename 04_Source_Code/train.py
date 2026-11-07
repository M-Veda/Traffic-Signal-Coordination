"""Training of IQL and cooperative IQL on randomly sampled demand scenarios (many independent seeds)."""
import json, os, pickle, sys, time
import numpy as np
import config as C
from traffic_env import TrafficNetwork, generate_arrivals
from agents import MultiAgentQ
from evaluate import run_episode

VAL_SEEDS = [5000, 5001]


def validate(env, agent):
    old, agent.eps = agent.eps, 0.0
    waits = []
    for sc in C.SCENARIOS:
        for s in VAL_SEEDS:
            waits.append(run_episode(env, agent, generate_arrivals(sc, s))["avg_wait_s"])
    agent.eps = old
    return float(np.mean(waits))


def train_one(cooperative, seed, episodes=C.TRAIN_EPISODES, verbose=True, coop_info="downstream"):
    env = TrafficNetwork()
    cfg = C.QConfig()
    agent = MultiAgentQ(env, cooperative, cfg, seed=seed, coop_info=coop_info)
    rng = np.random.default_rng(10_000 + seed)
    log = {"ep_return": [], "ep_wait": [], "eps": [], "scenario": [], "val_ep": [], "val_wait": []}
    decay_eps = max(1, int(cfg.eps_decay_frac * episodes))
    for ep in range(episodes):
        agent.eps = max(cfg.eps_end, cfg.eps_start - (cfg.eps_start - cfg.eps_end) * ep / decay_eps)
        sc = C.SCENARIOS[rng.integers(len(C.SCENARIOS))]
        env.reset(generate_arrivals(sc, int(rng.integers(0, 1000))))
        s = agent.states(env)
        done, ret, steps = False, 0.0, 0
        while not done:
            a = agent.act(env, explore=True)
            raw_r, done = env.step(a)
            s2 = agent.states(env)
            agent.update(s, a, agent.shape_rewards(env, raw_r), s2)
            s = s2
            ret += float(np.mean(raw_r)); steps += 1
        m = env.metrics()
        log["ep_return"].append(ret); log["ep_wait"].append(m["avg_wait_s"])
        log["eps"].append(agent.eps); log["scenario"].append(sc)
        if (ep + 1) % C.VAL_EVERY == 0 or ep == 0:
            v = validate(env, agent)
            log["val_ep"].append(ep + 1); log["val_wait"].append(v)
            if verbose:
                print(f"  [{agent.name} seed {seed}] ep {ep+1:4d} eps={agent.eps:.2f} "
                      f"mean return(last 20)={np.mean(log['ep_return'][-20:]):8.2f} val avg wait={v:6.2f}s", flush=True)
    return agent, log


def main(results_dir):
    os.makedirs(results_dir, exist_ok=True)
    out = {"IQL": [], "Coop-IQL (proposed)": []}
    logs = {"IQL": [], "Coop-IQL (proposed)": []}
    t0 = time.time()
    for coop, name in ((False, "IQL"), (True, "Coop-IQL (proposed)")):
        for seed in C.TRAIN_SEEDS:
            agent, log = train_one(coop, seed)
            out[name].append(agent.Q); logs[name].append(log)
    with open(os.path.join(results_dir, "trained_q_tables.pkl"), "wb") as f:
        pickle.dump(out, f)
    with open(os.path.join(results_dir, "training_logs.json"), "w") as f:
        json.dump(logs, f)
    print(f"Training finished in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join("..", "05_Experimental_Results"))
