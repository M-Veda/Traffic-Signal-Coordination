"""Central configuration for the traffic-signal MARL project (single source of truth)."""
import os
from dataclasses import dataclass

# ---- Network / simulation ---------------------------------------------------
ROWS, COLS = 3, 3            # 3x3 grid = 9 signalised intersections
LINK_TIME = 8                # seconds to travel one link (free flow)
QUEUE_CAP = 20               # max vehicles (queued + in transit) heading into one approach
HEADWAY = 2                  # one vehicle discharged per 2 s per green approach (saturation flow 0.5 veh/s)
LOST_TIME = 2                # seconds of all-red lost time whenever a signal changes phase
HORIZON = 1800               # simulated seconds per episode (30 min)
DECISION_INTERVAL = 10       # an agent chooses a phase every 10 s

# ---- Demand scenarios (per-entry-point arrival rates are defined in traffic_env.generate_arrivals)
# 12 entry points (3 per side).  'asym' = heavy S-moving and E-moving flows (rush hour),
# 'peak' = demand ramps up to a peak at mid-episode and falls again.
SCENARIOS = ["low", "medium", "high", "asym", "peak"]


# ---- Q-learning -------------------------------------------------------------
@dataclass
class QConfig:
    alpha: float = 0.1
    gamma: float = 0.9
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_frac: float = 0.7      # epsilon reaches eps_end after this fraction of training
    coop_beta: float = 0.3           # weight of neighbours' mean reward in the shared reward


# ---- Reward weights ---------------------------------------------------------
W_QUEUE = 0.10       # per vehicle of average queued vehicles at the intersection
W_WAIT = 0.50        # per 60 s of waiting of the longest-waiting head vehicle
W_BLOCK = 0.50       # spill-back (blocked discharge) penalty

TRAIN_EPISODES = 1000
TRAIN_SEEDS = [0, 1, 2, 3, 4]               # independent training runs (different random seeds)
TEST_SEEDS = list(range(1000, 1010))   # 10 unseen demand realisations per scenario
VAL_EVERY = 50

# Quick smoke-test mode (python run_all.py --quick): fewer episodes/seeds, NOT used for the reported results
if os.environ.get("RL_QUICK") == "1":
    TRAIN_EPISODES = 100
    TRAIN_SEEDS = [0, 1]
    TEST_SEEDS = [1000, 1001, 1002]
    VAL_EVERY = 25
