"""Signal controllers: baselines (fixed-time, random, longest-queue-first) and the tabular
Q-learning agents (Independent Q-Learning, and cooperative IQL with information/reward sharing)."""
import numpy as np
import config as C
from traffic_env import PHASE_DIRS


# ----------------------------------------------------------------------------- baselines
class FixedTimeController:
    name = "Fixed-time"

    def __init__(self, green_intervals=2):
        self.g = green_intervals          # 2 x 10 s = 20 s green per phase (40 s cycle)

    def reset(self, env):
        self.k = 0

    def act(self, env, explore=False):
        phase = (self.k // self.g) % 2
        self.k += 1
        return [phase] * env.n


class RandomController:
    name = "Random"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def reset(self, env):
        pass

    def act(self, env, explore=False):
        return [int(self.rng.integers(2)) for _ in range(env.n)]


class LongestQueueController:
    """Simple actuated heuristic: give green to the axis with the larger queue (hysteresis of 2 vehicles)."""
    name = "Longest-queue-first"

    def reset(self, env):
        pass

    def act(self, env, explore=False):
        acts = []
        for i in range(env.n):
            q = [sum(env.qlen(i, d) for d in PHASE_DIRS[p]) for p in (0, 1)]
            cur = env.phase[i]
            acts.append(1 - cur if q[1 - cur] > q[cur] + 2 else cur)
        return acts


# ----------------------------------------------------------------------------- RL agents
def q_bin(x):            # queue sum of two approaches
    return 0 if x <= 3 else 1 if x <= 9 else 2 if x <= 19 else 3


def w_bin(w):            # waiting time (s) of the longest-waiting head vehicle on the red axis
    return 0 if w < 15 else 1 if w < 40 else 2


def d_flag(occ):         # downstream approach nearly full (>= 60% of QUEUE_CAP)?
    return 1 if occ >= 0.6 * C.QUEUE_CAP else 0


def n_bin(x):            # mean total queue of neighbouring intersections
    return 0 if x < 6 else 1 if x < 15 else 2


class MultiAgentQ:
    """One independent tabular Q-learner per intersection.

    cooperative=False -> Independent Q-Learning (IQL): local state, local reward.
    cooperative=True  -> IQL + (a) neighbour congestion level appended to the state (information sharing)
                         and (b) reward = (1-beta)*own + beta*mean(neighbour rewards) (reward sharing).
    """
    N_ACTIONS = 2

    def __init__(self, env, cooperative, cfg=None, seed=0, coop_info="downstream"):
        # coop_info: "downstream" (final design) = downstream-approach occupancy flags of each axis
        #            "nbr_queue"  (first iteration) = mean total queue of neighbouring intersections
        self.coop_info = coop_info
        self.cfg = cfg or C.QConfig()
        self.cooperative = cooperative
        self.name = "Coop-IQL (proposed)" if cooperative else "IQL"
        self.n = env.n
        self.rng = np.random.default_rng(seed)
        n_states = 4 * 4 * 2 * 3 * ((4 if coop_info == 'downstream' else 3) if cooperative else 1)
        self.Q = np.zeros((self.n, n_states, self.N_ACTIONS))
        self.eps = 0.0

    def reset(self, env):
        pass

    def state(self, env, i):
        qs = [sum(env.qlen(i, d) for d in PHASE_DIRS[p]) for p in (0, 1)]
        cur = env.phase[i]
        red = 1 - cur
        red_wait = max(env.head_wait(i, d) for d in PHASE_DIRS[red])
        s = ((q_bin(qs[0]) * 4 + q_bin(qs[1])) * 2 + cur) * 3 + w_bin(red_wait)
        if self.cooperative:
            if self.coop_info == "downstream":
                code = 0
                for p in (0, 1):      # shared information: how full are the approaches this axis feeds into?
                    full = 0
                    for d in PHASE_DIRS[p]:
                        nxt = env.next_node[(i, d)]
                        if nxt is not None and d_flag(len(env.queue[nxt][d]) + len(env.transit[nxt][d])):
                            full = 1
                    code = code * 2 + full
                s = s * 4 + code
            else:
                nb = env.neighbors[i]
                mean_nb = np.mean([sum(env.qlen(j, d) for d in range(4)) for j in nb])
                s = s * 3 + n_bin(mean_nb)
        return s

    def states(self, env):
        return [self.state(env, i) for i in range(self.n)]

    def act(self, env, explore=False):
        acts = []
        for i, s in enumerate(self.states(env)):
            if explore and self.rng.random() < self.eps:
                acts.append(int(self.rng.integers(2)))
            else:
                q = self.Q[i, s]
                acts.append(int(self.rng.integers(2)) if q[0] == q[1] else int(np.argmax(q)))
        return acts

    def shape_rewards(self, env, rewards):
        if not self.cooperative:
            return rewards
        b = self.cfg.coop_beta
        return [(1 - b) * rewards[i] + b * float(np.mean([rewards[j] for j in env.neighbors[i]]))
                for i in range(self.n)]

    def update(self, s, a, r, s2):
        al, g = self.cfg.alpha, self.cfg.gamma
        for i in range(self.n):
            target = r[i] + g * self.Q[i, s2[i]].max()
            self.Q[i, s[i], a[i]] += al * (target - self.Q[i, s[i], a[i]])
