"""Lightweight custom multi-intersection traffic simulator (NOT SUMO).

Store-and-forward / queue model with individually tracked vehicles:
  * 3x3 grid of signalised intersections, each with 4 approaches (vehicle moving S, N, E, W).
  * Two-phase signals: phase 0 = north-south green, phase 1 = east-west green.
  * Every second: external arrivals enter, vehicles finish link travel, green approaches
    discharge one vehicle every HEADWAY seconds if the downstream approach has room
    (spill-back); a phase change costs LOST_TIME seconds of all-red.
  * Vehicles follow dimension-order routes (columns first, then rows) to a boundary exit.
"""
from collections import deque
import numpy as np
import config as C

# movement direction -> (d_row, d_col)
DELTA = {0: (1, 0), 1: (-1, 0), 2: (0, 1), 3: (0, -1)}   # 0=S(down) 1=N(up) 2=E 3=W
OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2}
PHASE_DIRS = {0: (0, 1), 1: (2, 3)}                        # phase -> served approaches
DIR_NAME = {0: "S", 1: "N", 2: "E", 3: "W"}


def entry_points(rows, cols):
    """(node, direction the vehicle moves when it enters the network)."""
    pts = []
    for c in range(cols):
        pts.append((c, 0))                           # top side, moving S
        pts.append(((rows - 1) * cols + c, 1))       # bottom side, moving N
    for r in range(rows):
        pts.append((r * cols, 2))                    # left side, moving E
        pts.append((r * cols + cols - 1, 3))         # right side, moving W
    return pts


def exit_points(rows, cols):
    """(node, direction the vehicle moves when it leaves the network)."""
    pts = []
    for c in range(cols):
        pts.append(((rows - 1) * cols + c, 0))       # leaves bottom, moving S
        pts.append((c, 1))                           # leaves top, moving N
    for r in range(rows):
        pts.append((r * cols + cols - 1, 2))         # leaves right, moving E
        pts.append((r * cols, 3))                    # leaves left, moving W
    return pts


def generate_arrivals(scenario, seed, rows=C.ROWS, cols=C.COLS, horizon=C.HORIZON):
    """Pre-generate the whole demand of one episode so every controller sees IDENTICAL traffic."""
    rng = np.random.default_rng(seed * 7919 + C.SCENARIOS.index(scenario) * 104729 + 17)
    entries, exits = entry_points(rows, cols), exit_points(rows, cols)
    base = {"low": 0.07, "medium": 0.10, "high": 0.13}
    arrivals = []
    for (node, d) in entries:
        for t in range(horizon):
            if scenario in base:
                lam = base[scenario]
            elif scenario == "asym":
                lam = 0.16 if d in (0, 2) else 0.05
            else:  # peak: triangular profile 0.05 -> 0.18 -> 0.05
                x = t / horizon
                lam = 0.05 + 0.13 * (1 - abs(2 * x - 1))
            if rng.random() < lam:          # Bernoulli per second ~ Poisson at these rates
                while True:
                    en, ed = exits[rng.integers(len(exits))]
                    if en != node:          # destination is a different boundary intersection
                        break
                arrivals.append((t, node, d, en, ed))
    arrivals.sort(key=lambda a: a[0])
    return arrivals


class TrafficNetwork:
    def __init__(self, rows=C.ROWS, cols=C.COLS):
        self.rows, self.cols, self.n = rows, cols, rows * cols
        self.neighbors = []
        self.next_node = {}      # (node, moving direction) -> next node, or None = leaves network
        for i in range(self.n):
            r, c = divmod(i, cols)
            nb = []
            for d, (dr, dc) in DELTA.items():
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols:
                    self.next_node[(i, d)] = rr * cols + cc
                    nb.append(rr * cols + cc)
                else:
                    self.next_node[(i, d)] = None
            self.neighbors.append(nb)

    def reset(self, arrivals):
        self.arrivals, self.ptr, self.t = arrivals, 0, 0
        n = self.n
        self.queue = [[deque() for _ in range(4)] for _ in range(n)]
        self.transit = [[deque() for _ in range(4)] for _ in range(n)]
        self.phase = [0] * n
        self.lost = [0] * n
        self.v_dest, self.v_exit, self.v_t_in, self.v_wait, self.v_qenter, self.v_done = [], [], [], [], [], []
        self.completed = 0
        self.q_time_sum = 0.0
        self.cong_time_sum = 0.0
        self.block_events = 0
        self._reset_interval()
        return self

    def _reset_interval(self):
        self.int_qacc = [0.0] * self.n
        self.int_block = [0] * self.n
        self.int_dis = [0] * self.n

    def _next_dir(self, node, vid):
        r, c = divmod(node, self.cols)
        dr_, dc_ = divmod(self.v_dest[vid], self.cols)
        if c != dc_:
            return 2 if dc_ > c else 3
        if r != dr_:
            return 0 if dr_ > r else 1
        return self.v_exit[vid]

    def _second(self):
        t = self.t
        while self.ptr < len(self.arrivals) and self.arrivals[self.ptr][0] <= t:
            _, node, d, dest, ex = self.arrivals[self.ptr]
            self.ptr += 1
            vid = len(self.v_dest)
            self.v_dest.append(dest); self.v_exit.append(ex); self.v_t_in.append(t)
            self.v_wait.append(0); self.v_qenter.append(t); self.v_done.append(-1)
            self.queue[node][d].append(vid)
        for i in range(self.n):
            for d in range(4):
                tr = self.transit[i][d]
                while tr and tr[0][0] <= t:
                    _, vid = tr.popleft()
                    self.v_qenter[vid] = t
                    self.queue[i][d].append(vid)
        if t % C.HEADWAY == 0:
            for i in range(self.n):
                if self.lost[i] > 0:
                    continue
                for d in PHASE_DIRS[self.phase[i]]:
                    q = self.queue[i][d]
                    if not q:
                        continue
                    vid = q[0]
                    nd = self._next_dir(i, vid)          # direction the vehicle leaves this junction in
                    nxt = self.next_node[(i, nd)]
                    if nxt is None:                      # leaves the network
                        q.popleft()
                        self.v_wait[vid] += t - self.v_qenter[vid]
                        self.v_done[vid] = t
                        self.completed += 1
                        self.int_dis[i] += 1
                    elif len(self.queue[nxt][nd]) + len(self.transit[nxt][nd]) < C.QUEUE_CAP:
                        q.popleft()
                        self.v_wait[vid] += t - self.v_qenter[vid]
                        self.transit[nxt][nd].append((t + C.LINK_TIME, vid))
                        self.int_dis[i] += 1
                    else:                                # spill-back: downstream approach is full
                        self.int_block[i] += 1
                        self.block_events += 1
        tot_q, cong = 0, 0
        for i in range(self.n):
            if self.lost[i] > 0:
                self.lost[i] -= 1
            s = 0
            for d in range(4):
                L = len(self.queue[i][d])
                s += L
                if L >= 10:
                    cong += 1
            self.int_qacc[i] += s
            tot_q += s
        self.q_time_sum += tot_q
        self.cong_time_sum += cong / (4 * self.n)
        self.t += 1

    def step(self, actions, per_second_callback=None):
        """actions[i] in {0,1} = phase for the next DECISION_INTERVAL seconds. Returns (rewards, done)."""
        for i, a in enumerate(actions):
            if a != self.phase[i]:
                self.phase[i] = a
                self.lost[i] = C.LOST_TIME
        self._reset_interval()
        for _ in range(C.DECISION_INTERVAL):
            self._second()
            if per_second_callback is not None:
                per_second_callback(self)
        rewards = []
        for i in range(self.n):
            avgq = self.int_qacc[i] / C.DECISION_INTERVAL
            maxw = max(self.head_wait(i, d) for d in range(4))
            blk = min(self.int_block[i] / 5.0, 2.0)
            rewards.append(-(C.W_QUEUE * avgq + C.W_WAIT * maxw / 60.0 + C.W_BLOCK * blk))
        return rewards, self.t >= C.HORIZON

    def qlen(self, i, d):
        return len(self.queue[i][d])

    def head_wait(self, i, d):
        q = self.queue[i][d]
        return (self.t - self.v_qenter[q[0]]) if q else 0

    def in_network(self):
        return len(self.v_dest) - self.completed

    def metrics(self):
        n_veh = len(self.v_dest)
        wait = list(self.v_wait)
        for i in range(self.n):                      # add current (unfinished) wait of queued vehicles
            for d in range(4):
                for vid in self.queue[i][d]:
                    wait[vid] += self.t - self.v_qenter[vid]
        tt = [self.v_done[v] - self.v_t_in[v] for v in range(n_veh) if self.v_done[v] >= 0]
        return {
            "avg_wait_s": float(np.mean(wait)) if n_veh else 0.0,
            "avg_queue_len": self.q_time_sum / max(self.t, 1) / (4 * self.n),
            "throughput": self.completed,
            "avg_travel_time_s": float(np.mean(tt)) if tt else 0.0,
            "congestion_index": self.cong_time_sum / max(self.t, 1),
            "spillback_events": self.block_events,
            "vehicles_generated": n_veh,
            "vehicles_left_in_network": n_veh - self.completed,
        }

    def snapshot(self):
        """Vehicle positions for the animation: (node, approach dir, progress on incoming link, queue index or -1)."""
        veh = []
        for i in range(self.n):
            for d in range(4):
                for k, _ in enumerate(self.queue[i][d]):
                    veh.append((i, d, 1.0, k))
                for (arr, _) in self.transit[i][d]:
                    prog = 1.0 - (arr - self.t) / C.LINK_TIME
                    veh.append((i, d, max(0.0, min(prog, 1.0)), -1))
        return {"t": self.t, "phase": list(self.phase), "lost": list(self.lost), "veh": veh}
