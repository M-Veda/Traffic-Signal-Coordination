"""Sanity tests: python -m unittest discover -s tests   (run from 04_Source_Code)"""
import os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as C
from traffic_env import TrafficNetwork, generate_arrivals
from agents import FixedTimeController, RandomController, LongestQueueController, MultiAgentQ
from evaluate import run_episode


class EnvTests(unittest.TestCase):
    def test_vehicle_conservation(self):
        env = TrafficNetwork()
        m = run_episode(env, FixedTimeController(), generate_arrivals("medium", 1))
        in_net = sum(len(env.queue[i][d]) + len(env.transit[i][d]) for i in range(env.n) for d in range(4))
        self.assertEqual(m["vehicles_generated"], m["throughput"] + in_net)

    def test_identical_demand_for_all_controllers(self):
        a = generate_arrivals("high", 7)
        self.assertEqual(a, generate_arrivals("high", 7))
        env = TrafficNetwork()
        gens = {run_episode(env, c, a)["vehicles_generated"] for c in (FixedTimeController(), RandomController(), LongestQueueController())}
        self.assertEqual(len(gens), 1)

    def test_internal_queue_capacity_respected(self):
        from traffic_env import DELTA
        env = TrafficNetwork(); env.reset(generate_arrivals("peak", 3)); ctrl = RandomController(); done = False
        while not done:
            _, done = env.step(ctrl.act(env))
            for i in range(env.n):
                r, c = divmod(i, env.cols)
                for d in range(4):
                    ur, uc = r - DELTA[d][0], c - DELTA[d][1]
                    if 0 <= ur < env.rows and 0 <= uc < env.cols:     # fed by an internal link
                        self.assertLessEqual(len(env.queue[i][d]) + len(env.transit[i][d]), C.QUEUE_CAP)

    def test_agent_state_index_in_range(self):
        env = TrafficNetwork(); env.reset(generate_arrivals("high", 2))
        for coop in (False, True):
            ag = MultiAgentQ(env, coop)
            for _ in range(60):
                s = ag.states(env)
                self.assertTrue(all(0 <= x < ag.Q.shape[1] for x in s))
                env.step(ag.act(env))

    def test_q_update_moves_value(self):
        env = TrafficNetwork(); env.reset(generate_arrivals("low", 2)); ag = MultiAgentQ(env, True)
        s = ag.states(env); ag.update(s, [0] * env.n, [-1.0] * env.n, s)
        self.assertLess(ag.Q[0, s[0], 0], 0)


if __name__ == "__main__":
    unittest.main()
