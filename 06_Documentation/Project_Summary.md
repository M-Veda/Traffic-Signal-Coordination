# Project Summary - Intelligent Traffic Signal Coordination Using Multi-Agent RL

A concise technical summary of what was built, what was run and what was found. See the root `README.md` for the full description.

## Components
- `01_Project_Report/` - technical report (9 pages)
- `02_Problem_Formulation/` - one-page MDP / algorithm / multi-agent interaction brief
- `03_Presentation/` - five-slide project presentation
- `04_Source_Code/` - simulator, agents, training, evaluation, visualization, tests
- `05_Experimental_Results/` - charts, tables, raw results, trained Q-tables, animations, archived first design

## Experiments executed
- Training: IQL and Coop-IQL, 5 independent runs each, 1000 episodes (about 10 min total on one CPU core).
- Evaluation: 5 controllers x 5 demand scenarios x 10 unseen seeds (RL controllers: x 5 trained runs), identical arrivals per seed.

## Results (mean over all scenarios and test runs)
| Controller | Avg wait (s) | Avg queue (veh/approach) | Throughput (veh/30 min) | Avg travel time (s) | Congestion index |
|---|---|---|---|---|---|
| Random | 90.8 | 3.32 | 2083 | 106.3 | 0.109 |
| Fixed-time | 37.5 | 1.35 | 2179 | 55.8 | 0.019 |
| Longest-queue-first | 33.8 | 1.19 | 2190 | 52.2 | 0.005 |
| IQL | 33.1 | 1.18 | 2188 | 50.9 | 0.007 |
| Coop-IQL | 37.4 | 1.35 | 2169 | 52.3 | 0.012 |

IQL reduced waiting time by about 12% versus Fixed-time (about 28% under asymmetric demand) and was roughly on par with Longest-queue-first.
Coop-IQL did not outperform IQL: it matched IQL in 2 of 5 runs and was unstable (about 80 s wait) under peak demand in 3 of 5 runs.

## Verification
- 5 unit tests pass; the full pipeline was executed; `python run_all.py --quick` smoke test passes end to end.
- Report, slides and problem-formulation numbers are generated from the result files by script.
- Not verified: the Windows commands in the README were not run on a Windows machine (developed on Linux, Python 3.12).

## Limitations
Custom simulator (not SUMO or CityFlow); two-phase signals, no left turns or pedestrians; tabular states; 5 training runs; Fixed-time baseline not tuned;
test seeds were seen once before the Coop-IQL redesign (disclosed in the report, Section 9). Simulation only - no real-world deployment or testing.
