# Handover Summary - RL Homework (Project 10: Multi-Agent RL Traffic Signal Coordination)

## Files
- 01_Project_Report/RL_Traffic_Signal_Coordination_Report.pdf (9 pages)
- 02_Problem_Formulation/Problem_Formulation_One_Page.pdf (1 page)
- 03_Presentation/RL_Traffic_Signal_Coordination_5_Slides.pptx (exactly 5 slides)
- 04_Source_Code/ (config, traffic_env, agents, train, evaluate, make_results, visualize, run_all, tests, README)
- 05_Experimental_Results/ (learning curves, comparison charts, tables, raw JSON, trained Q-tables, 2 GIF animations, archived first-iteration results)
- 06_Documentation/ (this summary); README.md and requirements.txt are at the repository root

## Experiments executed
Training: IQL and Coop-IQL, 5 independent runs each, 1000 episodes (about 10 min total, 1 CPU core).
Evaluation: 5 controllers x 5 demand scenarios x 10 unseen seeds (RL: x 5 trained runs), identical arrivals per seed.

## Actual results (mean over all scenarios and test runs)
| Controller | Avg wait (s) | Avg queue (veh/approach) | Throughput (veh/30 min) | Avg travel time (s) | Congestion index |
|---|---|---|---|---|---|
| Random | 90.8 | 3.32 | 2083 | 106.3 | 0.109 |
| Fixed-time | 37.5 | 1.35 | 2179 | 55.8 | 0.019 |
| Longest-queue-first | 33.8 | 1.19 | 2190 | 52.2 | 0.005 |
| IQL | 33.1 | 1.18 | 2188 | 50.9 | 0.007 |
| Coop-IQL (proposed) | 37.4 | 1.35 | 2169 | 52.3 | 0.012 |

IQL beat Fixed-time by about 12% in waiting time (about 28% under asymmetric demand) and was roughly on par with Longest-queue-first.
The cooperative variant did NOT beat IQL: it matched IQL in 2 of 5 runs and was unstable (about 80 s wait) under peak demand in 3 of 5 runs.

## Verification status
- 5 unit tests pass; full pipeline executed; `run_all.py --quick` smoke test passed end to end.
- Report/slides/one-pager numbers are generated from the result files by script.
- Slides: exactly 5 (checked); one-pager: 1 page (checked via pdfinfo).
- NOT done: README commands were not run on a real Windows machine (built and run on Linux, Python 3.12).

## Limitations
Custom simulator (not SUMO); two-phase signals, no left turns/pedestrians; tabular states; 5 training runs; fixed-time baseline not tuned;
test seeds were seen once before the redesign of Coop-IQL (disclosed in report Section 11).
