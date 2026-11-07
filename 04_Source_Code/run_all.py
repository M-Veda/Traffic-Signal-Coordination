"""One-command pipeline:  python run_all.py            (full: ~10-15 min on one CPU core)
                          python run_all.py --quick    (smoke test, ~1 min, writes to ../05_Experimental_Results_quick)"""
import os, subprocess, sys

quick = "--quick" in sys.argv
rd = os.path.join("..", "05_Experimental_Results_quick" if quick else "05_Experimental_Results")
os.makedirs(rd, exist_ok=True)
env = dict(os.environ, RL_QUICK="1" if quick else "0")
py = sys.executable
steps = [[py, "-m", "unittest", "discover", "-s", "tests"],
         [py, "train.py", rd], [py, "evaluate.py", rd], [py, "make_results.py", rd], [py, "make_architecture.py", rd],
         [py, "visualize.py", "--controller", "iql", "--scenario", "asym", "--start", "700", "--duration", "100", "--results_dir", rd],
         [py, "visualize.py", "--controller", "fixed", "--scenario", "asym", "--start", "700", "--duration", "100", "--results_dir", rd]]
for cmd in steps:
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)
print("Done. Results in", os.path.abspath(rd))
