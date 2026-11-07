"""Builds the one-page problem formulation PDF and the full project report PDF.
All numbers are read from the experiment outputs in ../05_Experimental_Results (nothing is typed in by hand)."""
import json, os, sys
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether)
import config as C
from make_results import aggregate, ORDER

RD = sys.argv[1] if len(sys.argv) > 1 else os.path.join("..", "05_Experimental_Results")
OUT_REPORT = os.path.join("..", "01_Project_Report", "Traffic_Signal_Coordination_Technical_Report.pdf")
OUT_ONE = os.path.join("..", "02_Problem_Formulation", "Problem_Formulation.pdf")


def load_numbers():
    res = json.load(open(os.path.join(RD, "evaluation_raw.json")))
    logs = json.load(open(os.path.join(RD, "training_logs.json")))
    paired = json.load(open(os.path.join(RD, "paired_comparison_wait.json")))
    overall, per = aggregate(res)
    v1 = json.load(open(os.path.join(RD, "iteration1_v1_400ep", "evaluation_raw.json")))
    v1_overall = {}
    for n in ("IQL", "Coop-IQL (proposed)"):
        v1_overall[n] = np.mean([r["avg_wait_s"] for sc in C.SCENARIOS for r in v1[n][sc]])
    peak = {}
    for n in ("IQL", "Coop-IQL (proposed)"):
        peak[n] = [float(np.mean([r["avg_wait_s"] for r in res[n]["peak"] if r["ctrl_instance"] == k])) for k in range(5)]
    return dict(res=res, logs=logs, paired=paired, overall=overall, per=per, v1=v1_overall, peak=peak)


def pct(a, b):  # relative change of a w.r.t. b in %
    return 100.0 * (a - b) / b


# ============================================================================ one-page formulation
def one_page(N):
    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["Normal"], fontName="Helvetica", fontSize=9.2, leading=11.6, alignment=TA_JUSTIFY)
    h = ParagraphStyle("h", parent=ss["Normal"], fontName="Helvetica-Bold", fontSize=10.2, leading=12.4, textColor=colors.HexColor("#1c2b4a"), spaceBefore=4, spaceAfter=1)
    t = ParagraphStyle("t", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=13, leading=15, spaceAfter=1)
    sub = ParagraphStyle("s", parent=body, alignment=TA_CENTER, fontSize=8.2, textColor=colors.grey)
    cell = ParagraphStyle("c", parent=body, fontSize=8.8, leading=10.8, alignment=0)
    doc = SimpleDocTemplate(OUT_ONE, pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm, topMargin=1.2 * cm, bottomMargin=1.0 * cm,
                            title="Problem Formulation - Intelligent Traffic Signal Coordination Using Multi-Agent Reinforcement Learning")
    P = lambda s, st=body: Paragraph(s, st)
    el = [P("Problem Formulation: Intelligent Traffic Signal Coordination Using Multi-Agent Reinforcement Learning", t),
          P("Technical brief: MDP formulation, algorithm and multi-agent interaction | custom Python simulator (not SUMO)", sub),
          P("1. Problem statement and motivation", h),
          P("Urban congestion wastes time and fuel. Conventional fixed-time signals repeat a pre-set cycle regardless of the actual traffic, so green time is "
            "wasted on empty approaches while long queues build elsewhere, and queues at one junction spill back and block the next. The task is to let "
            "<b>every signal learn, from experience, which phase to display</b> so that vehicles wait less across the whole network, while neighbouring "
            "signals exchange information instead of acting blindly. This is a sequential decision problem under uncertain demand, hence a natural fit for RL.")]
    el += [P("2. Environment", h),
           P("Custom Python simulator (<i>not</i> SUMO): a 3x3 grid of 9 two-phase signalised intersections joined by links of 8 s travel time, 12 boundary entry "
             "points and individually tracked vehicles on fixed routes. A green approach discharges one vehicle per 2 s; switching phase costs 2 s of all-red; "
             "an approach that is full (20 vehicles) blocks the upstream junction (spill-back). Episode = 1800 s; an agent acts every 10 s (180 decisions). "
             "Five demand scenarios (low, medium, high, asymmetric rush-hour, time-varying peak).")]
    rows = [[P("<b>Component</b>", cell), P("<b>Definition for agent i (one per intersection)</b>", cell)],
            [P("State s<sub>i</sub>", cell), P("(queue of N+S axis, queue of E+W axis) each in 4 levels {0-3, 4-9, 10-19, 20+} x current phase (2) x longest head-of-queue wait of the red axis in 3 levels "
                                             "{&lt;15, 15-40, &gt;40 s} = 96 local states; <b>Coop-IQL</b> adds 2 binary flags 'downstream approach of axis is &ge;60% full' (received from neighbours) = 384 states.", cell)],
            [P("Action a<sub>i</sub>", cell), P("{0 = north-south green, 1 = east-west green} held for the next 10 s (a change inserts 2 s all-red).", cell)],
            [P("Reward r<sub>i</sub>", cell), P("r<sub>i</sub> = &minus;( 0.10 &middot; q<sub>avg,i</sub> + 0.50 &middot; w<sub>max,i</sub>/60 + 0.50 &middot; min(b<sub>i</sub>/5, 2) ), with q<sub>avg</sub> = mean number of queued vehicles at the junction over the "
                                              "interval, w<sub>max</sub> = longest head-of-queue waiting time (s), b = blocked discharges (spill-back). Penalises queues, waiting and congestion.", cell)],
            [P("Transition", cell), P("Stochastic: arrivals are random and the next state also depends on neighbours' phases; discount &gamma; = 0.9.", cell)]]
    tb = Table(rows, colWidths=[2.2 * cm, 15.8 * cm])
    tb.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe9f7")),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
    el += [Spacer(1, 2), tb]
    el += [P("3. RL algorithm", h),
           P("Tabular <b>Independent Q-Learning (IQL)</b>: each of the 9 agents keeps its own table Q<sub>i</sub>(s,a) and updates "
             "Q<sub>i</sub>(s,a) &larr; Q<sub>i</sub>(s,a) + &alpha;[ r + &gamma; max<sub>a'</sub> Q<sub>i</sub>(s',a') &minus; Q<sub>i</sub>(s,a) ], &alpha; = 0.1, "
             "&epsilon;-greedy exploration decayed linearly from 1.0 to 0.05 over the first 70% of 1000 training episodes (random demand scenario each episode). "
             "Tabular learning keeps the system small, interpretable and reproducible on a laptop."),
           P("4. Multi-agent interaction", h),
           P("The 9 agents act <i>simultaneously</i>; the environment is therefore non-stationary for each agent (the neighbours are learning too), which is the central difficulty of IQL. "
             "<b>Cooperation</b> is added in two ways without a central controller (centralised training would not scale): (a) <i>information sharing</i> - each agent observes how full "
             "the approaches it feeds into are; (b) <i>reward sharing</i> - r~<sub>i</sub> = 0.7 r<sub>i</sub> + 0.3 &middot; mean<sub>j&isin;N(i)</sub> r<sub>j</sub>, so a junction is penalised "
             "for pushing congestion onto its neighbours. Plain IQL (no sharing), Fixed-time (20 s green), Random and Longest-queue-first are compared on identical demand."),
           P("5. Objective of optimisation", h),
           P("Each agent maximises E[&Sigma;<sub>t</sub> &gamma;<sup>t</sup> r<sub>i,t</sub>]; jointly this approximates the network goal of "
             "<b>minimising average vehicle waiting time and queue length while maximising throughput</b>: min &pi; E[ (1/|V|) &Sigma;<sub>v</sub> W<sub>v</sub> ] with W<sub>v</sub> = total time vehicle v spends queued. "
             "Evaluation metrics: average waiting time, average queue length, throughput, average travel time, congestion index.")]
    doc.build(el)


# ============================================================================ full report
def fig(path, width_cm, caption, st):
    from PIL import Image as PI
    w, h = PI.open(path).size
    img = Image(path, width=width_cm * cm, height=width_cm * cm * h / w)
    return KeepTogether([img, Paragraph(caption, st)])


def report(N):
    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["Normal"], fontName="Times-Roman", fontSize=10.5, leading=14, alignment=TA_JUSTIFY, spaceAfter=5)
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=14, textColor=colors.HexColor("#1c2b4a"), spaceBefore=10, spaceAfter=5)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=11, textColor=colors.HexColor("#1c2b4a"), spaceBefore=6, spaceAfter=3)
    cap = ParagraphStyle("cap", parent=body, fontName="Times-Italic", fontSize=9, alignment=TA_CENTER, spaceAfter=8)
    cell = ParagraphStyle("cell", parent=body, fontName="Helvetica", fontSize=8, leading=9.5, alignment=0, spaceAfter=0)
    P = lambda s, st=body: Paragraph(s, st)
    o, per, pr = N["overall"], N["per"], N["paired"]
    W = lambda n: o[n]["avg_wait_s"][0]
    iql, coop, fx, lq, rnd = "IQL", "Coop-IQL (proposed)", "Fixed-time", "Longest-queue-first", "Random"
    imp_iql_fx = -pct(W(iql), W(fx)); imp_iql_lq = -pct(W(iql), W(lq)); imp_iql_rnd = -pct(W(iql), W(rnd))
    peak_bad = sum(1 for v in N["peak"][coop] if v > 70); peak_bad_iql = sum(1 for v in N["peak"][iql] if v > 70)
    ncu = len(N["peak"][coop])

    def table_overall():
        hdr = ["Controller", "Avg wait (s)", "Avg queue (veh/approach)", "Throughput (veh)", "Avg travel time (s)", "Congestion index", "Spill-back events"]
        data = [[P(f"<b>{x}</b>", cell) for x in hdr]]
        for n in ORDER:
            d = o[n]
            data.append([P(n, cell), f"{d['avg_wait_s'][0]:.1f}", f"{d['avg_queue_len'][0]:.2f}", f"{d['throughput'][0]:.0f}",
                         f"{d['avg_travel_time_s'][0]:.1f}", f"{d['congestion_index'][0]:.3f}", f"{d['spillback_events'][0]:.0f}"])
        t = Table(data, colWidths=[3.6 * cm, 2 * cm, 2.6 * cm, 2.3 * cm, 2.4 * cm, 2.2 * cm, 2.2 * cm], repeatRows=1)
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe9f7")),
                               ("FONTSIZE", (0, 1), (-1, -1), 8), ("ALIGN", (1, 1), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        return t

    def table_scen(metric, fmt):
        data = [[P("<b>Controller</b>", cell)] + [P(f"<b>{s}</b>", cell) for s in C.SCENARIOS]]
        for n in ORDER:
            data.append([P(n, cell)] + [fmt(per[n][s][metric][0]) for s in C.SCENARIOS])
        t = Table(data, colWidths=[4 * cm] + [2.4 * cm] * 5, repeatRows=1)
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe9f7")),
                               ("FONTSIZE", (0, 1), (-1, -1), 8), ("ALIGN", (1, 1), (-1, -1), "CENTER")]))
        return t

    def kv(rows, widths):
        t = Table([[P(a, cell), P(b, cell)] for a, b in rows], colWidths=widths)
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f3f8"))]))
        return t

    def page_deco(canvas, doc):
        canvas.saveState(); canvas.setFont("Helvetica", 8); canvas.setFillColor(colors.grey)
        canvas.drawString(2 * cm, 1.1 * cm, "Intelligent Traffic Signal Coordination Using Multi-Agent Reinforcement Learning - Technical Report")
        canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(OUT_REPORT, pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
                            title="Intelligent Traffic Signal Coordination Using Multi-Agent Reinforcement Learning")
    E = []
    # ---- title page
    E += [Spacer(1, 3.6 * cm), P("<font size=22><b>Intelligent Traffic Signal Coordination Using Multi-Agent Reinforcement Learning</b></font>",
                               ParagraphStyle("tt", parent=body, alignment=TA_CENTER, leading=28, fontName="Helvetica-Bold")),
          Spacer(1, 0.6 * cm), P("<i>A Multi-Agent Reinforcement Learning Approach to Adaptive Urban Traffic Management and Congestion Optimization</i>",
                               ParagraphStyle("t2", parent=body, alignment=TA_CENTER, fontSize=12.5, leading=17)),
          Spacer(1, 0.9 * cm), P("<font size=13>Technical Project Report</font>", ParagraphStyle("t3", parent=body, alignment=TA_CENTER)),
          Spacer(1, 1.4 * cm),
          P("<b>Contributors</b><br/>M. Sai Teja Sree Veda (100523729038) &middot; A. Maneesha (100523729039)<br/>R. Sravya (100523729052) &middot; Yusra Fateen (100523729062)",
            ParagraphStyle("t4", parent=body, alignment=TA_CENTER, leading=16, fontSize=11)),
          Spacer(1, 3 * cm),
          P("<i>All results in this report were produced by the accompanying code on a custom Python traffic simulator (SUMO and CityFlow were not used). "
            "The system has only been evaluated in simulation; it has not been deployed or tested on real roads.</i>", ParagraphStyle("t5", parent=body, alignment=TA_CENTER, fontSize=9.5)),
          PageBreak()]
    # ---- abstract
    E += [P("Abstract", h1),
          P(f"Fixed-time traffic signals cannot react to changing demand. This project formulates network-wide signal control as a cooperative multi-agent reinforcement learning (MARL) problem and implements it end to end in Python. "
            f"A custom queue-based simulator of a 3x3 grid of signalised intersections is controlled by nine tabular Q-learning agents, one per junction, trained for 1000 episodes on randomly drawn demand scenarios (five independent training runs per method). "
            f"Two variants are studied: Independent Q-Learning (IQL) and a cooperative variant (Coop-IQL) that shares downstream-congestion information and part of the reward between neighbours. "
            f"They are compared with Random, Fixed-time and Longest-queue-first controllers on 50 unseen, identical demand realisations (5 scenarios x 10 seeds). "
            f"IQL reduced the average waiting time to {W(iql):.1f} s versus {W(fx):.1f} s for fixed-time control ({imp_iql_fx:.1f}% lower) and {W(lq):.1f} s for the longest-queue heuristic ({imp_iql_lq:.1f}% lower), with the largest gains under asymmetric and high demand. "
            f"The proposed cooperative variant did <b>not</b> improve on IQL in our experiments ({W(coop):.1f} s on average): it matched IQL in {ncu - peak_bad} of {ncu} training runs but became unstable under the time-varying peak scenario in {peak_bad} of {ncu} runs. "
            f"We report this negative result, discuss likely causes, and list the limitations of the simulator and of the study.")]
    # ---- 1 intro
    E += [P("1. Introduction", h1),
          P("Road congestion increases travel times, fuel use and emissions. A large part of urban delay arises at signalised intersections, where the timing plan decides who waits. "
            "Most deployed signals use fixed-time plans or simple vehicle-actuated rules. Fixed-time plans are tuned to an average day and cannot adapt to incidents, events or rush-hour imbalance. "
            "Reinforcement learning (RL) offers a way to learn a control policy directly from interaction with traffic, using only a reward signal that expresses what 'good traffic' means."),
          P("Because a city has many intersections, each signal can be treated as an agent. Neighbouring agents influence each other: a long green at one junction releases a platoon that arrives at the next, and a full downstream link blocks the upstream junction. "
            "This makes traffic-signal coordination a canonical multi-agent RL problem.")]
    E += [P("2. Problem Definition", h1),
          P("<i>Given a network of interconnected signalised intersections with stochastic vehicle arrivals, learn a decentralised signal-control policy - one agent per intersection - that reduces average vehicle waiting time, queue length and congestion, "
            "and increases throughput, compared with fixed-time control, using only locally observable information and limited communication with neighbouring agents.</i>")]
    E += [P("3. Objectives", h1),
          P("(1) Build a reproducible multi-intersection simulator with variable demand. (2) Implement independent RL agents with state, action and reward definitions suited to traffic. "
            "(3) Add cooperation between neighbouring agents through information and reward sharing. (4) Compare against Random, Fixed-time and Longest-queue-first baselines on identical traffic. "
            "(5) Measure waiting time, queue length, throughput, travel time and congestion, and report the results - including unfavourable ones - honestly.")]
    E += [P("4. Related Technical Concepts", h1),
          P("<b>Classical signal control.</b> Webster (1958) derived fixed-time cycle lengths and splits that minimise delay for a given average flow. Actuated and adaptive systems extend this idea with detectors. "
            "Max-pressure control (Varaiya, 2013) is a decentralised rule with stability guarantees that serves the phase with the largest pressure (upstream minus downstream queue). "
            "Our Longest-queue-first baseline is a simpler actuated heuristic in the same spirit."),
          P("<b>RL for signals.</b> Early work applied Q-learning to a single junction (Abdulhai et al., 2003) and to several junctions with cooperating agents (Wiering, 2000). "
            "Tan (1993) discussed independent versus cooperative learners, the basis of the IQL and cooperative variants here. More recent work uses deep networks, for example PressLight (Wei et al., 2019) and "
            "multi-agent actor-critic for large networks (Chu et al., 2019); Wei et al. (2019) also provide a survey. SUMO (Lopez et al., 2018) is the usual open-source simulator in this literature. "
            "We use tabular Q-learning, which is simple and transparent but less powerful than those deep methods.")]
    E += [P("Reinforcement learning fundamentals", h2),
          P("An RL problem is a Markov decision process (MDP) (S, A, P, R, &gamma;): at each step the agent observes state s, chooses action a, receives reward r and moves to s' with probability P(s'|s,a). "
            "The goal is a policy &pi;(a|s) maximising the expected discounted return G = &Sigma; &gamma;<super>t</super> r<sub>t</sub> (Sutton &amp; Barto, 2018). The action-value function Q(s,a) is the expected return of taking a in s and following &pi; afterwards."),
          P("<b>Q-learning</b> (Watkins &amp; Dayan, 1992) is an off-policy temporal-difference method: Q(s,a) &larr; Q(s,a) + &alpha;[r + &gamma; max<sub>a'</sub>Q(s',a') &minus; Q(s,a)]. "
            "Exploration uses &epsilon;-greedy action selection. In a <b>multi-agent</b> setting, each agent sees an environment that changes as the other agents learn (non-stationarity), so the single-agent convergence guarantee of Q-learning no longer applies. "
            "In <b>Independent Q-Learning</b> every agent simply ignores this and learns as if alone; it often works in practice but can be unstable.")]
    E += [P("5. System Architecture", h1),
          fig(os.path.join(RD, "architecture_diagram.png"), 15.5, "Figure 1. System architecture and files.", cap),
          P("The demand generator pre-computes every vehicle arrival of an episode from a seed, so all controllers face exactly the same vehicles. The simulator advances in 1 s steps; agents are queried every 10 s.")]
    # ---- 6 MDP
    E += [P("6. Reinforcement Learning Methodology", h1), P("6.1 MDP formulation", h2),
          P("Each intersection i is an agent in a decentralised, partially observable multi-agent MDP (a Markov game): its observation covers only its own approaches, plus (for Coop-IQL) two flags about downstream approaches. "
            "Table 1 summarises the formulation; Figure 2 shows how neighbouring agents interact."),
          kv([("<b>State</b>", "Own queue on the N+S axis and on the E+W axis, each binned to 4 levels {0-3, 4-9, 10-19, 20+}; current phase (2); longest head-of-queue waiting time on the red axis binned to 3 levels {&lt;15 s, 15-40 s, &gt;40 s}. "
                                "= 96 states (IQL). Coop-IQL appends two binary flags (per axis: is any downstream approach it feeds &ge;60% full, i.e. &ge;12 of 20 vehicles?) = 384 states."),
              ("<b>Action</b>", "Phase for the next 10 s: 0 = north-south green, 1 = east-west green. Changing phase inserts 2 s of all-red (lost time)."),
              ("<b>Reward</b>", "r<sub>i</sub> = &minus;(0.10 q<sub>avg,i</sub> + 0.50 w<sub>max,i</sub>/60 + 0.50 min(b<sub>i</sub>/5, 2)). q<sub>avg,i</sub>: mean queued vehicles at junction i during the 10 s; w<sub>max,i</sub>: longest head-of-queue wait; b<sub>i</sub>: blocked discharge attempts (spill-back)."),
              ("<b>Shared reward (Coop-IQL)</b>", "r~<sub>i</sub> = 0.7 r<sub>i</sub> + 0.3 &middot; mean<sub>j&isin;N(i)</sub> r<sub>j</sub>, N(i) = adjacent junctions."),
              ("<b>Transitions</b>", "Stochastic (random arrivals, other agents' behaviour). Time-limit episodes of 180 decisions; the update bootstraps at the last step."),
              ("<b>Discount</b>", "&gamma; = 0.9 (about a 100 s horizon)."),
              ("<b>Objective</b>", "Maximise E[&Sigma; &gamma;<sup>t</sup> r<sub>i,t</sub>] for every i; network goal: minimise mean vehicle waiting time.")], [3.6 * cm, 12.8 * cm]),
          P("Table 1. MDP formulation for each agent.", cap),
          fig(os.path.join(RD, "agent_interaction_diagram.png"), 9.5, "Figure 2. Neighbouring agents share information and part of their reward; each keeps its own Q-table.", cap)]
    # ---- 7 methodology / 8 architecture
    E += [P("6.2 Algorithm and training procedure", h2),
          P("<b>Algorithm.</b> Nine tabular Q-learners, one per intersection, with separate Q-tables (intersections on the border and at the centre face different traffic, so sharing one table was not assumed). "
            "All agents act simultaneously every 10 s. Hyper-parameters: &alpha; = 0.1, &gamma; = 0.9, &epsilon; decayed linearly from 1.0 to 0.05 over the first 700 of 1000 episodes, "
            "co-operation weight &beta; = 0.3. Training episodes use a randomly drawn scenario (low, medium, high, asym, peak) and a random demand seed from 0-999, so the policy must generalise across demand patterns. "
            "Test demand uses seeds 1000-1009, never seen in training; validation seeds 5000-5001 are used to monitor training."),
          P("<b>Cooperation.</b> Coop-IQL differs from IQL in exactly two ways: the state includes the downstream-fullness flags (information sharing) and the reward blends in the neighbours' rewards (reward sharing). "
            "It is a decentralised design: no central Q-function or central controller is used."),
          P("<b>Baselines.</b> <i>Random</i>: a uniformly random phase every 10 s. <i>Fixed-time</i>: all junctions alternate 20 s NS / 20 s EW (40 s cycle, no offsets). "
            "<i>Longest-queue-first</i>: switch to the axis with the larger queue when it exceeds the current axis by more than 2 vehicles (extra, non-learning reference).")]
    E += [P("7. Implementation", h1),
          P("The code is plain Python 3 with NumPy and Matplotlib (no paid APIs). Modules: <font name='Courier'>config.py</font> (all constants), <font name='Courier'>traffic_env.py</font> (simulator), "
            "<font name='Courier'>agents.py</font> (baselines and Q-learners), <font name='Courier'>train.py</font>, <font name='Courier'>evaluate.py</font>, <font name='Courier'>make_results.py</font> (charts and tables), "
            "<font name='Courier'>visualize.py</font> (animation), <font name='Courier'>tests/test_env.py</font> (5 unit tests, e.g. vehicle conservation and queue-capacity checks), <font name='Courier'>run_all.py</font> (one-command pipeline)."),
          P("<b>Simulator model.</b> Each approach (vehicle moving S, N, E or W) has a FIFO queue; a green approach releases one vehicle every 2 s (saturation flow 0.5 veh/s). The vehicle then spends 8 s on the next link. "
            "A vehicle may enter an internal approach only if queue + vehicles on the link &lt; 20, otherwise it waits (spill-back). Routes are dimension-ordered (columns, then rows) to a random boundary exit; vehicles do not re-route. "
            "Left turns, pedestrians, yellow phases and vehicle dynamics are not modelled. "
            "Demand per entry point (veh/s): low 0.07, medium 0.10, high 0.13; <i>asym</i> 0.16 for southbound/eastbound and 0.05 for the other entries; <i>peak</i> rising linearly 0.05 &rarr; 0.18 at mid-episode and back to 0.05 (about 1500-2800 vehicles per 30 min episode). "
            "The demand levels were chosen so that fixed-time control is comfortable under low demand and strained under peak demand.")]
    E += [P("8. Experimental Setup", h1),
          kv([("Training", f"{C.TRAIN_EPISODES} episodes x 5 independent runs (seeds 0-4) for each of IQL and Coop-IQL; random scenario per episode."),
              ("Evaluation", "50 demand realisations (5 scenarios x 10 seeds, 1000-1009), 1800 s each; the same vehicle arrivals for every controller. "
                             "RL controllers are greedy (&epsilon; = 0); each of the 5 trained runs is evaluated on all 50 realisations (250 episodes per RL method), baselines 50 episodes."),
              ("Metrics", "<b>Average waiting time</b>: mean over all vehicles of time spent in a queue (including vehicles still queued at the end). <b>Average queue length</b>: time-averaged vehicles per approach. "
                          "<b>Throughput</b>: vehicles that left the network in 1800 s. <b>Average travel time</b>: mean over completed trips. <b>Congestion index</b>: time-averaged share of approaches holding &ge;10 queued vehicles. "
                          "<b>Spill-back events</b>: blocked discharge attempts."),
              ("Hardware", "Single CPU core. Training of all 10 runs took about 10 minutes; evaluation about a minute.")], [3 * cm, 13.4 * cm]),
          Spacer(1, 6)]
    # ---- results
    E += [P("9. Results and Discussion", h1),
          P("<b>Learning curves.</b> Figure 3 (left) shows the episode return during training: both RL variants improve from about &minus;300 to about &minus;100 and then plateau, with nearly identical curves. "
            "The training return is noisy because every episode uses a different random scenario. Figure 3 (right) shows the greedy validation waiting time: a policy that is already competitive (&asymp;33-35 s) appears after only about 50 episodes. "
            "Late in training the validation curves oscillate; with a constant learning rate and non-stationary neighbours, individual runs sometimes drift to worse policies, and the 'best' checkpoint is not the last. "
            "We report the final policy of every run (no cherry-picking of checkpoints)."),
          fig(os.path.join(RD, "learning_curves.png"), 16, "Figure 3. Left: training return (mean +/- std over 5 runs, 20-episode moving average). Right: greedy validation waiting time during training versus baselines.", cap),
          P("<b>Overall comparison.</b> Table 2 and Figure 4 average over all scenarios and test runs."),
          table_overall(),
          P(f"Table 2. Mean results over 5 scenarios x 10 test seeds (RL rows also averaged over 5 training runs).", cap),
          fig(os.path.join(RD, "comparison_overall.png"), 16.2, "Figure 4. Overall comparison of all controllers on the five metrics (plus spill-back events).", cap),
          P(f"IQL achieved the lowest average waiting time ({W(iql):.1f} s), {imp_iql_fx:.1f}% below Fixed-time and {imp_iql_lq:.1f}% below Longest-queue-first, and {imp_iql_rnd:.1f}% below Random. "
            f"Its average queue length ({o[iql]['avg_queue_len'][0]:.2f} vehicles per approach) and travel time ({o[iql]['avg_travel_time_s'][0]:.1f} s) were also the lowest. "
            f"Throughput differs little between sensible controllers (about {o[fx]['throughput'][0]:.0f}-{o[lq]['throughput'][0]:.0f} vehicles), because in 30 minutes almost all vehicles eventually get through when the network is not gridlocked; "
            f"the benefit of good control appears mainly in waiting time and queues, not in total vehicles served. Random control was clearly the worst."),
          P("<b>By scenario.</b> Table 3 and Figures 5-7 show that the advantage depends on demand.")]
    E += [table_scen("avg_wait_s", lambda v: f"{v:.1f}"), P("Table 3. Average waiting time (s) by demand scenario.", cap),
          table_scen("throughput", lambda v: f"{v:.0f}"), P("Table 4. Throughput (vehicles in 30 min) by demand scenario.", cap),
          fig(os.path.join(RD, "waiting_time_by_scenario.png"), 14.5, "Figure 5. Average waiting time by demand scenario.", cap),
          fig(os.path.join(RD, "queue_length_by_scenario.png"), 14.5, "Figure 6. Average queue length by demand scenario.", cap),
          fig(os.path.join(RD, "throughput_by_scenario.png"), 14.5, "Figure 7. Throughput by demand scenario (the y-axis starts at 80% of the smallest value).", cap)]
    pf = lambda s: per[s]
    E += [P(f"Under low and medium demand, Fixed-time and IQL are almost equivalent ({per[fx]['low']['avg_wait_s'][0]:.1f} vs {per[iql]['low']['avg_wait_s'][0]:.1f} s at low demand), and Longest-queue-first is actually worse than Fixed-time there because its switching causes extra lost time. "
            f"Learning pays off when demand is high or unbalanced: under <i>asym</i> demand IQL reached {per[iql]['asym']['avg_wait_s'][0]:.1f} s versus {per[fx]['asym']['avg_wait_s'][0]:.1f} s for Fixed-time "
            f"({-pct(per[iql]['asym']['avg_wait_s'][0], per[fx]['asym']['avg_wait_s'][0]):.0f}% lower), because the learned policy gives more green to the heavy direction. Under <i>peak</i> demand IQL reduced waiting from {per[fx]['peak']['avg_wait_s'][0]:.1f} to {per[iql]['peak']['avg_wait_s'][0]:.1f} s."),
          P("<b>Cooperative variant - an unexpected result.</b> "
            f"Coop-IQL averaged {W(coop):.1f} s, i.e. no better than Fixed-time (paired difference {pr['Coop-IQL minus Fixed-time']['mean_diff_wait_s']:+.1f} s, 95% bootstrap CI [{pr['Coop-IQL minus Fixed-time']['ci95_low']:.1f}, {pr['Coop-IQL minus Fixed-time']['ci95_high']:.1f}]) and worse than IQL "
            f"({pr['Coop-IQL minus IQL']['mean_diff_wait_s']:+.1f} s, CI [{pr['Coop-IQL minus IQL']['ci95_low']:.1f}, {pr['Coop-IQL minus IQL']['ci95_high']:.1f}]). "
            f"The difference is not uniform: in the low, medium, high and asym scenarios Coop-IQL is within about 1 s of IQL, but under <i>peak</i> demand {peak_bad} of {ncu} Coop-IQL training runs produced policies with waiting times of about 80 s, "
            f"while {peak_bad_iql} of {ncu} IQL runs did (per-run peak waiting times: Coop-IQL {', '.join(f'{v:.0f}' for v in N['peak'][coop])} s; IQL {', '.join(f'{v:.0f}' for v in N['peak'][iql])} s). "
            "So the average hides a bimodal outcome: some runs work as well as IQL and others do not. We did not find evidence that the cooperation mechanisms help. Plausible but <i>untested</i> explanations: "
            "(i) the larger state space (384 vs 96 states) is visited less often, especially the rarely seen 'downstream full' states that matter only during peaks; "
            "(ii) reward sharing makes each agent's reward noisier and couples learning across agents, adding to non-stationarity; "
            "(iii) the downstream flag is a coarse 0/1 signal. We did not run ablations that separate (a) from (b), so we cannot say which one hurts."),
          P("<b>Design iteration (full disclosure).</b> A first version of Coop-IQL, using the mean total queue of neighbouring junctions as shared information and 400 training episodes, was evaluated first "
            f"(avg. waiting time: IQL {N['v1']['IQL']:.1f} s, Coop-IQL {N['v1']['Coop-IQL (proposed)']:.1f} s; archived in <font name='Courier'>05_Experimental_Results/iteration1_v1_400ep</font>). "
            "Because it also failed in the peak scenario, we changed the shared information to downstream-fullness flags and trained for 1000 episodes (both methods), using validation seeds to check the change. "
            "Note that this redesign was motivated by seeing the first test-set results, so the test seeds are not completely untouched; the improvement for IQL from 400 to 1000 episodes is part of the same change. "
            "The final design is still outperformed by plain IQL.")]
    E += [P("10. Limitations", h1),
          P("(1) <b>Simulator realism:</b> a custom queue-based model, not SUMO; no vehicle dynamics, lane changing, left turns, pedestrians, yellow phases, or re-routing, and a single network size. Absolute numbers (e.g. seconds of waiting) are properties of this simulator and must not be read as real-world predictions. "
            "(2) <b>Algorithm:</b> tabular Q-learning with hand-chosen bins; no deep RL; constant learning rate; hyper-parameters and reward weights were not tuned systematically. "
            "(3) <b>Baselines:</b> the Fixed-time plan was not optimised (no Webster calculation, no offsets/green wave), so a tuned fixed plan may be stronger than the one used here. Max-pressure was not implemented. "
            "(4) <b>Statistics:</b> 5 training runs and 50 test episodes per scenario set; the bimodal Coop-IQL outcome shows that 5 runs are few. Throughput differences between sensible controllers are small relative to scenario variance. "
            "(5) <b>Evaluation protocol:</b> test seeds were used once before redesign (Section 9). (6) The 'travel time' averages only vehicles that completed their trip, which flatters controllers that leave many vehicles in the network; the counts of vehicles left are stored in the raw results.")]
    E += [P("11. Safety and Ethical Considerations", h1),
          P("<b>Safety and reliability.</b> A learned controller must never compromise safety: real deployments need hard-coded safety layers (minimum green, clearance and conflict monitors, fail-safe flashing or fixed-time fallback when sensors or software fail). "
            "Our agents have no such layer; a learned policy can fail abruptly in situations not seen in training, as the peak-scenario failures of some runs illustrate. "
            "<b>Fairness.</b> Minimising average waiting time can starve minor roads, pedestrians, cyclists and buses; we include a longest-wait term in the reward, but a real system should include these road users explicitly and audit who waits longest. "
            "<b>Emergency vehicles</b> need priority mechanisms. <b>Privacy.</b> Queue counts from loop detectors are less sensitive than camera or licence-plate data; any camera-based sensing must follow data-protection law. "
            "<b>Accountability and transparency.</b> Operators need explainable behaviour, logs, and a clear responsibility when a learned policy contributes to an accident or to inequitable delays. "
            "<b>Claims.</b> This work is a simulation study; it makes no claim about performance on real roads.")]
    E += [P("12. Future Scope", h1),
          P("Recommended next steps: SUMO or CityFlow validation, Webster-tuned and max-pressure baselines, function approximation (DQN/actor-critic) with parameter sharing, ablations of information versus reward sharing, more training seeds, and safety layers.")]
    E += [P("13. Conclusion", h1),
          P(f"We implemented and evaluated multi-agent Q-learning for coordinated traffic-signal control in a 3x3 simulated network. Independent Q-learning learned within a few hundred episodes a policy that lowered waiting time by "
            f"{imp_iql_fx:.0f}% compared with a 20-s fixed-time plan overall and by about {-pct(per[iql]['asym']['avg_wait_s'][0], per[fx]['asym']['avg_wait_s'][0]):.0f}% under asymmetric demand, but offered little benefit at low demand and was roughly on par with a simple longest-queue heuristic "
            f"({imp_iql_lq:.0f}% better overall). The cooperative variant we proposed did not improve on IQL and was less reliable under peak demand. The main lessons are that (i) even simple tabular MARL can beat fixed-time control when demand is uneven, "
            "(ii) cooperation is not automatically beneficial and needs careful design and ablation, and (iii) reliability across training seeds matters as much as average performance.")]
    E += [P("References", h1)]
    refs = ["Abdulhai, B., Pringle, R., &amp; Karakoulas, G. J. (2003). Reinforcement learning for true adaptive traffic signal control. <i>Journal of Transportation Engineering</i>, 129(3), 278-285.",
            "Chu, T., Wang, J., Codeca, L., &amp; Li, Z. (2019). Multi-agent deep reinforcement learning for large-scale traffic signal control. <i>IEEE Transactions on Intelligent Transportation Systems</i>, 21(3), 1086-1095.",
            "Lopez, P. A., et al. (2018). Microscopic traffic simulation using SUMO. <i>Proc. IEEE International Conference on Intelligent Transportation Systems (ITSC)</i>, 2575-2582.",
            "Sutton, R. S., &amp; Barto, A. G. (2018). <i>Reinforcement Learning: An Introduction</i> (2nd ed.). MIT Press.",
            "Tan, M. (1993). Multi-agent reinforcement learning: Independent vs. cooperative agents. <i>Proc. 10th International Conference on Machine Learning</i>, 330-337.",
            "Varaiya, P. (2013). Max pressure control of a network of signalized intersections. <i>Transportation Research Part C</i>, 36, 177-195.",
            "Watkins, C. J. C. H., &amp; Dayan, P. (1992). Q-learning. <i>Machine Learning</i>, 8, 279-292.",
            "Webster, F. V. (1958). <i>Traffic signal settings</i> (Road Research Technical Paper No. 39). Her Majesty's Stationery Office, London.",
            "Wei, H., Zheng, G., Gayah, V., &amp; Li, Z. (2019). A survey on traffic signal control methods. <i>arXiv:1904.08117</i>.",
            "Wei, H., Chen, C., Zheng, G., Wu, K., Gayah, V., Xu, K., &amp; Li, Z. (2019). PressLight: Learning max pressure control to coordinate traffic signals in arterial network. <i>Proc. ACM SIGKDD</i>, 1290-1298.",
            "Wiering, M. (2000). Multi-agent reinforcement learning for traffic light control. <i>Proc. 17th International Conference on Machine Learning</i>, 1151-1158."]
    for r in refs:
        E.append(P(r, ParagraphStyle("ref", parent=body, leftIndent=14, firstLineIndent=-14, fontSize=9.5, leading=12, spaceAfter=3)))
    doc.build(E, onFirstPage=lambda c, d: None, onLaterPages=page_deco)


if __name__ == "__main__":
    N = load_numbers()
    one_page(N)
    report(N)
    print("documents built")
