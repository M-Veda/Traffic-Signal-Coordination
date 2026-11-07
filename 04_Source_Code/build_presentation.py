"""Builds the exactly-five-slide PowerPoint using the real experiment outputs."""
import json, os, sys
import numpy as np
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
import config as C
from make_results import aggregate, ORDER

RD = sys.argv[1] if len(sys.argv) > 1 else os.path.join("..", "05_Experimental_Results")
OUT = os.path.join("..", "03_Presentation", "RL_Traffic_Signal_Coordination_5_Slides.pptx")
NAVY, ACC, LIGHT, GREY, RED, GREEN = (RGBColor(0x1C, 0x2B, 0x4A), RGBColor(0x2E, 0x86, 0xC1), RGBColor(0xEE, 0xF3, 0xFA),
                                       RGBColor(0x55, 0x55, 0x55), RGBColor(0xC0, 0x39, 0x2B), RGBColor(0x1E, 0x84, 0x49))

res = json.load(open(os.path.join(RD, "evaluation_raw.json")))
overall, per = aggregate(res)
W = lambda n: overall[n]["avg_wait_s"][0]
imp = lambda a, b: 100 * (W(b) - W(a)) / W(b)

prs = Presentation(); prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
blank = prs.slide_layouts[6]


def add_text(slide, x, y, w, h, paras, size=14, color=RGBColor(0x22, 0x22, 0x22), bold=False, fill=None, anchor=MSO_ANCHOR.TOP, line=None):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE if fill else MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    if fill: shp.fill.solid(); shp.fill.fore_color.rgb = fill
    else: shp.fill.background()
    shp.shadow.inherit = False
    if line: shp.line.color.rgb = line; shp.line.width = Pt(1)
    else: shp.line.fill.background()
    tf = shp.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.12); tf.margin_top = tf.margin_bottom = Inches(0.06)
    first = True
    for p in paras:
        if isinstance(p, str): p = {"t": p}
        para = tf.paragraphs[0] if first else tf.add_paragraph(); first = False
        r = para.add_run(); r.text = p["t"]; r.font.size = Pt(p.get("size", size)); r.font.bold = p.get("bold", bold)
        r.font.color.rgb = p.get("color", color); r.font.name = "Calibri"; para.space_after = Pt(p.get("after", 4)); para.alignment = p.get("align", PP_ALIGN.LEFT)
    return shp


def header(slide, n, title):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(1.0)); bar.fill.solid(); bar.fill.fore_color.rgb = NAVY; bar.line.fill.background()
    add_text(slide, 0.4, 0.12, 11.5, 0.8, [title], size=28, color=RGBColor(255, 255, 255), bold=True, anchor=MSO_ANCHOR.MIDDLE)
    add_text(slide, 11.9, 0.12, 1.3, 0.8, [f"{n} / 5"], size=14, color=RGBColor(0xBB, 0xD0, 0xEA), anchor=MSO_ANCHOR.MIDDLE)
    add_text(slide, 0.4, 7.1, 12.5, 0.3, ["Multi-Agent RL Traffic Signal Coordination  |  RL Homework (Unit 5, Project 10)  |  all results from simulation"], size=10, color=GREY)


def pic(slide, name, x, y, w=None, h=None):
    return slide.shapes.add_picture(os.path.join(RD, name), Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None)


def bullets(items, size=15):
    out = []
    for it in items:
        if isinstance(it, tuple):
            out.append({"t": it[0], "bold": True, "size": size, "color": NAVY, "after": 1}); out.append({"t": it[1], "size": size - 1, "after": 7})
        else:
            out.append({"t": "\u2022 " + it, "size": size, "after": 6})
    return out


# ---------------------------------------------------------------- slide 1
s = prs.slides.add_slide(blank); header(s, 1, "Motivation and Problem Statement")
add_text(s, 0.4, 1.2, 6.3, 2.5, [{"t": "Urban traffic congestion", "bold": True, "size": 18, "color": NAVY},
    "\u2022 Delay, fuel use and emissions concentrate at signalised intersections",
    "\u2022 Queues at one junction spill back and block the next, so signals must be coordinated, not tuned one by one"], size=15)
add_text(s, 0.4, 3.35, 6.3, 2.0, [{"t": "Why conventional signals are not enough", "bold": True, "size": 18, "color": NAVY},
    "\u2022 Fixed-time plans repeat a set cycle whatever the demand",
    "\u2022 Green time is wasted on empty approaches; rush-hour imbalance is not handled"], size=15)
add_text(s, 0.4, 5.15, 6.3, 1.9, [{"t": "Project objectives", "bold": True, "size": 18, "color": NAVY},
    "1. One RL agent per signal in a 3x3 simulated grid",
    "2. Neighbours share information and reward",
    "3. Compare with Fixed-time, Random and Longest-queue baselines on identical traffic"], size=14, fill=LIGHT)
pic(s, "simulation_iql_asym_snapshot.png", 7.0, 1.2, h=5.1)
add_text(s, 7.0, 6.35, 6, 0.6, ["Our simulator (frame from the animation): vehicles queue at red lights, green lights release them."], size=11, color=GREY)

# ---------------------------------------------------------------- slide 2
s = prs.slides.add_slide(blank); header(s, 2, "Proposed Method: Multi-Agent Q-Learning")
rows = [("State", "Queue on N+S and E+W axes (4 levels each), phase, longest wait of red axis (3 levels); Coop-IQL adds 2 'downstream full' flags"),
        ("Action", "Next 10 s: North-South green or East-West green (2 s all-red when switching)"),
        ("Reward", "-(0.10 x avg queue + 0.50 x longest wait/60 s + 0.50 x spill-back)  penalises queues, waiting, congestion"),
        ("Algorithm", "Independent Q-Learning, 9 agents, own Q-table each; alpha 0.1, gamma 0.9, epsilon 1.0 to 0.05"),
        ("Coordination", "Information sharing (downstream occupancy) + reward sharing: 0.7 own + 0.3 mean of neighbours")]
y = 1.2
for k, v in rows:
    add_text(s, 0.4, y, 1.9, 0.95, [k], size=15, color=RGBColor(255, 255, 255), bold=True, fill=ACC, anchor=MSO_ANCHOR.MIDDLE)
    add_text(s, 2.3, y, 5.4, 0.95, [v], size=12.5, fill=LIGHT, anchor=MSO_ANCHOR.MIDDLE)
    y += 1.05
pic(s, "agent_interaction_diagram.png", 8.0, 1.3, w=5.0)
add_text(s, 8.0, 4.1, 5.0, 2.8, [{"t": "How the agents interact", "bold": True, "size": 16, "color": NAVY},
    "\u2022 All agents act at the same time, so each faces a changing environment (non-stationarity)",
    "\u2022 Cooperation is decentralised: no central controller",
    "\u2022 Goal: minimise average vehicle waiting time over the whole network"], size=13)

# ---------------------------------------------------------------- slide 3
s = prs.slides.add_slide(blank); header(s, 3, "Implementation")
pic(s, "architecture_diagram.png", 0.4, 1.15, w=8.2)
add_text(s, 8.9, 1.2, 4.1, 2.3, [{"t": "Tools and technologies", "bold": True, "size": 16, "color": NAVY},
    "\u2022 Python 3, NumPy, Matplotlib", "\u2022 Custom queue-based simulator (SUMO not used)", "\u2022 python-pptx and ReportLab for documents", "\u2022 No paid APIs; runs on a laptop (Windows)"], size=13, fill=LIGHT)
add_text(s, 0.4, 4.35, 12.5, 2.6, [{"t": "Training and simulation workflow", "bold": True, "size": 16, "color": NAVY},
    "1. Each episode: draw a random demand scenario (low, medium, high, asym, peak) and seed; 1800 s simulated, one decision per agent every 10 s",
    "2. Each agent: observe state, choose phase (epsilon-greedy), receive reward, apply the Q-learning update",
    "3. Train 1000 episodes x 5 independent runs for IQL and Coop-IQL (about 10 min on one CPU core)",
    "4. Evaluate greedy policies and baselines on 50 unseen, identical demand realisations (5 scenarios x 10 seeds)",
    "5. Generate charts, tables, report; 5 unit tests check vehicle conservation, queue limits, state ranges"], size=13)

# ---------------------------------------------------------------- slide 4
s = prs.slides.add_slide(blank); header(s, 4, "Experimental Results")
pic(s, "learning_curves.png", 0.3, 1.1, w=7.6)
pic(s, "waiting_time_by_scenario.png", 0.3, 3.95, w=5.5)
tbl_rows = [["Controller", "Wait (s)", "Queue", "Thr."]]
for n in ORDER:
    d = overall[n]; tbl_rows.append([n.replace(" (proposed)", ""), f"{d['avg_wait_s'][0]:.1f}", f"{d['avg_queue_len'][0]:.2f}", f"{d['throughput'][0]:.0f}"])
gt = s.shapes.add_table(len(tbl_rows), 4, Inches(8.1), Inches(1.15), Inches(5.0), Inches(2.1)).table
for i, r in enumerate(tbl_rows):
    for j, v in enumerate(r):
        c = gt.cell(i, j); c.text = v
        for p in c.text_frame.paragraphs:
            p.font.size = Pt(12); p.font.bold = (i == 0) or (r[0] == "IQL")
            if j > 0: p.alignment = PP_ALIGN.CENTER
for j, wd in enumerate([2.0, 1.0, 1.0, 1.0]): gt.columns[j].width = Inches(wd)
add_text(s, 5.9, 4.0, 7.2, 3.05, [{"t": "Key findings (means over all test runs)", "bold": True, "size": 14, "color": NAVY},
    f"\u2022 IQL cut waiting time {imp('IQL', 'Fixed-time'):.0f}% vs Fixed-time ({W('IQL'):.1f} vs {W('Fixed-time'):.1f} s); {-per['IQL']['asym']['avg_wait_s'][0]*100/per['Fixed-time']['asym']['avg_wait_s'][0]+100:.0f}% under asym demand",
    f"\u2022 About equal to Longest-queue-first ({W('Longest-queue-first'):.1f} s); Random is far worse ({W('Random'):.0f} s)",
    "\u2022 No gain at low demand; throughput barely differs",
    f"\u2022 Coop-IQL did NOT beat IQL ({W('Coop-IQL (proposed)'):.1f} s): 3 of 5 runs unstable in the peak scenario (about 80 s) - negative result reported",
    "\u2022 Policy good after ~50 episodes; late oscillations seen"], size=14)

# ---------------------------------------------------------------- slide 5
s = prs.slides.add_slide(blank); header(s, 5, "Limitations and Ethics")
cards = [("Simulation limitations", ["Custom queue model, not SUMO; no left turns, pedestrians, vehicle dynamics", "Single 3x3 network; tabular states; only 5 training runs", "Fixed-time baseline not optimised; no max-pressure"]),
         ("Safety and reliability", ["Needs hard safety layer: min green, conflict monitor, fail-safe fallback", "Learned policy can fail in unseen demand (as peak runs did)", "Never validated on real roads"]),
         ("Ethical implications", ["Average-delay goals can starve side roads, pedestrians, buses", "Emergency vehicles need priority", "Privacy of sensing data; accountability and transparency of decisions"]),
         ("Future improvements", ["Validate in SUMO/CityFlow with real demand", "Deep RL with parameter sharing; ablate info vs reward sharing", "More seeds, tuned fixed-time and max-pressure baselines"])]
for k, (t, its) in enumerate(cards):
    x = 0.4 + (k % 2) * 6.35; y = 1.25 + (k // 2) * 2.85
    add_text(s, x, y, 6.15, 0.5, [t], size=17, color=RGBColor(255, 255, 255), bold=True, fill=ACC if k < 3 else GREEN, anchor=MSO_ANCHOR.MIDDLE)
    add_text(s, x, y + 0.5, 6.15, 2.1, [{"t": "\u2022 " + i, "size": 17, "after": 10} for i in its], size=17, fill=LIGHT)
prs.save(OUT)
print("saved", OUT, "slides:", len(prs.slides))
