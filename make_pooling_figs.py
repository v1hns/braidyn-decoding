"""Figures for the population-pooling drift-resistance paper.

Reads pooling_leakfix.json (session-held-out protocol). The earlier pooling_results.json used a
trial-level split for the personal arm and scored the count-matched self arm on its own training
data; both inflate the personal side only. See pooling_leakfix.py for the details.
"""
import json, os
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "figure.dpi": 200, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "paper_neurips", "figs"); os.makedirs(OUT, exist_ok=True)
R = json.load(open(os.path.join(HERE, "pooling_leakfix.json")))
T = ["state_lever", "lick", "reward", "tone"]
nice = {"state_lever": "lever-pull", "lick": "lick", "reward": "reward", "tone": "tone"}

fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.2, 2.8))

# LEFT: count-matched control -- aging of self / one-other / many-others at matched n
groups = [("self", "self", "#c0392b"), ("one", "one\nother", "#e59866"),
          ("many", "many\nothers", "#5d6d7e")]
x = np.arange(len(T)); w = 0.26
for i, (k, lab, c) in enumerate(groups):
    vals = [R["C"][t]["fixed"][k]["aging"] for t in T]
    err = [R["C"][t]["fixed"][k]["sem"] for t in T]
    axL.bar(x + (i - 1) * w, vals, w, yerr=err, capsize=2, color=c, label=lab,
            edgecolor="k", linewidth=0.4)
axL.axhline(0, c="gray", lw=0.7)
axL.set_xticks(x); axL.set_xticklabels([nice[t] for t in T], fontsize=7)
axL.set_ylabel("aging  (AUC lost early$\\to$late)")
axL.set_title("Count-matched: whose data ages less?\n(equal training-event count, sessions held out)",
              fontsize=8)
axL.legend(fontsize=6, ncol=3, loc="upper center", frameon=False)
axL.invert_yaxis()  # more-negative = ages less = better -> plot downward as "better"
axL.text(0.01, 0.02, "lower = ages less", transform=axL.transAxes, fontsize=6, color="#555")

# RIGHT: scaling -- pooled aging vs number of training mice N
for t in T:
    b = R["B"][t]; ns = sorted(int(k) for k in b)
    y = [b[str(n)]["pooled_aging"] for n in ns]
    axR.plot(ns, y, "o-", ms=3, lw=1, label=nice[t])
axR.axhline(0, c="gray", lw=0.7)
axR.set_xscale("log", base=2); axR.set_xticks(ns); axR.set_xticklabels(ns)
axR.set_xlabel("number of training mice $N$")
axR.set_ylabel("pooled aging")
axR.set_title("Scaling with training-set diversity", fontsize=8)
axR.legend(fontsize=6, frameon=False)

fig.savefig(os.path.join(OUT, "fig_mechanism.pdf")); plt.close(fig)
print("wrote fig_mechanism.pdf")
for t in T:
    c = R["C"][t]["fixed"]
    print(f"{t:11s} self {c['self']['aging']:+.4f} one {c['one']['aging']:+.4f} "
          f"many {c['many']['aging']:+.4f}  (many<one {c['many']['aging'] < c['one']['aging']})")
