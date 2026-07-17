"""Figures for the BraiDyn-BC cross-session drift study."""
import json, os
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "figure.dpi": 200, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})
HERE = os.path.dirname(os.path.abspath(__file__))
# Write into both paper dirs so the live ICLR draft and the archived IEEE one stay in sync.
OUTS = [os.path.join(HERE, "paper_iclr", "figs"), os.path.join(HERE, "paper", "figs")]
for _o in OUTS:
    os.makedirs(_o, exist_ok=True)
OUT = OUTS[0]
JSON = os.path.join(HERE, "braidyn_drift.json")
if not os.path.exists(JSON):
    JSON = os.path.expanduser("~/braidyn_drift.json")
R = json.load(open(JSON))["targets"]
ORDER = ["state_lever", "lick", "reward", "tone"]
targets = [t for t in ORDER if t in R] + [t for t in R if t not in ORDER]
nice = {"state_lever": "lever-pull", "lick": "lick", "reward": "reward", "tone": "tone"}


def save(fig, name):
    for o in OUTS:
        fig.savefig(os.path.join(o, name))

# Short codes match the paper's Table 2 notation; long names collide once there
# are four panels downscaled to \linewidth.
COND = [("WS_within_same", "WS", "#95a5a6"),
        ("WX_within_cross", "WX", "#5d6d7e"),
        ("LS_lomo_same", "LS", "#e08e79"),
        ("LX_lomo_cross", "LX", "#c0392b")]

# Fig 4: 2x2 drift bars per target (WS/WX/LS/LX) with cluster-bootstrap CI + per-mouse dots
fig, axes = plt.subplots(1, len(targets), figsize=(1.45 * len(targets) + 0.4, 2.6), sharey=True)
for ax, t in zip(np.atleast_1d(axes), targets):
    d = R[t]["drift_2x2"]; x = np.arange(len(COND))
    aucs = [d[k]["auc"] for k, _, _ in COND]
    err = np.array([[ (d[k]["auc"]-d[k]["ci95"][0]) for k,_,_ in COND],
                    [ (d[k]["ci95"][1]-d[k]["auc"]) for k,_,_ in COND]])
    ax.bar(x, aucs, 0.7, yerr=err, capsize=3, color=[c for _,_,c in COND])
    pm = d["per_mouse"]
    for xi, (k, _, _) in zip(x, COND):
        vals = [pm[m][k[:2]] for m in pm if pm[m][k[:2]] == pm[m][k[:2]]]  # drop NaN
        ax.scatter([xi]*len(vals), vals, s=5, color="k", alpha=0.3, zorder=3)
    ax.axhline(0.5, ls="--", c="gray", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels([lab for _, lab, _ in COND], fontsize=7)
    # Delta / p live in Table 2; annotating them here collided with the bars
    # (ylim starts at 0.4, so low axes-coords text sits behind them).
    ax.set_title(f"{nice.get(t, t)}  (n={d['n_mice']})", fontsize=8)
np.atleast_1d(axes)[0].set_ylabel("decoding AUC")
np.atleast_1d(axes)[0].set_ylim(0.4, 1.0)
fig.suptitle("Cross-session design: WS/WX = within-mouse same/early->late block, "
             "LS/LX = leave-one-mouse-out same/early->late\n"
             "The cross-animal pair (LS->LX) does not fall for any target",
             y=1.07, fontsize=7)
save(fig, "fig4_drift_2x2.pdf"); plt.close(fig)

# Fig 5: within-mouse day-gap decay curve (AUC vs gap in days), both targets
fig, ax = plt.subplots(figsize=(3.4, 2.5))
for t in targets:
    cur = R[t]["drift_curve"]["curve"]
    gaps = sorted(int(g) for g in cur)
    auc = [cur[str(g)]["auc"] for g in gaps]
    ax.plot(gaps, auc, "o-", ms=3, label=f"{nice.get(t,t)} (slope {R[t]['drift_curve']['slope_per_day']:+.4f}/day)")
ax.axhline(0.5, ls="--", c="gray", lw=0.7)
ax.set_xlabel("days between train and test session")
ax.set_ylabel("within-mouse AUC"); ax.set_ylim(0.4, 1.0)
ax.set_title("Within-mouse decoding vs day-gap")
ax.legend(fontsize=6, loc="lower left")
save(fig, "fig5_daygap.pdf"); plt.close(fig)

print("wrote:", sorted(os.listdir(OUT)))
for t in targets:
    d = R[t]["drift_2x2"]
    print(t, {k: round(d[k]["auc"], 3) if d[k]["auc"] else None for k, _, _ in COND})
