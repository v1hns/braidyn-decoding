"""Figure: linear vs nonlinear temporal decoders (within-mouse vs LOMO)."""
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


def save(fig, name):
    for o in OUTS:
        fig.savefig(os.path.join(o, name))


JSON = os.path.join(HERE, "braidyn_nonlinear.json")
if not os.path.exists(JSON):
    JSON = os.path.expanduser("~/braidyn_nonlinear.json")
R = json.load(open(JSON))["targets"]
ORDER = ["state_lever", "lick", "reward", "tone"]
targets = [t for t in ORDER if t in R] + [t for t in R if t not in ORDER]
nice = {"state_lever": "lever-pull", "lick": "lick", "reward": "reward", "tone": "tone"}
MODELS = [("linear", "linear", "#95a5a6"),
          ("cnn", "CNN", "#2c6fbb"),
          ("gru", "GRU", "#c0392b")]

# Fig 6: grouped bars, within vs LOMO per model, one panel per target
fig, axes = plt.subplots(1, len(targets), figsize=(1.45 * len(targets) + 0.4, 2.6), sharey=True)
for ax, t in zip(np.atleast_1d(axes), targets):
    x = np.arange(len(MODELS)); w = 0.38
    within = [R[t][k]["within_auc"] for k, _, _ in MODELS]
    lomo = [R[t][k]["lomo_auc"] for k, _, _ in MODELS]
    ax.bar(x - w/2, within, w, label="within-mouse", color="#d7dbdd", edgecolor="k", linewidth=0.5)
    ax.bar(x + w/2, lomo, w, label="leave-mouse-out",
           color=[c for _, _, c in MODELS], edgecolor="k", linewidth=0.5)
    # Gaps live in Table 3; annotating them here would render on top of the bars
    # (ylim starts at 0.4, so the bars occupy the low data coords).
    ax.axhline(0.5, ls="--", c="gray", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels([lab for _, lab, _ in MODELS], fontsize=7)
    ax.set_title(f"{nice.get(t, t)}  (n={R[t]['n_mice']})", fontsize=8)
np.atleast_1d(axes)[0].set_ylabel("decoding AUC")
np.atleast_1d(axes)[0].set_ylim(0.4, 1.0)
np.atleast_1d(axes)[0].legend(fontsize=6, loc="lower left")
fig.suptitle("Temporal models raise accuracy without systematically widening the cross-animal gap",
             y=1.04, fontsize=8)
save(fig, "fig6_nonlinear.pdf"); plt.close(fig)

print("wrote fig6_nonlinear.pdf")
for t in targets:
    for k, _, _ in MODELS:
        r = R[t][k]
        print(f"{t:12s} {k:6s} within {r['within_auc']:.3f} LOMO {r['lomo_auc']:.3f} "
              f"gap {r['gap_lomo_minus_within']:+.3f} p={r['gap_p']:.3f}")
