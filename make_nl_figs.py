"""Figure: linear vs nonlinear temporal decoders (within-mouse vs LOMO)."""
import json, os
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "figure.dpi": 200, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "paper", "figs"); os.makedirs(OUT, exist_ok=True)
JSON = os.path.join(HERE, "braidyn_nonlinear.json")
if not os.path.exists(JSON):
    JSON = os.path.expanduser("~/braidyn_nonlinear.json")
R = json.load(open(JSON))["targets"]
targets = list(R.keys()); nice = {"state_lever": "lever-pull", "lick": "lick"}
MODELS = [("linear", "linear\n(evoked)", "#95a5a6"),
          ("cnn", "1D-CNN\n(temporal)", "#2c6fbb"),
          ("gru", "GRU\n(temporal)", "#c0392b")]

# Fig 6: grouped bars, within vs LOMO per model, one panel per target
fig, axes = plt.subplots(1, len(targets), figsize=(6.8, 2.9), sharey=True)
for ax, t in zip(np.atleast_1d(axes), targets):
    x = np.arange(len(MODELS)); w = 0.38
    within = [R[t][k]["within_auc"] for k, _, _ in MODELS]
    lomo = [R[t][k]["lomo_auc"] for k, _, _ in MODELS]
    ax.bar(x - w/2, within, w, label="within-mouse", color="#d7dbdd", edgecolor="k", linewidth=0.5)
    ax.bar(x + w/2, lomo, w, label="leave-mouse-out",
           color=[c for _, _, c in MODELS], edgecolor="k", linewidth=0.5)
    for xi, (k, _, _) in zip(x, MODELS):
        g = R[t][k]["gap_lomo_minus_within"]; p = R[t][k]["gap_p"]
        ax.text(xi, 0.42, f"gap {g:+.3f}\np={p:.2f}", ha="center", fontsize=5.5, color="#555")
    ax.axhline(0.5, ls="--", c="gray", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels([lab for _, lab, _ in MODELS], fontsize=6.5)
    ax.set_title(nice.get(t, t) + f"  (n={R[t]['n_mice']} mice)")
axes[0].set_ylabel("decoding AUC"); axes[0].set_ylim(0.4, 1.0)
axes[0].legend(fontsize=6, loc="upper left")
fig.suptitle("Nonlinear temporal models: higher accuracy, gap stays closed", y=1.04, fontsize=8.5)
fig.savefig(f"{OUT}/fig6_nonlinear.pdf"); plt.close(fig)

print("wrote fig6_nonlinear.pdf")
for t in targets:
    for k, _, _ in MODELS:
        r = R[t][k]
        print(f"{t:12s} {k:6s} within {r['within_auc']:.3f} LOMO {r['lomo_auc']:.3f} "
              f"gap {r['gap_lomo_minus_within']:+.3f} p={r['gap_p']:.3f}")
