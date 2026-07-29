"""Figure for the 15-day decay analysis: per-day personal vs pooled accuracy + linear fits.

Regenerates paper_neurips/figs/fig_decay.pdf from full_decay.json, which full_decay.py writes.
Kept as a committed script so the figure in the paper is reproducible from the released code.
"""
import json, os
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "figure.dpi": 200, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "paper_neurips", "figs"); os.makedirs(OUT, exist_ok=True)
T = json.load(open(os.path.join(HERE, "full_decay.json")))["targets"]
nice = {"state_lever": "lever-pull", "lick": "lick", "reward": "reward", "tone": "tone"}

fig, axes = plt.subplots(1, len(T), figsize=(7.2, 2.2), sharey=True)
for ax, (t, r) in zip(np.atleast_1d(axes), T.items()):
    c = r["decay_curve"]
    days = np.array(sorted(int(k) for k in c))
    pers = np.array([c[str(d)]["personal"] for d in days])
    pool = np.array([c[str(d)]["pooled"] for d in days])
    ax.plot(days, pers, "o", ms=3, color="#c0392b", label="personal")
    ax.plot(days, pool, "s", ms=3, color="#5d6d7e", label="pooled")
    for y, col in ((pers, "#c0392b"), (pool, "#5d6d7e")):
        b, a = np.polyfit(days, y, 1)
        ax.plot(days, a + b * days, "-", lw=1.1, color=col)
    ax.set_title(nice[t], fontsize=8)
    ax.set_xlabel("session day")
    ax.set_xticks([1, 5, 10, 15])
    # gap slope annotation, bottom-left; legend goes top-left so the two never collide
    ax.text(0.03, 0.04, f"gap {r['slope_gap_per_day']:+.4f}/d\np={r['gap_slope_p']:.3f}",
            transform=ax.transAxes, fontsize=6, color="#333", va="bottom")

np.atleast_1d(axes)[0].set_ylabel("decoding AUC")
np.atleast_1d(axes)[0].set_ylim(0.60, 1.0)
np.atleast_1d(axes)[0].legend(fontsize=6, frameon=False, loc="upper left",
                              handletextpad=0.4, borderaxespad=0.2)
fig.suptitle("Decoding across the 15-day protocol: the personal decoder starts ahead, "
             "the pooled decoder gains faster, so the gap narrows", fontsize=8, y=1.04)
fig.savefig(os.path.join(OUT, "fig_decay.pdf")); plt.close(fig)
print("wrote fig_decay.pdf")
for t, r in T.items():
    print(f"{t:11s} personal {r['slope_personal_per_day']:+.5f}/d  pooled {r['slope_pooled_per_day']:+.5f}/d  "
          f"gap {r['slope_gap_per_day']:+.5f}/d p={r['gap_slope_p']:.4f}")
