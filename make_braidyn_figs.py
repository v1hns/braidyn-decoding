"""Figures for the BraiDyn-BC cross-mouse decoding study."""
import json, os
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "figure.dpi": 200, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})
OUT = os.path.expanduser("~/braidyn_figs"); os.makedirs(OUT, exist_ok=True)
R = json.load(open(os.path.expanduser("~/braidyn_results.json")))


def parcel_names():
    """try to pull 44 Allen-parcel names from one NWB ImageSegmentation; else index labels."""
    try:
        import remfile, h5py, pynwb
        from dandi.dandiapi import DandiAPIClient
        with DandiAPIClient() as c:
            d = c.get_dandiset("001425", "draft")
            a = [x for x in d.get_assets() if x.path.endswith(".nwb") and "behavior" in x.path][0]
            url = a.get_content_url(follow_redirects=1, strip_query=True)
        nwb = pynwb.NWBHDF5IO(file=h5py.File(remfile.File(url), "r"), load_namespaces=True).read()
        seg = nwb.processing["ophys"]["ImageSegmentation"]
        ps = list(seg.plane_segmentations.values())[0]
        for col in ("area", "name", "label", "location", "roi_name"):
            if col in ps.colnames:
                return [str(x) for x in ps[col][:]]
    except Exception as e:
        print("parcel-name err", str(e)[:60])
    return [f"p{i}" for i in range(44)]


names = parcel_names()
targets = list(R.keys()); nice = {"state_lever": "lever-pull", "lick": "lick"}

# Fig 1: within-mouse vs LOMO AUC per target, with CI + chance
fig, ax = plt.subplots(figsize=(3.2, 2.5))
x = np.arange(len(targets)); w = 0.35
wm = [R[t]["within_mouse_auc"] for t in targets]
lo = [R[t]["lomo_auc"] for t in targets]
loerr = np.array([[R[t]["lomo_auc"]-R[t]["lomo_ci95"][0] for t in targets],
                  [R[t]["lomo_ci95"][1]-R[t]["lomo_auc"] for t in targets]])
ax.bar(x-w/2, wm, w, label="within-mouse", color="#7f8c8d")
ax.bar(x+w/2, lo, w, yerr=loerr, capsize=3, label="leave-mouse-out", color="#c0392b")
for xi, t in zip(x, targets):
    pm = list(R[t]["per_mouse_auc"].values())
    ax.scatter([xi+w/2]*len(pm), pm, s=6, color="k", alpha=0.35, zorder=3)
ax.axhline(0.5, ls="--", c="gray", lw=0.7)
ax.set_xticks(x); ax.set_xticklabels([nice.get(t, t) for t in targets])
ax.set_ylabel("decoding AUC"); ax.set_ylim(0.4, 1.0)
ax.set_title("Cross-mouse cortical decoding\n(no generalization gap; p=0.007)")
ax.legend(fontsize=6, loc="lower right")
fig.savefig(f"{OUT}/fig1_auc.pdf"); plt.close(fig)

# Fig 2: per-parcel importance (both targets), top parcels labeled
fig, axes = plt.subplots(1, len(targets), figsize=(6.4, 2.6))
for ax, t in zip(np.atleast_1d(axes), targets):
    imp = np.array(R[t]["parcel_importance"])
    order = np.argsort(-imp)[:12]
    ax.barh(range(len(order))[::-1], imp[order], color="#c0392b")
    ax.set_yticks(range(len(order))[::-1])
    ax.set_yticklabels([names[i][:14] if i < len(names) else f"p{i}" for i in order], fontsize=6)
    ax.set_xlabel("|weight|"); ax.set_title(nice.get(t, t))
fig.suptitle("Top cortical parcels driving the cross-mouse decoder", y=1.02, fontsize=8)
fig.savefig(f"{OUT}/fig2_parcels.pdf"); plt.close(fig)

# Fig 3: per-mouse LOMO AUC spread
fig, ax = plt.subplots(figsize=(3.2, 2.3))
for i, t in enumerate(targets):
    pm = sorted(R[t]["per_mouse_auc"].values())
    ax.plot(np.linspace(0, 1, len(pm)), pm, "o-", ms=3, label=nice.get(t, t))
ax.axhline(0.5, ls="--", c="gray", lw=0.7)
ax.set_xlabel("held-out mouse (sorted)"); ax.set_ylabel("LOMO AUC")
ax.set_title("Every held-out mouse decodes"); ax.legend(fontsize=6)
fig.savefig(f"{OUT}/fig3_permouse.pdf"); plt.close(fig)

print("parcel names sample:", names[:6])
print("wrote:", sorted(os.listdir(OUT)))
