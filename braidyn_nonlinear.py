"""BraiDyn-BC nonlinear temporal decoding (DANDI:001425).

The conservation result (LOMO AUC >= within-mouse) was obtained with a LINEAR decoder on a
single evoked-difference vector (post-onset minus pre-onset mean). Two open questions:
  (1) Does exploiting the full nonlinear TEMPORAL dynamics raise absolute decoding accuracy?
  (2) Does the no-generalization-gap conservation SURVIVE a higher-capacity model, or was it an
      artifact of the linear model's low capacity? A nonlinear model has more room to fit
      mouse-specific idiosyncrasies and could OPEN a cross-animal gap.

We extract the full spatiotemporal window around each event (-0.5..+1.0 s = 45 frames x 44
Allen parcels) and compare three decoders on IDENTICAL data / events / splits:
  linear : logistic regression on the evoked-difference vector (reproduces the paper's model)
  cnn    : 1-D convolution over time (channels = 44 parcels) -> global pool -> FC
  gru    : GRU over the 45-step sequence (features = 44 parcels) -> FC
Evaluation per target (lever-pull `state_lever`, `lick`): within-mouse 5-fold CV and
leave-one-mouse-out (LOMO). We use 3 task sessions/mouse spread across the 15-day protocol for
adequate training data. Windows are cached to NPZ so model iteration does not re-stream.
Saves braidyn_nonlinear.json.
"""
import re, json, os, socket
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

socket.setdefaulttimeout(90)
TARGETS = ["state_lever", "lick"]
PRE, POST = 0.5, 1.0          # window seconds
A_FR, B_FR = 15, 30           # FIXED pre/post frame counts (dataset is ~30 Hz) -> constant L=45
CACHE = "/home/ubuntu/nl_windows.npz"


# ---------- data access ----------
def pick_sessions(k=3):
    """k task sessions per mouse, spread across the protocol (early/mid/late)."""
    from dandi.dandiapi import DandiAPIClient
    with DandiAPIClient() as c:
        d = c.get_dandiset("001425", "draft")
        beh = [a.path for a in d.get_assets()
               if a.path.endswith(".nwb") and "behavior" in a.path and "task" in a.path]
    by = defaultdict(list)
    for p in beh:
        mouse = re.search(r"(sub-[^/]+?)/", p).group(1)
        day = int(re.search(r"day(\d+)", p).group(1))
        by[mouse].append((day, p))
    out = {}
    for mouse, v in by.items():
        v = sorted(v)
        n = len(v)
        idx = sorted(set([n // 4, n // 2, (3 * n) // 4]))[:k] if n >= k else list(range(n))
        out[mouse] = [v[i] for i in idx]
    return out


def stream(path):
    import remfile, h5py, pynwb
    from dandi.dandiapi import DandiAPIClient
    with DandiAPIClient() as c:
        d = c.get_dandiset("001425", "draft")
        url = d.get_asset_by_path(path).get_content_url(follow_redirects=1, strip_query=True)
    f = h5py.File(remfile.File(url), "r")
    nwb = pynwb.NWBHDF5IO(file=f, load_namespaces=True).read()
    dff = np.asarray(nwb.processing["ophys"]["DfOverF"]["dFF"].data)
    dn = nwb.processing["downsampled"]
    beh = {}
    for kk in TARGETS:
        try:
            beh[kk] = np.asarray(dn[kk].data).ravel()
        except Exception:
            beh[kk] = None
    ts = nwb.processing["ophys"]["DfOverF"]["dFF"].timestamps
    rate = 1.0 / np.median(np.diff(np.asarray(ts[:200]))) if ts is not None else 30.0
    return dff, beh, rate


def onsets(sig, rate):
    if sig is None:
        return np.array([], int)
    x = sig.astype(float)
    if np.nanmax(x) <= np.nanmin(x):
        return np.array([], int)
    b = (x > (np.nanmax(x) + np.nanmin(x)) / 2).astype(int)
    on = np.where(np.diff(b) == 1)[0] + 1
    keep = []
    for o in on:
        if not keep or o - keep[-1] > rate:
            keep.append(o)
    return np.array(keep, int)


def windows(dff, ev, a, b):
    """Spatiotemporal windows (n, a+b, 44) for each valid onset."""
    T = dff.shape[0]
    return np.array([dff[e - a:e + b] for e in ev if e - a >= 0 and e + b < T], dtype=np.float32)


def build_session(args):
    mouse, day, path = args
    try:
        dff, beh, rate = stream(path)
    except Exception as e:
        return mouse, day, None, str(e)[:50]
    a, b = A_FR, B_FR          # fixed frame counts -> constant window length across sessions
    L = a + b
    rng = np.random.default_rng(day)
    T = dff.shape[0]; per = {}
    for tgt in TARGETS:
        ev = onsets(beh.get(tgt), rate)
        if len(ev) < 8:
            continue
        pos = windows(dff, ev, a, b)
        if len(pos) == 0:
            continue
        far = [t for t in rng.integers(a + 1, T - b - 1, size=len(ev) * 3)
               if np.min(np.abs(ev - t)) > 2 * rate][:len(pos)]
        neg = windows(dff, np.array(far), a, b)
        n = min(len(pos), len(neg))
        if n >= 6 and pos.shape[1] == L and neg.shape[1] == L:
            X = np.concatenate([pos[:n], neg[:n]]).astype(np.float32)   # (2n, L, 44)
            y = np.r_[np.ones(n), np.zeros(n)].astype(np.float32)
            per[tgt] = (X, y)
    return mouse, day, per, f"rate {rate:.0f} {list(per.keys())}"


def gather():
    """Stream all sessions, concatenate windows per (target, mouse). Cache to NPZ."""
    if os.path.exists(CACHE):
        print("loading cached windows", flush=True)
        z = np.load(CACHE, allow_pickle=True)
        return z["data"].item(), int(z["L"])
    sess = pick_sessions()
    jobs = [(m, day, path) for m, v in sess.items() for (day, path) in v]
    n = len(jobs)
    print(f"mice {len(sess)}  sessions {n}", flush=True)
    acc = {t: defaultdict(list) for t in TARGETS}
    ok = 0; L = None
    with ThreadPoolExecutor(max_workers=24) as ex:
        futs = {ex.submit(build_session, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            mouse, day, per, msg = fut.result()
            if per:
                ok += 1
                for tgt, (X, y) in per.items():
                    acc[tgt][mouse].append((X, y)); L = X.shape[1]
            if i % 15 == 0 or i == n:
                print(f"  streamed {i}/{n} (usable={ok})", flush=True)
    data = {t: {} for t in TARGETS}
    for t in TARGETS:
        for m, lst in acc[t].items():
            data[t][m] = (np.concatenate([x for x, _ in lst]),
                          np.concatenate([y for _, y in lst]))
    np.savez(CACHE, data=np.array(data, dtype=object), L=L)
    print(f"cached windows to {CACHE}; L={L}", flush=True)
    return data, L


# ---------- models ----------
def evoked_diff(X, a):
    """(n, L, 44) -> (n, 44) post-mean minus pre-mean = the linear model's feature."""
    return X[:, a:, :].mean(1) - X[:, :a, :].mean(1)


def auc_linear(Xtr, ytr, Xte, yte, a):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    ftr, fte = evoked_diff(Xtr, a), evoked_diff(Xte, a)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
    clf.fit(ftr, ytr)
    return roc_auc_score(yte, clf.predict_proba(fte)[:, 1])


def _torch_models():
    import torch, torch.nn as nn

    class CNN(nn.Module):
        def __init__(self, C=44):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(C, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
                nn.Conv1d(64, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1))
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(64, 1))

        def forward(self, x):                 # x: (B, L, C)
            h = self.net(x.transpose(1, 2)).squeeze(-1)
            return self.head(h).squeeze(-1)

    class GRU(nn.Module):
        def __init__(self, C=44, h=64):
            super().__init__()
            self.gru = nn.GRU(C, h, batch_first=True)
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(h, 1))

        def forward(self, x):                 # x: (B, L, C)
            o, _ = self.gru(x)
            return self.head(o[:, -1]).squeeze(-1)

    return CNN, GRU


def auc_net(kind, Xtr, ytr, Xte, yte, seed=0, epochs=35):
    import torch, torch.nn as nn
    from sklearn.metrics import roc_auc_score
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed); np.random.seed(seed)
    CNN, GRU = _torch_models()
    # per-channel standardization from TRAIN only (mean/std over samples x time)
    mu = Xtr.reshape(-1, Xtr.shape[-1]).mean(0)
    sd = Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-6
    xt = torch.tensor((Xtr - mu) / sd, dtype=torch.float32, device=dev)
    yt = torch.tensor(ytr, dtype=torch.float32, device=dev)
    xe = torch.tensor((Xte - mu) / sd, dtype=torch.float32, device=dev)
    model = (CNN() if kind == "cnn" else GRU()).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    n = len(yt); bs = 128
    g = torch.Generator(device="cpu").manual_seed(seed)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g).to(dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if idx.numel() < 2:          # BatchNorm1d needs >1 sample per batch
                continue
            opt.zero_grad()
            loss_fn(model(xt[idx]), yt[idx]).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(xe)).cpu().numpy()
    return roc_auc_score(yte, p)


# ---------- evaluation ----------
def evaluate(target, data, L, a):
    from sklearn.model_selection import StratifiedKFold
    mice = sorted(data)
    res = {k: {"within": {}, "lomo": {}} for k in ("linear", "cnn", "gru")}

    # within-mouse 5-fold
    for m in mice:
        X, y = data[m]
        if len(y) < 40 or len(set(y)) < 2:
            continue
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        acc = {k: [] for k in res}
        for tr, te in skf.split(X, y):
            acc["linear"].append(auc_linear(X[tr], y[tr], X[te], y[te], a))
            acc["cnn"].append(auc_net("cnn", X[tr], y[tr], X[te], y[te]))
            acc["gru"].append(auc_net("gru", X[tr], y[tr], X[te], y[te]))
        for k in res:
            res[k]["within"][m] = float(np.mean(acc[k]))
        print(f"  [{target}] within {m}: "
              f"lin {res['linear']['within'][m]:.3f} cnn {res['cnn']['within'][m]:.3f} "
              f"gru {res['gru']['within'][m]:.3f}", flush=True)

    # leave-one-mouse-out
    for m in mice:
        Xte, yte = data[m]
        if len(set(yte)) < 2:
            continue
        others = [o for o in mice if o != m]
        Xtr = np.concatenate([data[o][0] for o in others])
        ytr = np.concatenate([data[o][1] for o in others])
        res["linear"]["lomo"][m] = float(auc_linear(Xtr, ytr, Xte, yte, a))
        res["cnn"]["lomo"][m] = float(auc_net("cnn", Xtr, ytr, Xte, yte))
        res["gru"]["lomo"][m] = float(auc_net("gru", Xtr, ytr, Xte, yte))
        print(f"  [{target}] LOMO {m}: lin {res['linear']['lomo'][m]:.3f} "
              f"cnn {res['cnn']['lomo'][m]:.3f} gru {res['gru']['lomo'][m]:.3f}", flush=True)

    out = {"target": target, "n_mice": len(mice)}
    for k in res:
        w = np.array(list(res[k]["within"].values()))
        l = np.array(list(res[k]["lomo"].values()))
        # paired gap over mice present in both
        common = [mm for mm in res[k]["within"] if mm in res[k]["lomo"]]
        gap = np.array([res[k]["lomo"][mm] - res[k]["within"][mm] for mm in common])
        rng = np.random.default_rng(1)
        null = np.array([np.mean(gap * rng.choice([-1, 1], len(gap))) for _ in range(5000)]) if len(gap) else np.array([0.0])
        p_gap = float((1 + np.sum(np.abs(null) >= abs(gap.mean()))) / (1 + len(null))) if len(gap) else None
        out[k] = {"within_auc": float(w.mean()) if len(w) else None,
                  "lomo_auc": float(l.mean()) if len(l) else None,
                  "gap_lomo_minus_within": float(gap.mean()) if len(gap) else None,
                  "gap_p": p_gap, "n_within": int(len(w)), "n_lomo": int(len(l)),
                  "within_per_mouse": {mm: round(v, 3) for mm, v in res[k]["within"].items()},
                  "lomo_per_mouse": {mm: round(v, 3) for mm, v in res[k]["lomo"].items()}}
    return out


def main():
    data, L = gather()
    a = A_FR                                  # pre-onset frames within the window
    print(f"window L={L}, pre-frames a={a}", flush=True)
    out = {"window_frames": L, "pre_frames": a, "targets": {}}
    for tgt in TARGETS:
        d = {m: xy for m, xy in data[tgt].items() if len(xy[1]) >= 20}
        print(f"\n=== {tgt}: {len(d)} mice ===", flush=True)
        r = evaluate(tgt, d, L, a)
        out["targets"][tgt] = r
        for k in ("linear", "cnn", "gru"):
            rk = r[k]
            print(f"  {k:6s} within {rk['within_auc']:.3f}  LOMO {rk['lomo_auc']:.3f}  "
                  f"gap {rk['gap_lomo_minus_within']:+.3f} (p={rk['gap_p']:.3f})", flush=True)
    json.dump(out, open("/home/ubuntu/braidyn_nonlinear.json", "w"), indent=1)
    print("\nsaved braidyn_nonlinear.json\nBRAIDYN_NL_DONE", flush=True)


if __name__ == "__main__":
    main()
