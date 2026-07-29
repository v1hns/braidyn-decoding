"""Nonlinear-decoder drift test (DANDI:001425): does 'pooling resists drift' survive a CNN/GRU,
not just the linear evoked-mean decoder?

Every aging number in the paper comes from linear logistic regression. A reviewer will ask whether
the drift-resistance is an artifact of that weak decoder. Here we recompute the personal-vs-pooled
aging asymmetry with three decoders on the SAME early/late blocks and events:
  linear : logistic regression on the evoked-difference vector (paper's model)
  cnn    : 1-D CNN over the -0.5..+1.0 s window
  gru    : GRU over the window
Per held-out mouse M and target:
  personal aging = WS - WX  (train M early, test M early-holdout vs M late)
  pooled aging   = LS - LX  (train other mice's early, test M early vs M late)
  asymmetry      = personal_aging - pooled_aging  (>0 => own decoder ages more)
Paired sign-flip test over mice, per target and per model. Saves nonlinear_drift.json.
GOTCHA: install torch cu128 on the a10 (driver CUDA 12.8), not default wheels.
"""
import re, json, os, socket
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

socket.setdefaulttimeout(90)
TARGETS = ["state_lever", "lick", "reward", "tone"]
EARLY_MAX, LATE_MIN = 5, 11
A_FR, B_FR = 15, 30          # fixed pre/post frames -> constant window length L=45
HERE = os.path.dirname(os.path.abspath(__file__))


def list_task_sessions():
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
    return {m: sorted(v) for m, v in by.items()}


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
    for k in TARGETS:
        try:
            beh[k] = np.asarray(dn[k].data).ravel()
        except Exception:
            beh[k] = None
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
    T = dff.shape[0]
    return np.array([dff[e - a:e + b] for e in ev if e - a >= 0 and e + b < T], dtype=np.float32)


def build_session(args):
    mouse, day, path = args
    try:
        dff, beh, rate = stream(path)
    except Exception as e:
        return mouse, day, None, str(e)[:50]
    a, b = A_FR, B_FR; L = a + b
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
            X = np.concatenate([pos[:n], neg[:n]]).astype(np.float32)
            y = np.r_[np.ones(n), np.zeros(n)].astype(np.float32)
            per[tgt] = (X, y)
    return mouse, day, per, f"{list(per.keys())}"


# ---------- models (reused from braidyn_nonlinear.py) ----------
def evoked_diff(X, a=A_FR):
    return X[:, a:, :].mean(1) - X[:, :a, :].mean(1)


def auc_linear(Xtr, ytr, Xte, yte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
    clf.fit(evoked_diff(Xtr), ytr)
    return float(roc_auc_score(yte, clf.predict_proba(evoked_diff(Xte))[:, 1]))


def _torch_models():
    import torch.nn as nn

    class CNN(nn.Module):
        def __init__(self, C=44):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(C, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
                nn.Conv1d(64, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1))
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(64, 1))

        def forward(self, x):
            h = self.net(x.transpose(1, 2)).squeeze(-1)
            return self.head(h).squeeze(-1)

    class GRU(nn.Module):
        def __init__(self, C=44, h=64):
            super().__init__()
            self.gru = nn.GRU(C, h, batch_first=True)
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(h, 1))

        def forward(self, x):
            o, _ = self.gru(x)
            return self.head(o[:, -1]).squeeze(-1)

    return CNN, GRU


def auc_net(kind, Xtr, ytr, Xte, yte, seed=0, epochs=30):
    import torch, torch.nn as nn
    from sklearn.metrics import roc_auc_score
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed); np.random.seed(seed)
    CNN, GRU = _torch_models()
    mu = Xtr.reshape(-1, Xtr.shape[-1]).mean(0); sd = Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-6
    xt = torch.tensor((Xtr - mu) / sd, dtype=torch.float32, device=dev)
    yt = torch.tensor(ytr, dtype=torch.float32, device=dev)
    xe = torch.tensor((Xte - mu) / sd, dtype=torch.float32, device=dev)
    model = (CNN() if kind == "cnn" else GRU()).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(); n = len(yt); bs = 128
    g = torch.Generator(device="cpu").manual_seed(seed)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g).to(dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if idx.numel() < 2:
                continue
            opt.zero_grad(); loss_fn(model(xt[idx]), yt[idx]).backward(); opt.step()
    model.eval()
    import torch as _t
    with _t.no_grad():
        p = _t.sigmoid(model(xe)).cpu().numpy()
    return float(roc_auc_score(yte, p))


def _both(y):
    return len(set(y.tolist())) == 2


def _pool(sess, days):
    xs = [sess[d][0] for d in days if d in sess]; ys = [sess[d][1] for d in days if d in sess]
    if not xs:
        return None
    return np.concatenate(xs), np.concatenate(ys)


def auc(kind, Xtr, ytr, Xte, yte):
    if kind == "linear":
        return auc_linear(Xtr, ytr, Xte, yte)
    return auc_net(kind, Xtr, ytr, Xte, yte)


def analyze(target, per_mouse):
    early = {}; late = {}
    for m in per_mouse:
        e = _pool(per_mouse[m], [d for d in per_mouse[m] if d <= EARLY_MAX])
        l = _pool(per_mouse[m], [d for d in per_mouse[m] if d >= LATE_MIN])
        if e is not None and l is not None and _both(e[1]) and _both(l[1]):
            early[m] = e; late[m] = l
    mice = sorted(early)
    out = {}
    for kind in ("linear", "cnn", "gru"):
        rows = []
        for m in mice:
            Xe, ye = early[m]; Xl, yl = late[m]
            rng = np.random.default_rng(hash(m) % 10000)
            idx = rng.permutation(len(ye)); cut = int(0.8 * len(ye))
            tr, ho = idx[:cut], idx[cut:]
            if not (_both(ye[tr]) and _both(ye[ho])):
                continue
            others = [o for o in mice if o != m]
            Xtr = np.concatenate([early[o][0] for o in others]); ytr = np.concatenate([early[o][1] for o in others])
            if not _both(ytr):
                continue
            # personal: train M-early(80%), test M-early(20%)=WS and M-late=WX
            WS = auc(kind, Xe[tr], ye[tr], Xe[ho], ye[ho])
            WX = auc(kind, Xe[tr], ye[tr], Xl, yl)
            # pooled: train others-early, test M-early=LS and M-late=LX
            LS = auc(kind, Xtr, ytr, Xe, ye)
            LX = auc(kind, Xtr, ytr, Xl, yl)
            rows.append((m, WS - WX, LS - LX))
        arr = np.array([(r[1], r[2]) for r in rows])
        asym = arr[:, 0] - arr[:, 1]
        rng = np.random.default_rng(1)
        null = np.array([(asym * rng.choice([-1, 1], len(asym))).mean() for _ in range(20000)])
        p = float((1 + np.sum(np.abs(null) >= abs(asym.mean()))) / (1 + len(null)))
        out[kind] = {"n": len(rows), "personal_aging": float(arr[:, 0].mean()),
                     "pooled_aging": float(arr[:, 1].mean()), "asymmetry": float(asym.mean()),
                     "p_signflip": p, "frac_personal_ages_more": float(np.mean(asym > 0))}
        print(f"  [{target} {kind:6s}] asym {out[kind]['asymmetry']:+.4f} p={p:.4f} "
              f"(personal {out[kind]['personal_aging']:+.3f} pooled {out[kind]['pooled_aging']:+.3f}, "
              f"n={len(rows)})", flush=True)
    return out


def main():
    task = list_task_sessions()
    jobs = [(m, day, path) for m, sess in task.items() for (day, path) in sess]
    print(f"mice {len(task)}  task-sessions {len(jobs)}", flush=True)
    data = {t: defaultdict(dict) for t in TARGETS}
    ok = 0
    with ThreadPoolExecutor(max_workers=24) as ex:
        futs = {ex.submit(build_session, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            mouse, day, per, msg = fut.result()
            if per:
                ok += 1
                for t, xy in per.items():
                    data[t][mouse][day] = xy
            if i % 25 == 0 or i == len(jobs):
                print(f"  streamed {i}/{len(jobs)} usable={ok}", flush=True)
    out = {"targets": {}}
    for tgt in TARGETS:
        pm = {m: dict(v) for m, v in data[tgt].items() if len(v) >= 2}
        print(f"\n=== {tgt}: {len(pm)} mice ===", flush=True)
        out["targets"][tgt] = analyze(tgt, pm)
    dest = os.path.join(HERE, "nonlinear_drift.json")
    json.dump(out, open(dest, "w"), indent=1)
    print(f"\nsaved {dest}\nNL_DRIFT_DONE", flush=True)


if __name__ == "__main__":
    main()
