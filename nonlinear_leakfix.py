"""Leakage-corrected re-run of the capacity analysis (Table 3 of the pooling paper).

nonlinear_drift.py splits the personal arm with `rng.permutation` over trials POOLED ACROSS
EARLY SESSIONS:

    idx = rng.permutation(len(ye)); cut = int(0.8 * len(ye))
    WS = auc(kind, Xe[tr], ye[tr], Xe[ho], ye[ho])

so the decoder trains on session s and is tested on other trials of session s. Same bug as
pooling_drift.py analysis_A, but it matters MORE here: a 1-D CNN or GRU has far more capacity
to memorise session-specific structure (imaging conditions, hemodynamic state, slow arousal)
than a 44-dim logistic regression does. That makes "the asymmetry is LARGER with nonlinear
decoders" -- the paper's capacity-robustness claim -- exactly what leakage alone predicts.

This runs the ORIGINAL and CORRECTED protocols on identical cached features so the two can
be compared directly.

  fixed: for each held-out early session s,
           train on early \\ {s}; score on s (WS) and on the late block (WX)
         pooled decoder trained on other mice's early; scored on the same s (LS) and late (LX)

Also refactors to train ONCE and score multiple test sets -- the original calls auc() twice on
identical training data, training the same model twice.

Writes nonlinear_leakfix.json.
"""
import json, os
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import nonlinear_drift as nd

TARGETS = nd.TARGETS
EARLY_MAX, LATE_MIN = nd.EARLY_MAX, nd.LATE_MIN
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "nl_leakfix_features.npz")


def evoked_diff(X):
    """(n, L, P) window -> (n, P) evoked contrast, matching nonlinear_drift."""
    return nd.evoked_diff(X) if hasattr(nd, "evoked_diff") else \
        X[:, nd.A_FR:, :].mean(1) - X[:, :nd.A_FR, :].mean(1)


def fit_and_score(kind, Xtr, ytr, tests, seed=0, epochs=30):
    """Train ONE model on (Xtr,ytr); return AUC on each (Xte,yte) in tests."""
    from sklearn.metrics import roc_auc_score
    if kind == "linear":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
        clf.fit(evoked_diff(Xtr), ytr)
        return [float(roc_auc_score(yte, clf.predict_proba(evoked_diff(Xte))[:, 1]))
                for Xte, yte in tests]
    import torch, torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed); np.random.seed(seed)
    CNN, GRU = nd._torch_models()
    mu = Xtr.reshape(-1, Xtr.shape[-1]).mean(0); sd = Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-6
    xt = torch.tensor((Xtr - mu) / sd, dtype=torch.float32, device=dev)
    yt = torch.tensor(ytr, dtype=torch.float32, device=dev)
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
    outs = []
    with torch.no_grad():
        for Xte, yte in tests:
            xe = torch.tensor((Xte - mu) / sd, dtype=torch.float32, device=dev)
            p = torch.sigmoid(model(xe)).cpu().numpy()
            outs.append(float(roc_auc_score(yte, p)))
    return outs


def _sign_test(vals, seed=1, n=20000):
    v = np.array(vals, float); rng = np.random.default_rng(seed)
    null = np.array([(v * rng.choice([-1, 1], len(v))).mean() for _ in range(n)])
    return {"mean": float(v.mean()),
            "p_signflip": float((1 + np.sum(np.abs(null) >= abs(v.mean()))) / (1 + len(null))),
            "n": len(v), "frac_positive": float(np.mean(v > 0))}


def analyze(target, per_mouse):
    usable = {}
    for m, sess in per_mouse.items():
        e_days = sorted(d for d in sess if d <= EARLY_MAX)
        e = nd._pool(sess, e_days)
        l = nd._pool(sess, [d for d in sess if d >= LATE_MIN])
        if e is None or l is None or len(e_days) < 2:
            continue
        if nd._both(e[1]) and nd._both(l[1]):
            usable[m] = (e_days, e, l)
    mice = sorted(usable)
    out = {}
    for kind in ("linear", "cnn", "gru"):
        leaky, fixed = [], []
        for m in mice:
            e_days, (Xe, ye), (Xl, yl) = usable[m]
            sess = per_mouse[m]
            others = [o for o in mice if o != m]
            Xtr_o = np.concatenate([usable[o][1][0] for o in others])
            ytr_o = np.concatenate([usable[o][1][1] for o in others])
            if not nd._both(ytr_o):
                continue

            # ---- ORIGINAL (leaky): trial-level 80/20 over pooled early sessions ----
            rng = np.random.default_rng(hash(m) % 10000)
            idx = rng.permutation(len(ye)); cut = int(0.8 * len(ye))
            tr, ho = idx[:cut], idx[cut:]
            if nd._both(ye[tr]) and nd._both(ye[ho]):
                ws, wx = fit_and_score(kind, Xe[tr], ye[tr], [(Xe[ho], ye[ho]), (Xl, yl)])
                ls, lx = fit_and_score(kind, Xtr_o, ytr_o, [(Xe, ye), (Xl, yl)])
                leaky.append((ws - wx) - (ls - lx))

            # ---- CORRECTED: leave-one-early-SESSION-out ----
            ls_pool, lx_pool = fit_and_score(kind, Xtr_o, ytr_o, [(Xe, ye), (Xl, yl)])
            ws_s, wx_s, lsx_s = [], [], []
            for s in e_days:
                trp = nd._pool(sess, [d for d in e_days if d != s])
                if trp is None or not nd._both(trp[1]):
                    continue
                Xs, ys = sess[s]
                if not nd._both(ys):
                    continue
                a, b = fit_and_score(kind, trp[0], trp[1], [(Xs, ys), (Xl, yl)])
                ws_s.append(a); wx_s.append(b)
                # pooled decoder scored on the SAME held-out session
                lsx_s.append(fit_and_score(kind, Xtr_o, ytr_o, [(Xs, ys)])[0]
                             if kind == "linear" else None)
            if ws_s:
                WS, WX = float(np.mean(ws_s)), float(np.mean(wx_s))
                LSv = float(np.mean([v for v in lsx_s if v is not None])) if any(
                    v is not None for v in lsx_s) else ls_pool
                fixed.append((WS - WX) - (LSv - lx_pool))
            print(f"    [{target} {kind} {m}] done", flush=True)

        out[kind] = {"leaky": _sign_test(leaky) if leaky else None,
                     "fixed": _sign_test(fixed) if fixed else None}
        L, F = out[kind]["leaky"], out[kind]["fixed"]
        print(f"  [{target} {kind:6s}] LEAKY asym {L['mean']:+.4f} (p={L['p_signflip']:.4f})  ->  "
              f"FIXED asym {F['mean']:+.4f} (p={F['p_signflip']:.4f})", flush=True)
    return out


def gather():
    if os.path.exists(CACHE):
        print(f"loading cached features {CACHE}", flush=True)
        return np.load(CACHE, allow_pickle=True)["data"].item()
    task = nd.list_task_sessions()
    jobs = [(m, day, path) for m, sess in task.items() for (day, path) in sess]
    print(f"mice {len(task)}  task-sessions {len(jobs)}", flush=True)
    data = {t: defaultdict(dict) for t in TARGETS}
    ok = 0
    with ThreadPoolExecutor(max_workers=24) as ex:
        futs = {ex.submit(nd.build_session, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            mouse, day, per, msg = fut.result()
            if per:
                ok += 1
                for t, xy in per.items():
                    data[t][mouse][day] = xy
            if i % 25 == 0 or i == len(jobs):
                print(f"  streamed {i}/{len(jobs)} usable={ok}", flush=True)
    data = {t: dict(v) for t, v in data.items()}
    np.savez(CACHE, data=np.array(data, dtype=object))
    print(f"cached {CACHE}", flush=True)
    return data


def main():
    data = gather()
    out = {"targets": {}}
    for tgt in TARGETS:
        pm = {m: dict(v) for m, v in data[tgt].items() if len(v) >= 2}
        print(f"\n=== {tgt}: {len(pm)} mice ===", flush=True)
        out["targets"][tgt] = analyze(tgt, pm)
    dest = os.path.join(HERE, "nonlinear_leakfix.json")
    json.dump(out, open(dest, "w"), indent=1)
    print(f"\nsaved {dest}\nNL_LEAKFIX_DONE", flush=True)


if __name__ == "__main__":
    main()
