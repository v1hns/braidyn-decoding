"""Ground-truth simulation: why trial-level splits bias within- vs cross-subject comparisons.

The claim, stated generally. Suppose you compare a decoder trained on a subject's own data against
one trained on other subjects, and you estimate the within-subject arm with a trial-level split.
Trials recorded in the same session share a nuisance component (imaging conditions, impedance,
arousal, drift). A within-subject decoder trained on some trials of session k and tested on others
from session k can absorb that session's nuisance; a cross-subject decoder never shares a session
with its test data and cannot. The inflation is therefore ONE-SIDED, and it lands directly on the
within-minus-cross difference that such studies report.

Generative model, D dims, N subjects, K sessions each, T trials per session:

    x = delta * y * w  +  u_subject  +  v_{subject,session}  +  eps

w is the shared code, u a subject offset, v a per-session nuisance (sigma_sess), eps ~ N(0, 1).
y is balanced Bernoulli, so v carries NO label information: any gain from it is pure leakage.

Three estimators of the within-subject arm:
  trial   -- 5-fold over trials pooled across the subject's sessions   (the biased one)
  session -- leave-one-session-out                                     (the correct one)
and one cross-subject arm (train on other subjects), which is immune by construction.

Sweeps sigma_sess, model capacity, and trials-per-session, and reports the inflation
(trial - session) and the resulting bias in the reported within-minus-cross gap.

Writes leakage_sim.json.
"""
import json, os, itertools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = 44                    # matches the 44-parcel setting that motivated this
N_SUB, K_SESS = 12, 5
DELTA = 0.55              # signal strength -> AUC in a realistic 0.75-0.90 band


def make_cohort(rng, sigma_sess, T, sigma_sub=0.5, gamma=0.0):
    """gamma controls the SESSION-SPECIFIC LABEL-CORRELATED nuisance.

    sigma_sess alone adds a constant offset per session. That is label-independent, so it does not
    move a linear boundary and -- as sweep 1 shows -- produces no leakage. The mechanism that does is
    a direction z_{s,k} that separates the classes WITHIN one session only: real recordings drift,
    and if that drift happens to co-vary with when the animal performed, the session contains a
    spurious discriminative direction that is valid inside it and worthless outside. A decoder
    trained on part of a session inherits it and is rewarded on the rest of the same session.
    """
    w = rng.normal(size=D); w /= np.linalg.norm(w)
    subs = []
    for s in range(N_SUB):
        u = rng.normal(scale=sigma_sub, size=D)
        sess = []
        for k in range(K_SESS):
            v = rng.normal(scale=sigma_sess, size=D)
            z = rng.normal(size=D); z /= np.linalg.norm(z)   # this session's spurious direction
            y = rng.integers(0, 2, size=T).astype(float)
            x = (DELTA * y[:, None] * w
                 + gamma * y[:, None] * z            # label-correlated, session-specific
                 + u + v + rng.normal(size=(T, D)))
            sess.append((x, y))
        subs.append(sess)
    return subs


def model(kind):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "lin-C0.01":
        clf = LogisticRegression(max_iter=2000, C=0.01)
    elif kind == "lin-C0.5":
        clf = LogisticRegression(max_iter=2000, C=0.5)
    elif kind == "lin-C100":
        clf = LogisticRegression(max_iter=2000, C=100.0)
    elif kind == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=600, random_state=0)
    else:
        raise ValueError(kind)
    return make_pipeline(StandardScaler(), clf)


def auc(clf, X, y):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, clf.predict_proba(X)[:, 1]))


def run_one(subs, kind, seed=0):
    """Return mean over subjects of (within_trial, within_session, cross)."""
    from sklearn.model_selection import StratifiedKFold
    rng = np.random.default_rng(seed)
    wt, ws, cr = [], [], []
    for s, sess in enumerate(subs):
        Xs = np.vstack([x for x, _ in sess]); ys = np.concatenate([y for _, y in sess])

        # (a) within-subject, TRIAL-level 5-fold -- sessions are mixed across folds
        fold = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xs, ys):
            m = model(kind).fit(Xs[tr], ys[tr]); fold.append(auc(m, Xs[te], ys[te]))
        wt.append(np.mean(fold))

        # (b) within-subject, leave-one-SESSION-out
        fold = []
        for k in range(len(sess)):
            tr_x = np.vstack([x for j, (x, _) in enumerate(sess) if j != k])
            tr_y = np.concatenate([y for j, (_, y) in enumerate(sess) if j != k])
            m = model(kind).fit(tr_x, tr_y); fold.append(auc(m, sess[k][0], sess[k][1]))
        ws.append(np.mean(fold))

        # (c) cross-subject: train on every other subject, test on this one
        ox = np.vstack([x for j, o in enumerate(subs) if j != s for x, _ in o])
        oy = np.concatenate([y for j, o in enumerate(subs) if j != s for _, y in o])
        m = model(kind).fit(ox, oy); cr.append(auc(m, Xs, ys))
    return float(np.mean(wt)), float(np.mean(ws)), float(np.mean(cr))


def main():
    out = {"config": {"D": D, "n_subjects": N_SUB, "n_sessions": K_SESS, "delta": DELTA}, "runs": []}

    print("=== sweep 1: session-nuisance magnitude (T=100, lin-C0.5) ===", flush=True)
    print(f"{'sigma_sess':>10s} {'within_trial':>13s} {'within_sess':>12s} {'cross':>8s} "
          f"{'inflation':>10s} {'gap_trial':>10s} {'gap_sess':>9s} {'bias':>8s}")
    for sig in [0.0, 0.25, 0.5, 1.0, 2.0]:
        rng = np.random.default_rng(0)
        subs = make_cohort(rng, sig, T=100)
        a, b, c = run_one(subs, "lin-C0.5")
        r = {"sweep": "sigma", "sigma_sess": sig, "T": 100, "model": "lin-C0.5",
             "within_trial": a, "within_session": b, "cross": c,
             "inflation": a - b, "gap_trial": a - c, "gap_session": b - c, "bias": (a - c) - (b - c)}
        out["runs"].append(r)
        print(f"{sig:10.2f} {a:13.4f} {b:12.4f} {c:8.4f} {a-b:+10.4f} {a-c:+10.4f} {b-c:+9.4f} {(a-b):+8.4f}", flush=True)

    print("\n=== sweep 2: model capacity (sigma_sess=1.0, T=100) ===", flush=True)
    print(f"{'model':>12s} {'within_trial':>13s} {'within_sess':>12s} {'cross':>8s} {'inflation':>10s}")
    for kind in ["lin-C0.01", "lin-C0.5", "lin-C100", "mlp"]:
        rng = np.random.default_rng(0)
        subs = make_cohort(rng, 1.0, T=100)
        a, b, c = run_one(subs, kind)
        r = {"sweep": "capacity", "sigma_sess": 1.0, "T": 100, "model": kind,
             "within_trial": a, "within_session": b, "cross": c,
             "inflation": a - b, "gap_trial": a - c, "gap_session": b - c}
        out["runs"].append(r)
        print(f"{kind:>12s} {a:13.4f} {b:12.4f} {c:8.4f} {a-b:+10.4f}", flush=True)

    print("\n=== sweep 3: trials per session (sigma_sess=1.0, lin-C0.5) ===", flush=True)
    print(f"{'T':>6s} {'within_trial':>13s} {'within_sess':>12s} {'cross':>8s} {'inflation':>10s}")
    for T in [25, 50, 100, 200]:
        rng = np.random.default_rng(0)
        subs = make_cohort(rng, 1.0, T=T)
        a, b, c = run_one(subs, "lin-C0.5")
        r = {"sweep": "T", "sigma_sess": 1.0, "T": T, "model": "lin-C0.5",
             "within_trial": a, "within_session": b, "cross": c,
             "inflation": a - b, "gap_trial": a - c, "gap_session": b - c}
        out["runs"].append(r)
        print(f"{T:6d} {a:13.4f} {b:12.4f} {c:8.4f} {a-b:+10.4f}", flush=True)

    print("\n=== sweep 4: SESSION-SPECIFIC LABEL-CORRELATED nuisance (T=100, lin-C0.5) ===", flush=True)
    print("    sweep 1 showed a label-INDEPENDENT session offset leaks nothing; this is the mechanism", flush=True)
    print(f"{'gamma':>7s} {'within_trial':>13s} {'within_sess':>12s} {'cross':>8s} "
          f"{'inflation':>10s} {'gap_trial':>10s} {'gap_sess':>9s}")
    for g in [0.0, 0.25, 0.5, 1.0, 2.0]:
        rng = np.random.default_rng(0)
        subs = make_cohort(rng, 0.5, T=100, gamma=g)
        a, b, c = run_one(subs, "lin-C0.5")
        out["runs"].append({"sweep": "gamma", "gamma": g, "sigma_sess": 0.5, "T": 100,
                            "model": "lin-C0.5", "within_trial": a, "within_session": b, "cross": c,
                            "inflation": a - b, "gap_trial": a - c, "gap_session": b - c})
        print(f"{g:7.2f} {a:13.4f} {b:12.4f} {c:8.4f} {a-b:+10.4f} {a-c:+10.4f} {b-c:+9.4f}", flush=True)

    print("\n=== sweep 5: capacity x label-correlated nuisance (gamma=1.0, T=100) ===", flush=True)
    print(f"{'model':>12s} {'within_trial':>13s} {'within_sess':>12s} {'cross':>8s} {'inflation':>10s}")
    for kind in ["lin-C0.01", "lin-C0.5", "lin-C100", "mlp"]:
        rng = np.random.default_rng(0)
        subs = make_cohort(rng, 0.5, T=100, gamma=1.0)
        a, b, c = run_one(subs, kind)
        out["runs"].append({"sweep": "gamma_capacity", "gamma": 1.0, "T": 100, "model": kind,
                            "within_trial": a, "within_session": b, "cross": c,
                            "inflation": a - b, "gap_trial": a - c, "gap_session": b - c})
        print(f"{kind:>12s} {a:13.4f} {b:12.4f} {c:8.4f} {a-b:+10.4f}", flush=True)

    dest = os.path.join(HERE, "leakage_sim.json")
    json.dump(out, open(dest, "w"), indent=1)
    print(f"\nsaved {dest}\nSIM_DONE", flush=True)


if __name__ == "__main__":
    main()
