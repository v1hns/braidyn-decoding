"""Do the four decoded "events" actually index four separable moments?

This is a cued lever-pull operant task, so tone -> lever -> reward -> lick are chained within a
trial by design. That raises two questions the paper currently assumes away:

  1. POSITIVE-window contamination. A feature vector spans [-0.5, +1.0] s around an onset. If a
     reward onset almost always has a lick onset inside that window, the reward decoder and the
     lick decoder are reading substantially the same physiological moment, and "two independent
     events show the effect" is much weaker than it sounds.

  2. NEGATIVE-window contamination. build_session() draws negatives >2 s from onsets of the SAME
     target only (`ev` is per-target), so a lick decoder's negative may sit on a lever pull. The
     paper's Methods says "more than 2 s from any event", which is stronger than what the code does.

Reports, per session and pooled: onset counts, median nearest-neighbour latency for every ordered
pair of event types, the fraction of each target's positive windows containing another target's
onset, and the fraction of each target's negatives containing any other target's onset.

Streams behavioural channels only -- no dF/F, no model fitting. Light enough to run locally.
"""
import re, json, os, socket, sys
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

socket.setdefaulttimeout(90)
TARGETS = ["state_lever", "lick", "reward", "tone"]
PRE, POST = 0.5, 1.0          # feature window, matching feats()
NEG_GUARD = 2.0               # build_session's refractory for negatives
HERE = os.path.dirname(os.path.abspath(__file__))
N_SESSIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 8


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


def stream_beh(path):
    import remfile, h5py, pynwb
    from dandi.dandiapi import DandiAPIClient
    with DandiAPIClient() as c:
        d = c.get_dandiset("001425", "draft")
        url = d.get_asset_by_path(path).get_content_url(follow_redirects=1, strip_query=True)
    f = h5py.File(remfile.File(url), "r")
    nwb = pynwb.NWBHDF5IO(file=f, load_namespaces=True).read()
    dn = nwb.processing["downsampled"]
    beh = {}
    for k in TARGETS:
        try:
            beh[k] = np.asarray(dn[k].data).ravel()
        except Exception:
            beh[k] = None
    ts = nwb.processing["ophys"]["DfOverF"]["dFF"].timestamps
    rate = 1.0 / np.median(np.diff(np.asarray(ts[:200]))) if ts is not None else 30.0
    n = len(nwb.processing["ophys"]["DfOverF"]["dFF"].data)
    return beh, rate, n


def onsets(sig, rate):
    """Identical to build_session's detector."""
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


def analyse(args):
    mouse, day, path = args
    try:
        beh, rate, T = stream_beh(path)
    except Exception as e:
        return None, f"{mouse} d{day}: {str(e)[:50]}"
    ev = {t: onsets(beh.get(t), rate) for t in TARGETS}
    if sum(len(v) >= 8 for v in ev.values()) < 2:
        return None, f"{mouse} d{day}: too few events"
    a, b = int(PRE * rate), int(POST * rate)
    out = {"mouse": mouse, "day": day, "rate": float(rate),
           "counts": {t: int(len(ev[t])) for t in TARGETS},
           "latency_s": {}, "pos_contam": {}, "neg_contam": {}}

    # nearest-neighbour latency: for each A onset, signed distance to closest B onset
    for A in TARGETS:
        for B in TARGETS:
            if A == B or len(ev[A]) == 0 or len(ev[B]) == 0:
                continue
            d = np.array([ev[B][np.argmin(np.abs(ev[B] - o))] - o for o in ev[A]]) / rate
            out["latency_s"][f"{A}->{B}"] = {
                "median": float(np.median(d)), "median_abs": float(np.median(np.abs(d))),
                "frac_within_window": float(np.mean((d >= -PRE) & (d <= POST)))}

    # positive-window contamination: another target's onset inside [-PRE, +POST] of an A onset
    for A in TARGETS:
        if len(ev[A]) == 0:
            continue
        others = np.sort(np.concatenate([ev[B] for B in TARGETS if B != A and len(ev[B])] or [np.array([])]))
        if len(others) == 0:
            continue
        hit = [np.any((others >= o - a) & (others <= o + b)) for o in ev[A]]
        out["pos_contam"][A] = float(np.mean(hit))

    # negative-window contamination: negatives are >NEG_GUARD from A only; do they hit another event?
    rng = np.random.default_rng(day)
    for A in TARGETS:
        if len(ev[A]) == 0:
            continue
        cand = rng.integers(int(NEG_GUARD * rate), T - int(NEG_GUARD * rate), size=len(ev[A]) * 3)
        far = np.array([t for t in cand if np.min(np.abs(ev[A] - t)) > NEG_GUARD * rate][:len(ev[A])])
        if len(far) == 0:
            continue
        others = np.sort(np.concatenate([ev[B] for B in TARGETS if B != A and len(ev[B])] or [np.array([])]))
        if len(others) == 0:
            continue
        hit = [np.any((others >= t - a) & (others <= t + b)) for t in far]
        out["neg_contam"][A] = float(np.mean(hit))
    return out, f"{mouse} d{day}: ok"


def main():
    task = list_task_sessions()
    # spread across mice: first task day of the first N_SESSIONS mice
    jobs = []
    for m in sorted(task):
        if task[m]:
            day, path = task[m][len(task[m]) // 2]      # a mid-protocol session
            jobs.append((m, day, path))
        if len(jobs) >= N_SESSIONS:
            break
    print(f"checking {len(jobs)} sessions across {len(jobs)} mice", flush=True)

    rows = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(analyse, j): j for j in jobs}
        for fut in as_completed(futs):
            r, msg = fut.result()
            print("  " + msg, flush=True)
            if r:
                rows.append(r)

    if not rows:
        print("no usable sessions"); return

    print(f"\n=== onset counts (median over {len(rows)} sessions) ===")
    for t in TARGETS:
        v = [r["counts"][t] for r in rows]
        print(f"  {t:12s} median {np.median(v):6.0f}   range {min(v)}-{max(v)}")

    print("\n=== nearest-neighbour latency, A -> nearest B (median over sessions) ===")
    print(f"  {'pair':28s} {'med signed':>11s} {'med |lat|':>10s} {'frac inside [-0.5,+1.0]s':>26s}")
    keys = sorted({k for r in rows for k in r["latency_s"]})
    for k in keys:
        med = np.median([r["latency_s"][k]["median"] for r in rows if k in r["latency_s"]])
        mab = np.median([r["latency_s"][k]["median_abs"] for r in rows if k in r["latency_s"]])
        fw = np.median([r["latency_s"][k]["frac_within_window"] for r in rows if k in r["latency_s"]])
        print(f"  {k:28s} {med:+11.3f} {mab:10.3f} {fw:26.1%}")

    print("\n=== window contamination (median over sessions) ===")
    print(f"  {'target':12s} {'POS has another event':>22s} {'NEG has another event':>22s}")
    for t in TARGETS:
        p = [r["pos_contam"][t] for r in rows if t in r["pos_contam"]]
        n = [r["neg_contam"][t] for r in rows if t in r["neg_contam"]]
        ps = f"{np.median(p):.1%}" if p else "n/a"
        ns = f"{np.median(n):.1%}" if n else "n/a"
        print(f"  {t:12s} {ps:>22s} {ns:>22s}")

    dest = os.path.join(HERE, "event_overlap.json")
    json.dump({"n_sessions": len(rows), "sessions": rows}, open(dest, "w"), indent=1)
    print(f"\nsaved {dest}\nOVERLAP_DONE")


if __name__ == "__main__":
    main()
