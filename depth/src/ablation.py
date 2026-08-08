"""
Does the network actually learn, and which of the four fixes is doing the work?

The measurement that matters is not "does my favourite config win" but "how deep
can each one go before it stops learning at all". So every configuration gets its
own learning-rate sweep and is judged at its best -- comparing configs at one
shared learning rate would just be measuring which one happens to like that
number, and different depths have wildly different stable ranges.

Writes results.json, which figure.py reads. Takes roughly ten minutes.

    python3 src/ablation.py           # the real sweep
    QUICK=1 python3 src/ablation.py   # a tiny version, for checking the pipeline
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from depth import CONFIGS, LABELS, DeepNet, rings

RESULTS_PATH = os.path.join(HERE, "results.json")

DEPTHS = [2, 5, 10, 20, 40]
LRS = [0.01, 0.03, 0.1, 0.3]
SEEDS = [1, 2, 3, 4, 5]
WIDTH = 16
EPOCHS = 3000
MOMENTUM = 0.9

# Depths whose full per-layer gradient profile gets stored for the figure.
PROFILE_DEPTHS = [20, 40]


def one_run(depth, cfg, lr, seed, X, y):
    net = DeepNet(depth=depth, width=WIDTH, seed=seed, **cfg)
    norms0 = net.grad_norms(X, y)
    net.fit(X, y, epochs=EPOCHS, lr=lr, momentum=MOMENTUM)
    return net.accuracy(X, y), net.loss(X, y), norms0


def measure(depth, cfg, X, y):
    """Best-of-lr median accuracy, plus init diagnostics that don't depend on lr."""
    best = None
    for lr in LRS:
        accs, losses = [], []
        for s in SEEDS:
            a, l, _ = one_run(depth, cfg, lr, s, X, y)
            accs.append(a); losses.append(l)
        med = float(np.median(accs))
        if best is None or med > best["accuracy"]:
            best = dict(lr=lr, accuracy=med, accuracies=accs,
                        median_loss=float(np.median(losses)))

    # init-time diagnostics: gradient at both ends, and whether the forward
    # signal survived the trip up
    firsts, lasts, rms = [], [], []
    for s in SEEDS:
        net = DeepNet(depth=depth, width=WIDTH, seed=s, **cfg)
        n = net.grad_norms(X, y)
        _, cache = net.forward(X)
        firsts.append(n[0]); lasts.append(n[-1])
        rms.append(float(np.sqrt(np.mean(cache["a_in"][-1] ** 2))))
    f, l = float(np.median(firsts)), float(np.median(lasts))
    best.update(grad_layer1=f, grad_last=l,
                attenuation=(l / f) if f > 0 else None,   # None encodes "dead"
                forward_rms=float(np.median(rms)))
    return best


if __name__ == "__main__":
    if os.environ.get("QUICK"):
        DEPTHS = [2, 10]
        LRS = [0.05]
        SEEDS = [1]
        EPOCHS = 200

    X, y = rings()
    print(f"depths {DEPTHS} x {len(CONFIGS)} configs x {len(LRS)} lrs x {len(SEEDS)} seeds")
    print(f"{EPOCHS:,} epochs each, momentum {MOMENTUM}, width {WIDTH}")
    print(f"that is {len(DEPTHS)*len(CONFIGS)*len(LRS)*len(SEEDS):,} training runs\n", flush=True)

    table = {}
    for name, cfg in CONFIGS.items():
        print(f"{LABELS[name]}")
        print(f"  {'depth':>6}  {'acc':>7}  {'best lr':>8}  {'layer1 grad':>12}  "
              f"{'attenuation':>12}  {'fwd RMS':>10}", flush=True)
        table[name] = {}
        for depth in DEPTHS:
            r = measure(depth, cfg, X, y)
            table[name][str(depth)] = r
            att = "dead" if r["attenuation"] is None else f"{r['attenuation']:.2e}"
            print(f"  {depth:>6}  {r['accuracy']*100:>6.1f}%  {r['lr']:>8}  "
                  f"{r['grad_layer1']:>12.3e}  {att:>12}  {r['forward_rms']:>10.3f}",
                  flush=True)
        print(flush=True)

    # Per-layer gradient profiles: the left panel of the figure.
    profiles = {}
    for name, cfg in CONFIGS.items():
        profiles[name] = {}
        for depth in PROFILE_DEPTHS:
            if depth not in DEPTHS:
                continue
            per_seed = [DeepNet(depth=depth, width=WIDTH, seed=s, **cfg).grad_norms(X, y)
                        for s in SEEDS]
            profiles[name][str(depth)] = [float(np.median(v)) for v in zip(*per_seed)]

    with open(RESULTS_PATH, "w") as fh:
        json.dump(dict(depths=DEPTHS, lrs=LRS, seeds=SEEDS, width=WIDTH,
                       epochs=EPOCHS, momentum=MOMENTUM,
                       labels=LABELS, table=table, profiles=profiles), fh)
    print(f"wrote {os.path.basename(RESULTS_PATH)}", flush=True)

    print("\nTwo different ways to die, and each fix addresses one of them:")
    print("  sigmoid          the forward signal is fine; the GRADIENT vanishes")
    print("  ReLU + He        gradient survives; forward signal slowly thins out")
    print("  + residual       forward signal EXPLODES until the output saturates,")
    print("                   and a saturated sigmoid has exactly zero derivative")
    print("  + RMSNorm        forward signal pinned to 1 by construction, and")
    print("                   the gradient still arrives. This pair is what every")
    print("                   transformer is built from.")
