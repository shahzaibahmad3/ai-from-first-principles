"""
How much of the gradient actually reaches the first layer?

Nothing here trains. Every number is measured at initialisation, which is the
point: the network has not had a chance to do anything wrong yet, and the signal
is already gone. Fast enough to re-run whenever you want to poke at it.

Reported per configuration and depth:

  layer 1        the gradient norm at the layer furthest from the loss
  last layer     the gradient norm at the layer nearest it
  attenuation    last / first -- how many times larger the near end is
  fwd RMS        root-mean-square activation entering the last hidden layer,
                 which tells you whether the *forward* signal survived either

    python3 src/gradient_flow.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from depth import CONFIGS, LABELS, DeepNet, rings

DEPTHS = [2, 5, 10, 20, 40]
WIDTH = 16
SEEDS = [1, 2, 3, 4, 5]


def measure(depth, cfg, seed, X, y):
    net = DeepNet(depth=depth, width=WIDTH, seed=seed, **cfg)
    norms = net.grad_norms(X, y)
    _, cache = net.forward(X)
    # activation entering the final weight matrix: did the forward pass survive?
    last_hidden = cache["a_in"][-1]
    fwd_rms = float(np.sqrt(np.mean(last_hidden ** 2)))
    return norms, fwd_rms


def profile(depth, cfg, seed=1):
    """Per-layer gradient norms for one network -- the shape the figure plots."""
    X, y = rings()
    return measure(depth, cfg, seed, X, y)[0]


if __name__ == "__main__":
    X, y = rings()

    for name, cfg in CONFIGS.items():
        print(f"\n{LABELS[name]}")
        print(f"  {'depth':>6}  {'layer 1':>11}  {'last layer':>11}  "
              f"{'attenuation':>12}  {'fwd RMS':>9}")
        print("  " + "-" * 58)
        for depth in DEPTHS:
            first, last, rmss = [], [], []
            for s in SEEDS:
                norms, rms = measure(depth, cfg, s, X, y)
                first.append(norms[0]); last.append(norms[-1]); rmss.append(rms)
            # Ratio of the medians, not the median of the ratios, so this agrees
            # with the two columns printed beside it -- and with ablation.py,
            # which stores the same statistic.
            mf, ml = float(np.median(first)), float(np.median(last))
            att_s = "dead" if mf == 0 else f"{ml / mf:.2e}"
            print(f"  {depth:>6}  {mf:>11.3e}  {ml:>11.3e}  "
                  f"{att_s:>12}  {np.median(rmss):>9.3f}")

    print("\nWhy sigmoid cannot win this: its derivative peaks at 0.25, so every")
    print("layer the gradient passes through multiplies it by at most a quarter.")
    print("Twenty layers of that is a factor of 4^-20, and no amount of patience")
    print("recovers a number that small.")
    print("\nWhy the residual row goes to zero rather than merely small: a skip")
    print("connection adds its input back, so Var(x) grows with every block")
    print("(Var(x_next) = Var(x) + Var(f(x))). By layer 20 the activations are")
    print("large enough to saturate the output sigmoid completely, and a")
    print("saturated sigmoid has exactly zero derivative. Normalisation is what")
    print("stops the growth -- which is why the two are always paired.")
