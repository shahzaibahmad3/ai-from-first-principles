"""
What does a stuck run actually output?

credit_assignment.py reports *how often* gradient descent solves XOR. It does not
say what failure looks like from the inside, and that turns out to be the most
interesting part: the failures are not random noise, they have a signature.

This script retrains the specific seeds that fail and prints their four outputs
next to the four targets. Every number the write-up quotes about the failure mode
comes from here, so it can be checked rather than taken on trust.

    python3 src/inspect_stuck.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from credit_assignment import EPOCHS, MLP, X, Y

HIDDEN = 2


def outputs(seed: int, epochs: int = EPOCHS):
    """Train one net from `seed` and return (solved, final loss, the 4 outputs)."""
    net = MLP(n_in=2, n_hidden=HIDDEN, n_out=1, lr=1.0, seed=seed)
    losses = net.fit(X, Y, epochs)
    _, a2 = net.forward(X)
    return bool((net.predict(X) == Y).all()), float(losses[-1]), a2.flatten()


if __name__ == "__main__":
    n_seeds = int(os.environ.get("N_SEEDS", 60))
    print(f"retraining 2-{HIDDEN}-1 on XOR, seeds 0..{n_seeds - 1}, "
          f"{EPOCHS:,} epochs each")
    print(f"targets: {Y.tolist()}\n")

    stuck = []
    for seed in range(n_seeds):
        solved, loss, out = outputs(seed)
        if not solved:
            stuck.append((seed, loss, out))

    print(f"{len(stuck)} of {n_seeds} runs failed to solve XOR\n")
    print(f"{'seed':>5}  {'final loss':>10}   outputs")
    print("-" * 58)
    for seed, loss, out in stuck:
        pretty = "  ".join(f"{v:.3f}" for v in out)
        print(f"{seed:>5d}  {loss:>10.4f}   [{pretty}]")

    losses = np.array([s[1] for s in stuck])
    print(f"\nmedian final loss among the failures: {np.median(losses):.5f}")

    # The signature: two points learned confidently, the other two pinned at 0.5.
    # A point sitting at 0.5 contributes (0.5)^2 = 0.25 to the squared error, so
    # hedging on exactly two of four points costs (0 + 0 + 0.25 + 0.25) / 4 = 0.125.
    hedged = [s for s in stuck if np.sum(np.abs(s[2] - 0.5) < 0.05) == 2]
    print(f"{len(hedged)} of {len(stuck)} failures hedge at ~0.5 on exactly two "
          f"points (loss 0.125)")
    others = [s for s in stuck if s not in hedged]
    if others:
        vals = ", ".join(f"seed {s[0]} at {s[1]:.3f}" for s in others)
        print(f"the rest land elsewhere: {vals}")
