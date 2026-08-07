"""
Part 2 of the argument: learning the hidden layer is the hard part.

capacity.py showed that a 2-2-1 network represents XOR exactly with nine
numbers you can write down by hand. This file asks the question that actually
held the field up for a decade and a half: can you *find* those numbers
starting from random ones?

Backpropagation is written out longhand below -- no autograd -- because the
whole point is the line that pushes error backwards through W2 to work out how
much each hidden unit is to blame. That step is "credit assignment," and it is
the thing Rosenblatt's rule could not do: his rule needs a target for the unit
it is updating, and a hidden unit has no target.

The experiment: train the same architecture from many random initialisations
and count how often it actually gets there. It does not always get there.
"""

import os

import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class MLP:
    def __init__(self, n_in: int, n_hidden: int, n_out: int = 1, lr: float = 0.5, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 1, size=(n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, 1, size=(n_hidden, n_out))
        self.b2 = np.zeros(n_out)
        self.lr = lr

    def forward(self, X):
        a1 = sigmoid(X @ self.W1 + self.b1)
        a2 = sigmoid(a1 @ self.W2 + self.b2)
        return a1, a2

    def train_step(self, X, y):
        n = X.shape[0]
        a1, a2 = self.forward(X)
        target = y.reshape(-1, 1)

        # ---- backpropagation, by hand ----
        delta_out = (a2 - target) * a2 * (1 - a2)        # dL/dz2
        grad_W2 = a1.T @ delta_out / n
        grad_b2 = delta_out.mean(axis=0)

        # the credit-assignment step: project the output error back through W2
        delta_hidden = (delta_out @ self.W2.T) * a1 * (1 - a1)
        grad_W1 = X.T @ delta_hidden / n
        grad_b1 = delta_hidden.mean(axis=0)

        self.W2 -= self.lr * grad_W2
        self.b2 -= self.lr * grad_b2
        self.W1 -= self.lr * grad_W1
        self.b1 -= self.lr * grad_b1
        return float(np.mean((a2 - target) ** 2))

    def fit(self, X, y, epochs: int):
        return [self.train_step(X, y) for _ in range(epochs)]

    def predict(self, X):
        return (self.forward(X)[1] > 0.5).astype(int).flatten()


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
Y = np.array([0, 1, 1, 0])

EPOCHS = 20_000
N_SEEDS = 300
HIDDEN_SIZES = [2, 3, 4, 8]


def run(n_hidden: int, seed: int, epochs: int = EPOCHS):
    net = MLP(n_in=2, n_hidden=n_hidden, n_out=1, lr=1.0, seed=seed)
    losses = net.fit(X, Y, epochs)
    solved = bool((net.predict(X) == Y).all())   # plain bool, not np.bool_
    return solved, float(losses[-1]), losses


CURVE_HIDDEN = 2       # collect loss traces at the width that actually struggles
N_CURVES = 60
CURVE_STRIDE = 40      # subsample traces so results.json stays small

# Anchored to this file, not the cwd, so `python3 src/credit_assignment.py` from
# the repo root and `python3 credit_assignment.py` from src/ both land here.
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")


if __name__ == "__main__":
    import json

    # QUICK=1 runs a tiny version end to end. Useful for checking the whole
    # pipeline (including the JSON write) without waiting on the real sweep.
    if os.environ.get("QUICK"):
        EPOCHS, N_SEEDS, N_CURVES = 400, 12, 8

    print(f"training 2-{HIDDEN_SIZES}-1 nets on XOR from {N_SEEDS} random inits")
    print(f"{EPOCHS:,} epochs each, lr = 1.0, plain full-batch gradient descent")
    print(f"this is {N_SEEDS * len(HIDDEN_SIZES):,} full training runs\n", flush=True)
    print(f"{'hidden':>7}  {'solved':>8}  {'failed':>7}  {'median final loss':>18}")
    print("-" * 46, flush=True)

    sweep, curves = [], []
    for n_hidden in HIDDEN_SIZES:
        results = [run(n_hidden, seed, EPOCHS) for seed in range(N_SEEDS)]
        solved = int(sum(r[0] for r in results))
        med = float(np.median([r[1] for r in results]))
        sweep.append({"hidden": int(n_hidden), "solved": solved,
                      "n_seeds": int(N_SEEDS), "rate": solved / N_SEEDS,
                      "median_final_loss": med})
        print(f"{n_hidden:>7d}  {solved / N_SEEDS:>7.1%}  {N_SEEDS - solved:>7d}  "
              f"{med:>18.4f}", flush=True)

        if n_hidden == CURVE_HIDDEN:
            for seed in range(min(N_CURVES, len(results))):
                ok, _, losses = results[seed]
                curves.append({"seed": int(seed), "solved": bool(ok),
                               "loss": [round(float(v), 5) for v in losses[::CURVE_STRIDE]]})

    with open(RESULTS_PATH, "w") as fh:
        json.dump({"epochs": int(EPOCHS), "n_seeds": int(N_SEEDS),
                   "stride": int(CURVE_STRIDE), "curve_hidden": int(CURVE_HIDDEN),
                   "sweep": sweep, "curves": curves}, fh)

    print(f"\nwrote {os.path.basename(RESULTS_PATH)}", flush=True)
    print()
    print("The architecture can always represent XOR -- capacity.py proves that")
    print("by hand. Whether gradient descent *finds* a solution depends on where")
    print("it starts. Widening the hidden layer does not add representational")
    print("power here; it adds paths down, so optimisation fails less often.")
