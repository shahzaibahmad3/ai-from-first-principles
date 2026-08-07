"""
Part 1 of the argument: XOR was never a capacity problem.

You do not need to train anything to solve XOR with a hidden layer. You can
sit down and write the weights by hand, because the decomposition is obvious
once you see it:

        XOR(a, b) = OR(a, b) AND NOT AND(a, b)

So use two hidden units -- one computing OR, one computing AND -- and have the
output unit accept "OR fired" while vetoing "AND fired." Three threshold
neurons, nine numbers, zero training.

This is worth doing explicitly because the popular story ("Minsky and Papert
showed neural networks couldn't do XOR, so the field stalled") quietly implies
nobody knew how to represent XOR. Representing it is trivial, and was known to
be trivial. Rosenblatt's own 1962 book already described multilayer
architectures. What nobody had was a way to *learn* the hidden layer from
data -- see credit_assignment.py.
"""

import numpy as np


def step(z):
    return (z > 0).astype(float)


# --- weights written by hand, not learned ------------------------------------
# hidden unit 1: OR(a, b)   -> fires when a + b >= 1
# hidden unit 2: AND(a, b)  -> fires when a + b >= 2
W1 = np.array([[1.0, 1.0],
               [1.0, 1.0]])
b1 = np.array([-0.5, -1.5])

# output: fire if OR fired, unless AND also fired (the -2 weight vetoes it)
W2 = np.array([1.0, -2.0])
b2 = -0.5


def forward(X: np.ndarray) -> np.ndarray:
    h = step(X @ W1 + b1)
    return step(h @ W2 + b2)


if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])

    h = step(X @ W1 + b1)
    out = forward(X)

    print("a hand-built 2-2-1 threshold network, no training whatsoever\n")
    print(f"{'a':>2} {'b':>2} | {'OR':>3} {'AND':>4} | {'out':>4} {'want':>5}")
    print("-" * 30)
    for xi, hi, o, t in zip(X, h, out, y):
        print(f"{xi[0]:>2.0f} {xi[1]:>2.0f} | {hi[0]:>3.0f} {hi[1]:>4.0f} | {o:>4.0f} {t:>5d}")

    assert (out == y).all()
    print(f"\nexact XOR, {W1.size + b1.size + W2.size + 1} hand-picked numbers.")
    print("Capacity was never the bottleneck. Learning the numbers was.")
