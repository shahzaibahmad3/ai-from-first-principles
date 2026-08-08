"""
Hero figure for the post: the one idea, and its one limit, side by side.

Left  — separable data: the perceptron finds a line, and everything is on the
        right side of it.
Right — XOR: the same algorithm, the best line it can manage, and the whole
        corner of ten points it is still forced to get wrong. No straight line
        exists that fixes them.

Both panels plot the ACTUAL decision boundary the algorithm converges (or fails
to converge) to. Palette matches the interactive playground.

    python3 src/figure.py   ->  writes image.png beside the post
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# palette — the playground's "drafting table" light theme
BG, GRID, AXIS = "#F4F6F9", "#D7DDE5", "#B7C0CB"
INK, MUTED = "#16202B", "#5A6A78"
C1, C0 = "#DE8B27", "#2E6FB0"          # fires (1) amber, quiet (0) blue
C1_FILL, C0_FILL = "#F3E2CB", "#DBE6F1"
BAD = "#C6462F"
MONO = ["Menlo", "DejaVu Sans Mono"]
SANS = ["Helvetica Neue", "Arial", "DejaVu Sans"]

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "font.family": "sans-serif", "font.sans-serif": SANS,
    "text.color": INK, "axes.edgecolor": AXIS, "axes.labelcolor": MUTED,
})


def perceptron(points, labels, lr=0.1, epochs=1000):
    """Rosenblatt's rule. Returns the separating line if it finds one."""
    w = np.zeros(2); b = 0.0
    for _ in range(epochs):
        miss = 0
        for p, t in zip(points, labels):
            guess = 1 if w @ p + b > 0 else 0
            err = t - guess
            if err:
                w = w + lr * err * p; b += lr * err; miss += 1
        if miss == 0:
            return w, b, 0
    return w, b, -1          # -1 flags "did not converge"


def best_line(points, labels):
    """
    The genuinely optimal straight-line classifier, by brute force over every
    orientation and offset. Used for XOR, where no line is perfect -- this
    finds the least-bad one, which is the honest thing to show.
    """
    best = None
    for theta in np.linspace(0, 2 * np.pi, 360, endpoint=False):
        n = np.array([np.cos(theta), np.sin(theta)])
        proj = points @ n
        for rho in np.linspace(proj.min() - .1, proj.max() + .1, 80):
            miss = int(np.sum(((proj - rho) > 0).astype(int) != labels))
            if best is None or miss < best[0]:
                best = (miss, n.copy(), -rho)
    miss, n, b = best
    return n, b, miss


def blobs(centers, labels, n, spread, seed):
    rng = np.random.default_rng(seed)
    P, L = [], []
    for c, lab in zip(centers, labels):
        P.append(rng.normal(c, spread, size=(n, 2)))
        L += [lab] * n
    return np.vstack(P), np.array(L)


def panel(ax, points, labels, title, subtitle, solver="perceptron"):
    if solver == "best":
        w, b, miss = best_line(points, labels)
    else:
        w, b, miss = perceptron(points, labels)
    lim = 1.35
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")

    # half-plane tints
    gx, gy = np.meshgrid(np.linspace(-lim, lim, 400), np.linspace(-lim, lim, 400))
    fires = (w[0] * gx + w[1] * gy + b) > 0
    from matplotlib.colors import ListedColormap
    ax.contourf(gx, gy, fires, levels=[-.5, .5, 1.5],
                colors=[C0_FILL, C1_FILL])

    # grid + axes
    for g in (-1, -0.5, 0.5, 1):
        ax.axvline(g, color=GRID, lw=.8, zorder=1)
        ax.axhline(g, color=GRID, lw=.8, zorder=1)
    ax.axvline(0, color=AXIS, lw=1, zorder=1); ax.axhline(0, color=AXIS, lw=1, zorder=1)

    # decision line
    xs = np.linspace(-lim, lim, 2)
    if abs(w[1]) > 1e-9:
        ax.plot(xs, -(w[0] * xs + b) / w[1], color=INK, lw=2.4, zorder=4)
    elif abs(w[0]) > 1e-9:
        ax.axvline(-b / w[0], color=INK, lw=2.4, zorder=4)

    # points
    for p, t in zip(points, labels):
        wrong = (1 if w @ p + b > 0 else 0) != t
        ax.scatter(*p, s=46, c=(C1 if t == 1 else C0), zorder=5,
                   edgecolors=BG, linewidths=1.1)
        if wrong:
            ax.add_patch(Circle(p, 0.085, fill=False, ec=BAD, lw=2, zorder=6))

    ax.text(0, 1.11, title, transform=ax.transAxes, color=INK,
            fontsize=14, fontweight="bold", va="bottom")
    ax.text(0, 1.03, subtitle, transform=ax.transAxes, color=MUTED,
            fontsize=10.5, va="bottom")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return miss


fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.6, 5.5))

# left: two separable clusters
P, L = blobs([(-0.55, 0.5), (0.55, -0.5)], [1, 0], n=22, spread=0.26, seed=7)
panel(axL, P, L, "It works", "two clusters a line can separate — 0 mistakes")

# right: XOR, four corner clusters, fires on the diagonal
P, L = blobs([(-0.8, -0.8), (-0.8, 0.8), (0.8, -0.8), (0.8, 0.8)],
             [0, 1, 1, 0], n=10, spread=0.10, seed=3)
miss = panel(axR, P, L, "…until it can't",
             "XOR — even the best possible line gets a whole corner wrong",
             solver="best")

fig.suptitle("A single neuron draws one straight line. That is its power and its ceiling.",
             x=.02, ha="left", fontsize=15, fontweight="bold", y=.98)
fig.text(.02, .015, "each panel plots the real boundary the algorithm converges to  ·  "
                    "src/figure.py  ·  amber = fires (1), blue = quiet (0), red ring = misclassified",
         color=MUTED, fontsize=8.5, family="monospace")
fig.tight_layout(rect=[0, .03, 1, .93])
# Anchored to this file rather than the cwd, so the command the README
# documents (`python3 src/figure.py`, run from the post root) writes the
# image beside the post instead of one directory above it.
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "image.png")
fig.savefig(OUT, dpi=155)
print(f"wrote {os.path.normpath(OUT)}  (XOR panel left {miss} points misclassified)")
