"""
Generates the figure used with Post 2, from the measurements written by
credit_assignment.py.

    python3 src/credit_assignment.py   # produces results.json (a few minutes)
    python3 src/figure.py              # -> writes image.png
"""

import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# All paths are anchored to this file rather than the cwd, so the commands
# documented in the README work from the repo root as well as from inside src/.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import style

style.apply()

RESULTS = os.path.join(HERE, "results.json")
OUT = os.path.join(HERE, os.pardir, "image.png")

if not os.path.exists(RESULTS):
    sys.exit("results.json not found -- run `python3 src/credit_assignment.py` first")

with open(RESULTS) as fh:
    data = json.load(fh)

sweep = data["sweep"]
curves = data["curves"]
stride = data["stride"]
widths = [s["hidden"] for s in sweep]
rates = [s["rate"] for s in sweep]

worst = sweep[0]
print(f"hidden={worst['hidden']}: {worst['rate']:.1%} of "
      f"{worst['n_seeds']} inits solved XOR")


# ------------------------------------------------------------------- figure
fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.4, 5.1),
                               gridspec_kw={"width_ratios": [1.15, 1]})

# ---- left: loss traces, split by whether the run ever got there
solved_curves = [c for c in curves if c["solved"]]
stuck_curves = [c for c in curves if not c["solved"]]

for c in solved_curves:
    ep = np.arange(len(c["loss"])) * stride
    axL.plot(ep, c["loss"], color=style.BLUE, lw=.8, alpha=.30)
for c in stuck_curves:
    ep = np.arange(len(c["loss"])) * stride
    axL.plot(ep, c["loss"], color=style.RUST, lw=1.1, alpha=.55)

axL.set_xlabel("epoch")
axL.set_ylabel("training loss (MSE, log scale)")
axL.set_title(f"{len(curves)} runs, identical architecture, different starting weights",
              loc="left", color=style.INK)
axL.grid(alpha=.75)
axL.set_axisbelow(True)
# log scale: converged runs fall several decades while stuck runs sit near 0.125.
# On a linear axis that whole separation collapses into the top of the plot.
axL.set_yscale("log")
axL.set_ylim(8e-5, 0.6)

# The failures are not one population. Most hedge at exactly 0.5 on two of the four
# points, which costs (0 + 0 + 0.25 + 0.25)/4 = 0.125 precisely. A couple fail a
# different way — collapsing to one constant across the points they never separated.
# src/inspect_stuck.py prints both groups point by point.
hedge = [c for c in stuck_curves if abs(c["loss"][-1] - 0.125) < 0.005]
other = [c for c in stuck_curves if c not in hedge]

label = (f"{len(hedge)} of {len(curves)} runs stall at exactly 0.125\n"
         f"— two points learned, a coin flip on the other two")
if other:
    losses = sorted({round(c["loss"][-1], 3) for c in other})
    label += (f"\n{len(other)} more fail differently, at "
              f"{' / '.join(f'{v:.3f}' for v in losses)}")

# Sits in the gap between the plateau and the descending bundle, so it never
# overprints the very lines it is describing.
axL.text(.97, .60, label, transform=axL.transAxes, ha="right", va="top",
         fontsize=9.5, color=style.RUST)
axL.text(.03, .13, f"{len(solved_curves)} runs converge,\nat wildly different epochs",
         transform=axL.transAxes, ha="left", va="bottom",
         fontsize=9.5, color=style.BLUE)

# ---- right: how often gradient descent finds a solution, by hidden width
bars = axR.bar([str(w) for w in widths], rates, width=.62,
               color=[style.RUST if r < .9 else style.BLUE for r in rates],
               edgecolor="none")
for w, r in zip([str(x) for x in widths], rates):
    axR.text(w, r + .018, f"{r:.1%}", ha="center", va="bottom",
             fontsize=10.5, color=style.INK)

axR.axhline(1.0, color=style.INK_SOFT, ls=(0, (5, 3)), lw=1.0)
axR.set_ylim(0, 1.14)
axR.set_yticks([0, .25, .5, .75, 1.0])
axR.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
axR.set_xlabel("hidden units")
axR.set_ylabel(f"share of {worst['n_seeds']} random inits that solved XOR")
axR.set_title("Extra width buys reliability, not capacity",
              loc="left", color=style.INK)
axR.grid(axis="y", alpha=.75)
axR.set_axisbelow(True)

fig.suptitle("XOR was never the hard part. Finding the weights was.",
             x=.012, ha="left", fontsize=13.5, y=.985)
fig.text(.012, .925, "Two hidden units already represent XOR exactly (capacity.py writes the "
                     "weights by hand) — so every failure below is an optimisation failure, "
                     "not a limit on capacity.",
         ha="left", fontsize=9.8, color=style.INK_SOFT, style="italic")
style.credit(fig, f"measured: src/credit_assignment.py · 2-h-1 network, hand-written "
                  f"backprop · {data['epochs']:,} epochs · lr 1.0")
fig.tight_layout(rect=[0, .028, 1, .905])
fig.savefig(OUT)
print(f"wrote {os.path.normpath(OUT)}")
