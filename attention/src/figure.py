"""
Generates the figure for Post 4, and writes results.json so that every number the
write-up quotes comes out of the same run that drew the picture.

Left  — the attention matrix a single head learns on associative recall. Among the
        item tokens the query lands on the matching key every time, with roughly
        twenty-five times the weight of any wrong one. It also attends to itself,
        which is normal: its own value is a constant the readout can ignore.
Right — the order test. Shuffle the input tokens and see how far the output moves,
        for plain attention, with positional encoding, and with a causal mask, on
        a log scale against float64 machine epsilon.

    python3 src/figure.py     # -> results.json and image.png
"""

import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import style
from attention import N_ITEMS, Head, make_batch
from budget import report as budget_report
from permutation import CASES, run as permutation_run

style.apply()

RESULTS = os.path.join(HERE, "results.json")
OUT = os.path.join(HERE, os.pardir, "image.png")

TRAIN_STEPS = 4000
TRAIN_SEEDS = (0, 1, 2)
PERM_TRIALS = 200

# ------------------------------------------------------------- measure
print("training heads...", flush=True)
runs = []
head = None
for seed in TRAIN_SEEDS:
    h = Head(seed=seed)
    losses = h.fit(steps=TRAIN_STEPS, lr=0.5, seed=100 + seed)
    acc, points_at, _ = h.evaluate()
    runs.append(dict(seed=seed, accuracy=acc, points_at_match=points_at,
                     final_loss=losses[-1]))
    if head is None:
        head = h

print("running the order test...", flush=True)
perm = permutation_run(n_trials=PERM_TRIALS, seed=0)

budget = budget_report()

# How the query row actually splits. It does NOT put all its weight on the
# matching item: it also attends to itself, because its own value vector is not
# zero and the readout can simply ignore that constant. The signal is the contrast
# between the matching item and the wrong ones, so measure exactly that.
Xs, ys, kps = make_batch(400, np.random.default_rng(11))
_, cs = head.forward(Xs)
rows = cs["A"][:, -1, :]
idx = np.arange(len(kps))
w_match = float(rows[idx, kps].mean())
w_self = float(rows[:, N_ITEMS].mean())
w_other = float(((rows[:, :N_ITEMS].sum(1) - rows[idx, kps]) / (N_ITEMS - 1)).mean())
split = dict(match=w_match, self=w_self, other_each=w_other,
             contrast=w_match / w_other)

# one clean example for the heatmap, plus an untrained head for the contrast
X, y, key_pos = make_batch(1, np.random.default_rng(7))
_, cache = head.forward(X)
A_trained = cache["A"][0]
_, cache0 = Head(seed=0).forward(X)
A_init = cache0["A"][0]
match = int(key_pos[0])

with open(RESULTS, "w") as fh:
    json.dump(dict(train_steps=TRAIN_STEPS, train_seeds=list(TRAIN_SEEDS),
                   perm_trials=PERM_TRIALS, runs=runs,
                   permutation={k: v for k, v in perm.items()},
                   machine_eps=float(np.finfo(np.float64).eps),
                   budget=budget, query_split=split,
                   example=dict(match=match,
                                query_row_trained=A_trained[-1].tolist(),
                                query_row_init=A_init[-1].tolist())), fh)
print(f"wrote {os.path.basename(RESULTS)}")

# -------------------------------------------------------------- figure
fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.6, 5.1),
                               gridspec_kw={"width_ratios": [1.05, 1]})

# ---- left: the learned attention matrix
labels = [f"item {i+1}" for i in range(N_ITEMS)] + ["query"]
axL.imshow(A_trained, cmap="pink_r", vmin=0, vmax=1, aspect="equal")
axL.set_xticks(range(N_ITEMS + 1)); axL.set_xticklabels(labels, fontsize=8.6, rotation=30)
axL.set_yticks(range(N_ITEMS + 1)); axL.set_yticklabels(labels, fontsize=8.6)
axL.set_xlabel("attends to", fontsize=9.6)
axL.set_ylabel("token doing the attending", fontsize=9.6)
axL.set_title("What one head learns: find the matching key",
              loc="left", color=style.INK)
for i in range(N_ITEMS + 1):
    for j in range(N_ITEMS + 1):
        v = A_trained[i, j]
        if v > 0.02:
            axL.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.6,
                     color=style.INK if v < .6 else style.PAPER)
# ring the cell the query should be looking at
axL.add_patch(plt.Rectangle((match - .5, N_ITEMS - .5), 1, 1, fill=False,
                            edgecolor=style.RUST, lw=2.2))
axL.grid(False)
# The query also attends to itself, which is normal and harmless -- its own value
# is a constant the readout can ignore. Say so, and give the contrast that matters.
# axes coordinates, so it sits under the x-label rather than above the matrix
axL.text(0.0, -0.40,
         f"the ringed cell is the matching key. The query splits its weight: "
         f"{split['match']:.2f} there and "
         f"{split['self']:.2f} on itself\n(its own value is a constant the readout can ignore) — but only "
         f"{split['other_each']:.2f} on each wrong\nitem, a {split['contrast']:.0f}× contrast where it counts",
         transform=axL.transAxes, fontsize=8.6, color=style.INK_SOFT,
         style="italic", va="top")

# ---- right: the order test
names = [c[0] for c in CASES]
vals = [max(perm[n]["out_max"], 1e-16) for n in names]
eps = float(np.finfo(np.float64).eps)
colours = [style.BLUE if v < 1e-12 else style.RUST for v in vals]
axR.barh(range(len(names)), vals, color=colours, height=.55, edgecolor="none")
axR.set_yticks(range(len(names)))
axR.set_yticklabels(names, fontsize=9.4)
axR.invert_yaxis()
axR.set_xscale("log")
axR.set_xlim(1e-17, 1e3)
axR.set_xlabel("largest change in the output when the tokens are shuffled (log)",
               fontsize=9.4)
axR.set_title("What it cannot do: tell you the order", loc="left", color=style.INK)
axR.axvline(eps, color=style.INK_SOFT, ls=(0, (5, 3)), lw=1.0)
axR.text(eps * 1.5, -.42, "float64 machine epsilon", fontsize=8.2,
         color=style.INK_SOFT, style="italic")
for i, v in enumerate(vals):
    axR.text(v * 1.7, i, f"{v:.1e}", va="center", fontsize=8.8,
             color=style.INK)
axR.grid(axis="x", alpha=.6)
axR.set_axisbelow(True)

fig.suptitle("Attention has no idea what order your words are in.",
             x=.012, ha="left", fontsize=13.5, y=.985)
fig.text(.012, .925,
         "A = softmax(Q Kᵀ / √d_k) compares every token with every other and never asks where either one "
         "sits. Order arrives separately — as a positional signal, or as a mask.",
         ha="left", fontsize=9.8, color=style.INK_SOFT, style="italic")
style.credit(fig, f"measured: src/attention.py, src/permutation.py · one head, "
                  f"{TRAIN_STEPS:,} steps · worst of {PERM_TRIALS} random trials")
fig.tight_layout(rect=[0, .055, 1, .905])
fig.savefig(OUT)
print(f"wrote {os.path.normpath(OUT)}")
