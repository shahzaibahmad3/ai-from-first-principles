"""
Generates the figure used with Post 3, from the measurements written by
ablation.py.

    python3 src/ablation.py   # produces results.json (about ten minutes)
    python3 src/figure.py     # -> writes image.png
"""

import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import style

style.apply()

RESULTS = os.path.join(HERE, "results.json")
OUT = os.path.join(HERE, os.pardir, "image.png")

if not os.path.exists(RESULTS):
    sys.exit("results.json not found -- run `python3 src/ablation.py` first")

with open(RESULTS) as fh:
    data = json.load(fh)

table, profiles, depths = data["table"], data["profiles"], data["depths"]

ORDER = ["sigmoid", "relu_he", "relu_he_res", "relu_he_res_norm"]
SHORT = {
    "sigmoid": "sigmoid, N(0,1)",
    "relu_he": "ReLU + He",
    "relu_he_res": "ReLU + He + residual",
    "relu_he_res_norm": "ReLU + He + residual + RMSNorm",
}
COLOUR = {
    "sigmoid": style.RUST,
    "relu_he": style.BLUE,
    "relu_he_res": style.SAND,
    "relu_he_res_norm": style.INK,
}

PROFILE_DEPTH = "40" if "40" in profiles[ORDER[0]] else str(depths[-1])
FLOOR = 1e-15          # where an exactly-zero gradient gets drawn

fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.4, 5.1),
                               gridspec_kw={"width_ratios": [1.12, 1]})

# ---- left: how much gradient each layer receives, at one depth
zero_configs = []
for key in ORDER:
    prof = profiles[key].get(PROFILE_DEPTH)
    if prof is None:
        continue
    layers = np.arange(1, len(prof) + 1)
    vals = np.array(prof, dtype=float)
    if np.all(vals == 0):
        zero_configs.append(key)
        continue
    plotted = np.where(vals > 0, vals, FLOOR)
    axL.plot(layers, plotted, color=COLOUR[key], lw=1.6,
             marker="o", ms=2.6, label=SHORT[key])
    if np.any(vals == 0):     # partially dead: mark the zeros honestly
        z = layers[vals == 0]
        axL.plot(z, np.full(len(z), FLOOR), color=COLOUR[key],
                 lw=0, marker="x", ms=5)

axL.set_yscale("log")
axL.set_ylim(FLOOR / 3, 60)   # headroom so the legend clears the traces
axL.set_xlabel(f"layer (1 is furthest from the loss) — depth {PROFILE_DEPTH}")
axL.set_ylabel("gradient norm reaching this layer (log scale)")
axL.set_title("The gradient does not survive the trip back",
              loc="left", color=style.INK)
axL.grid(alpha=.75)
axL.set_axisbelow(True)
axL.axhline(1e-6, color=style.INK_SOFT, ls=(0, (5, 3)), lw=1.0)
axL.text(len(profiles[ORDER[0]][PROFILE_DEPTH]) * .98, 1.6e-6,
         "below here a layer is, in practice, frozen",
         ha="right", va="bottom", fontsize=8.6, color=style.INK_SOFT, style="italic")
if zero_configs:
    axL.text(.97, .05,
             "\n".join(f"{SHORT[k]}:\nexactly zero at every layer" for k in zero_configs),
             transform=axL.transAxes, fontsize=8.6, color=style.SAND,
             va="bottom", ha="right")
axL.legend(loc="upper left", fontsize=8.4, framealpha=.95)

# ---- right: does it actually learn, by depth
for key in ORDER:
    accs = [table[key][str(d)]["accuracy"] * 100 for d in depths]
    axR.plot(depths, accs, color=COLOUR[key], lw=1.7, marker="o", ms=4,
             label=SHORT[key])
axR.set_xscale("log")
axR.set_xticks(depths)
axR.set_xticklabels([str(d) for d in depths])
# a log axis also draws minor ticks (3x10^0, 4x10^0 ...) whose labels run straight
# through the plot area; we only ever want the five depths we measured
axR.minorticks_off()
axR.tick_params(axis="x", which="minor", bottom=False, labelbottom=False)
axR.set_ylim(40, 105)
axR.set_yticks([50, 60, 70, 80, 90, 100])
axR.set_yticklabels([f"{v}%" for v in [50, 60, 70, 80, 90, 100]])
axR.axhline(50, color=style.INK_SOFT, ls=(0, (2, 3)), lw=.9)
axR.text(depths[-1], 51.2, "chance", fontsize=8.4, color=style.INK_SOFT,
         style="italic", ha="right", va="bottom")
axR.set_xlabel("depth (weight matrices)")
axR.set_ylabel("points classified correctly, best of a learning-rate sweep")
axR.set_title("And so it stops learning at all", loc="left", color=style.INK)
axR.grid(alpha=.75)
axR.set_axisbelow(True)
axR.legend(loc="lower left", fontsize=8.4, framealpha=.95)

fig.suptitle("Stacking was the easy part. Getting the gradient back down wasn't.",
             x=.012, ha="left", fontsize=13.5, y=.985)
fig.text(.012, .925,
         "Every layer multiplies the gradient by something. With a sigmoid that something is at "
         "most 0.25, and twenty layers of it leaves nothing to learn from.",
         ha="left", fontsize=9.8, color=style.INK_SOFT, style="italic")
style.credit(fig, f"measured: src/ablation.py · width {data['width']} · "
                  f"{data['epochs']:,} epochs · momentum {data['momentum']} · "
                  f"median of {len(data['seeds'])} seeds")
fig.tight_layout(rect=[0, .028, 1, .905])
fig.savefig(OUT)
print(f"wrote {os.path.normpath(OUT)}")
