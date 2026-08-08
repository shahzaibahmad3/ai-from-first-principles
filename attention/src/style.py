"""
Shared plot styling: warm paper, ink, two muted accents, serif type.

Deliberately not a dark "tech" theme -- these are meant to read like figures
printed in a journal, because that is what they are: plots of real measured
data, not decoration.
"""

import matplotlib as mpl

PAPER = "#FAF7F2"
INK = "#1C1B19"
INK_SOFT = "#5C574F"
RULE = "#DDD5C8"
RUST = "#A8431C"
BLUE = "#2A5674"
SAND = "#C9A227"

SERIF = ["Charter", "Iowan Old Style", "Georgia", "DejaVu Serif"]
MONO = ["Menlo", "DejaVu Sans Mono"]


def apply():
    mpl.rcParams.update({
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "font.family": "serif",
        "font.serif": SERIF,
        "font.size": 11,
        "text.color": INK,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 12,
        "axes.titleweight": "normal",
        "axes.labelsize": 10.5,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "grid.color": RULE,
        "grid.linewidth": 0.7,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "lines.linewidth": 1.8,
        "figure.dpi": 160,
    })


def credit(fig, text):
    """Small source line, bottom-left -- these are measured, so say so."""
    fig.text(0.012, 0.015, text, fontsize=8, color=INK_SOFT,
             family="monospace", ha="left", va="bottom")
