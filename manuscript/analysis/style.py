"""Shared plotting style for the open-world ITS manuscript.

Palette: Paul Tol 'bright' qualitative scheme, which is distinguishable under
the common forms of colour-vision deficiency and survives greyscale printing
reasonably well. Fonts are serif so that figures sit comfortably next to the
body text.
"""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "figures"
DATADIR = ROOT / "data"

# Paul Tol bright
BLUE = "#4477AA"
RED = "#EE6677"
GREEN = "#228833"
YELLOW = "#CCBB44"
CYAN = "#66CCEE"
PURPLE = "#AA3377"
GREY = "#BBBBBB"
DARK = "#2B2B2B"

VIEW_COLOUR = {"core": BLUE, "its1": RED, "its2": GREEN}
VIEW_LABEL = {"core": "ITS-core", "its1": "ITS1", "its2": "ITS2"}
VIEW_ORDER = ["core", "its1", "its2"]

MODEL_ORDER = ["M0", "M1", "M2", "M3", "M4"]
MODEL_SUBTITLE = {
    "M0": "proxy",
    "M1": "+view",
    "M2": "+tax.",
    "M3": "+epis.",
    "M4": "+both",
}


def use_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def panel_label(ax, letter, x=-0.085, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="left")


def save(fig, stem):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / f"{stem}.pdf")
    fig.savefig(FIGDIR / f"{stem}.png")
    plt.close(fig)
    print(f"  wrote figures/{stem}.pdf and .png")
