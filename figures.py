#!/usr/bin/env python3
"""The eleven figures of the paper, drawn from the tables in results/.

One function per figure, named after the PDF it writes under figure/:
fig<N>_<words>, with N the figure number of the paper and the words taken
from its caption or the title of its section.  A figure reads only tables
that ``analysis.py`` writes under results/ (Fig. 1 is a schematic and reads
none).  Nothing here opens data/ or evaluates a network, and the eleven
figures take well under a minute.

The quantities on the axes are those defined in the docstring of
``common.py``: the effective Schmidt rank K_f = 1 / Tr rho_A^2 of the reduced
state across the cut d_L = 5, the crossing-gauge weight Gamma_1 on the uniform
left mode, the target-mode population w_star of the target-gauged sign table,
the constraint graph of a training set with its identifiable left-pair
fraction f_L and the graph-only rank K_perc, the pair correlations Q
(magnitude, mean C_h^2) and R (target sign) inside and between components,
and the splitting entropy B and subspace overlap R_chi of the clustered
conditional functions.  A state quantity is the mean over the four long-range
output positions j = 8, ..., 11 unless a docstring says otherwise.  Each
function below says what every panel plots and from which columns of which
table; the analysis item that writes the table (its directory under
results/) says in ``analysis.py`` how it is computed.

WHAT THE FIGURES SHOW
---------------------
Fig. 1 separates three layers.  The constraint layer is the bipartite graph
of the training set: adding distinct inputs is bond percolation on it, and
the identifiable support, the pairs of left substrings whose relative target
sign the data fix, grows monotonically with the number of added inputs.  The
state layer is rho_A of the learned sign table, which can differentiate
(K_f up) or recohere (K_f down) inside that support.  The prediction layer,
the block accuracy A_blk, is a many-to-one readout of the state.

Figs. 2-4 follow the state along training.  Differentiation reaches the
long-range outputs before they are predicted (Fig. 2); at N = 64
interpolation leaves a differentiated plateau, and at N = 1,280 the
generalization onset goes with low K_f (Fig. 3); many states share one
accuracy (Fig. 4).

Figs. 5-8 are the branching study: a starting model fitted to the N = 64 base
inputs is continued on the base inputs plus n distinct inputs, plus n
repeated base inputs, or on the base inputs alone.  Every contrast is
distinct minus duplicate, formed within a starting model and averaged over the
25 starting models.  The response of the state is nonmonotone in n:
differentiation at +16 to +64, target-selective recoherence (K_f down,
w_star up) from +192 on (Figs. 5, 6), while the graph percolates
monotonically (Fig. 7).  At +1,216 the graph is connected from update 0, so
the order in which the responses appear along the branch belongs to the
learned state alone (Fig. 8).

Figs. 9-11 are the first retreat events of the N = 1,280 runs, at their
last-high, retreat and recovery checkpoints: conditional functions split at
retreat and merge many-to-one at recovery (Fig. 9); C_h^2 and w_star return
while the sign table does not (Fig. 10); both hold across the clustering
threshold (Fig. 11).

CONVENTIONS
-----------
* Unevenly spaced updates and additions are plotted at equal spacing in
  their order (Figs. 5, 6) or on a symmetric-log axis (Figs. 7, 8).
* Bands and bars: interquartile ranges over starting models (Fig. 5), central
  95 per cent bootstrap intervals over starting models (Figs. 6, 8),
  interquartile ranges over runs or events (Figs. 3 inset, 11A).
* Matplotlib settings: every figure function first resets Matplotlib to its
  library defaults and then applies its own rcParams (``figure_style``), so
  a figure comes out the same whether it is drawn alone or after others in
  one process, and whatever matplotlibrc the reader has.

COMMAND LINE
------------
    python3 figures.py --list               every figure: name, paper location, file, analyses it needs
    python3 figures.py NAME [NAME ...]      draw the named figures in this process
    python3 figures.py --all                every figure, each in its own process

The file sets OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS,
VECLIB_MAXIMUM_THREADS and NUMEXPR_NUM_THREADS to 1 unless they are set
already, as ``analysis.py`` does.

Paper location        function (= file stem under figure/)   tables read (under results/)
--------------        ------------------------------------   ----------------------------
Fig. 1, Sec. 2        fig1_support_state_performance         none (schematic)
Fig. 2, Sec. 4.1      fig2_output_position_dependence        output_position_dependence/
Fig. 3, Secs. 4.2,    fig3_training_trajectories             training_trajectories/, retreat_events/,
  4.3                                                        state_vs_accuracy/
Fig. 4, Sec. 4.4      fig4_state_vs_accuracy                 state_vs_accuracy/
Fig. 5, Sec. 5.1      fig5_starting_model_branches           largest_addition_branches/
Fig. 6, Sec. 5.2      fig6_added_data_size                   branch_endpoint_state/
Fig. 7, Sec. 6        fig7_constraint_percolation_vs_state   constraint_percolation/, branch_endpoint_state/
Fig. 8, Sec. 7        fig8_fixed_identifiable_support        fixed_support_timing/
Fig. 9, Sec. 8.2      fig9_conditional_function_change       conditional_function_splitting/
Fig. 10, Sec. 8.3     fig10_retreat_recovery_fixed_graph     retreat_recurrence/, retreat_pair_correlations/
Fig. 11, App. B.2     fig11_retreat_recovery_robustness      conditional_function_splitting/
"""

from __future__ import annotations

import os

# One BLAS thread, set before numpy is imported, as for the analyses.
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
for _name in THREAD_VARIABLES:
    os.environ.setdefault(_name, "1")

import argparse
import csv
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.transforms import ScaledTranslation

from common import ADDITION_SIZES, FIGURE, N_LEFT, RESULTS, ROOT

# Colours shared by several figures.
INK = "#20252B"            # text, neutral lines, the control branch
TEAL = "#13866F"
PURPLE = "#5A4D9C"
BLUE = "#3478B8"
ORANGE = "#D86B2B"
CORAL = "#C94B49"
GRAY = "#737B87"
LIGHT_GRAY = "#D7DBE0"
# Colours of the N = 1,280 runs and of their retreat events (Figs. 3, 4, 9, 11).
RUN_BLUE = "#2a6fbb"       # held-out accuracy; checkpoints before onset; last-high; w_star in Fig. 11
RUN_ORANGE = "#e48c36"     # checkpoints after onset; Gamma_1 in Fig. 11
RETREAT_RED = "#c43d3d"    # retreat; K_f in Fig. 11; the line w_star = A_blk in Fig. 4
RECOVERY_GREEN = "#36845b"
CONTROL_GREEN = "#3f9350"  # control intervals


# ==========================================================================
# Helpers shared by the figures
# ==========================================================================

def figure_style(rc: Optional[dict] = None) -> None:
    """Reset Matplotlib to its library defaults, then apply the settings ``rc`` of one figure.

    Every figure function calls this before it creates its figure.  The
    reset removes whatever a figure drawn before it in the same process, or
    a matplotlibrc file of the reader, has set; the settings are left in force
    until the next call, so those read at ``savefig`` time (savefig.bbox,
    savefig.dpi, pdf.fonttype) apply to the file the figure writes.
    """
    mpl.rcdefaults()
    if rc:
        mpl.rcParams.update(rc)


def need(path: Path) -> Path:
    """``path`` if it exists; otherwise stop and name the analysis that writes it.

    Every table under results/ lies in the directory of the analysis item
    that writes it, so the item is the first component of the path.
    """
    if not path.exists():
        item = path.relative_to(RESULTS).parts[0]
        raise SystemExit(f"missing {path.relative_to(ROOT)}\n"
                         f"run  python3 analysis.py {item}  first")
    return path


def read_rows(path: Path) -> list[dict[str, str]]:
    """Rows of a CSV table as dictionaries of strings; each figure converts the cells it uses."""
    with need(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_npz(path: Path) -> dict[str, np.ndarray]:
    """Every array of an .npz file (no pickled objects)."""
    with np.load(need(path), allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def read_frame(path: Path, **options) -> pd.DataFrame:
    """A CSV table as a DataFrame; ``options`` go to ``pd.read_csv``."""
    return pd.read_csv(need(path), **options)


def save(fig, name: str, **options) -> None:
    """Write ``figure/<name>.pdf`` and close the figure; ``options`` go to ``savefig``."""
    output = FIGURE / f"{name}.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, **options)
    plt.close(fig)
    print(f"wrote {output.relative_to(ROOT)}")


# ==========================================================================
# Fig. 1: constraint support, learned state and predictive performance
# ==========================================================================

# Visual padding that the rounded-box style adds on every side, in axes units.
BOX_PAD = 0.008
# Fraction of the gap between two boxes spanned by the arrow drawn across it.
ARROW_SPAN = 0.66


def schematic_box(ax, x, y, width, height, edge, face, text, fontsize=8.2) -> None:
    """Rounded box with centred text, of visual extent (x-BOX_PAD, y-BOX_PAD) to (x+width+BOX_PAD, y+height+BOX_PAD)."""
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height,
        boxstyle=f"round,pad={BOX_PAD},rounding_size=0.020",
        linewidth=1.35, edgecolor=edge, facecolor=face,
    ))
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center",
            color=INK, fontsize=fontsize, linespacing=1.15)


def schematic_arrow(ax, start, end, color) -> None:
    """Arrow from ``start`` to ``end`` in axes units, the endpoints used as given."""
    ax.annotate("", xy=end, xytext=start, arrowprops={
        "arrowstyle": "-|>", "color": color, "lw": 1.35, "linestyle": "-",
        "shrinkA": 0, "shrinkB": 0, "mutation_scale": 9,
    })


def gap_arrow(ax, lo, hi, y, color) -> None:
    """Horizontal arrow at height ``y``, centred in the free gap between box edges ``lo`` and ``hi``."""
    inset = (hi - lo) * (1.0 - ARROW_SPAN) / 2.0
    schematic_arrow(ax, (lo + inset, y), (hi - inset, y), color)


def fig1_support_state_performance() -> None:
    """Fig. 1: constraint support, learned state and predictive performance (schematic, Sec. 2).

    The three layers the paper keeps apart, one row each.

        constraint layer   added unseen data -> constraint graph ->
                           identifiable support; a property of the training
                           set alone, monotone in the number of added inputs.
        model-state layer  data + optimization -> learned reduced state
                           rho_A, which differentiates (K_f up) under small
                           additions and recoheres on the target mode (K_f
                           down, w_star up) under larger ones, inside the
                           support the constraint layer makes available
                           (dashed hand-off).
        prediction layer   predictive performance, a many-to-one readout of
                           the state.

    The boxes are placed by an explicit grid of three columns and five rows,
    and every arrow sits in the free gap between two box edges, so no arrow
    enters a box.

    Reads   nothing
    Writes  figure/fig1_support_state_performance.pdf
    """
    teal = "#138A72"
    purple = "#5B4EA3"
    orange = "#D96822"
    blue = "#2F79BD"
    pale_teal = "#E5F1EF"
    pale_purple = "#F0EEF8"
    pale_orange = "#FAEEE7"
    pale_blue = "#E8F1F8"

    figure_style({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # Column geometry: x and width, and the visual edges that follow.
    c1_x, c1_w = 0.215, 0.145
    c2_x, c2_w = 0.425, 0.165
    c3_x, c3_w = 0.655, 0.285
    c1_r, c2_l, c2_r, c3_l = c1_x + c1_w + BOX_PAD, c2_x - BOX_PAD, c2_x + c2_w + BOX_PAD, c3_x - BOX_PAD

    # Row geometry.
    r1_y, r1_h = 0.715, 0.135          # constraint layer
    r2_y, r2_h = 0.415, 0.130          # model-state layer
    r3_y, r3_h = 0.050, 0.115          # prediction layer
    or_y, or_h = 0.500, 0.100          # outcome of small additions
    bl_y, bl_h = 0.335, 0.120          # outcome of larger additions
    r1_c, r2_c, r3_c = r1_y + r1_h / 2, r2_y + r2_h / 2, r3_y + r3_h / 2

    fig, ax = plt.subplots(figsize=(7.95, 3.865))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.990, "Constraint support, learned state, and predictive performance",
            ha="center", va="top", fontsize=12.0, fontweight="bold", color=INK)
    ax.text(0.60, 0.900, "monotone across added-data size",
            ha="center", va="center", fontsize=8.3, color=teal)

    label_x = 0.185
    ax.text(label_x, r1_c, "constraint layer", color=teal, fontsize=8.6,
            fontweight="bold", ha="right", va="center")
    ax.text(label_x, r2_c, "model-state layer", color=purple, fontsize=8.6,
            fontweight="bold", ha="right", va="center")
    ax.text(label_x, r3_c, "prediction layer", color=orange, fontsize=8.6,
            fontweight="bold", ha="right", va="center")

    # Constraint layer.
    schematic_box(ax, c1_x, r1_y, c1_w, r1_h, teal, pale_teal, "added unseen\ndata")
    schematic_box(ax, c2_x, r1_y, c2_w, r1_h, teal, pale_teal, "constraint graph")
    schematic_box(ax, c3_x, r1_y, c3_w, r1_h, teal, pale_teal, "identifiable support")
    gap_arrow(ax, c1_r, c2_l, r1_c, teal)
    gap_arrow(ax, c2_r, c3_l, r1_c, teal)

    # Model-state layer.
    schematic_box(ax, c1_x, r2_y, c1_w, r2_h, purple, pale_purple, "data +\noptimization")
    schematic_box(ax, c2_x, r2_y, c2_w, r2_h, purple, pale_purple,
                  "learned reduced\nstate $\\rho_A$")
    gap_arrow(ax, c1_r, c2_l, r2_c, purple)

    # The two outcomes, by size of the addition.
    schematic_box(ax, c3_x, or_y, c3_w, or_h, orange, pale_orange,
                  "small additions:\ndifferentiation, $K_f\\uparrow$", fontsize=7.8)
    schematic_box(ax, c3_x, bl_y, c3_w, bl_h, blue, pale_blue,
                  "larger additions:\ntarget-selective recoherence\n"
                  "$K_f\\downarrow,\\;w_\\star\\uparrow$", fontsize=7.7)

    inset = (c3_l - c2_r) * (1.0 - ARROW_SPAN) / 2.0
    schematic_arrow(ax, (c2_r + inset, 0.525), (c3_l - inset, or_y + or_h / 2), orange)
    schematic_arrow(ax, (c2_r + inset, 0.445), (c3_l - inset, bl_y + bl_h / 2), blue)

    # Hand-off of the support from the constraint layer to the outcome boxes.
    ax.plot([0.800, 0.800], [r1_y - BOX_PAD, or_y + or_h + BOX_PAD + 0.020],
            color=teal, lw=1.35, linestyle=(0, (2, 2)))
    ax.text(0.812, 0.668, "available support", color=teal, fontsize=8.3, va="center")

    # Prediction layer, read out of the state box above it.
    schematic_box(ax, c2_x, r3_y, c2_w, r3_h, orange, pale_orange,
                  "predictive\nperformance", fontsize=8.2)
    lo, hi = r3_y + r3_h + BOX_PAD, r2_y - BOX_PAD
    inset_v = (hi - lo) * (1.0 - ARROW_SPAN) / 2.0
    schematic_arrow(ax, (c2_x + c2_w / 2, hi - inset_v), (c2_x + c2_w / 2, lo + inset_v), orange)
    ax.text(c2_x + c2_w / 2 - 0.014, 0.290, "many-to-one readout",
            color=orange, fontsize=7.6, ha="right", va="center")

    save(fig, "fig1_support_state_performance", bbox_inches="tight", pad_inches=0.04)


# ==========================================================================
# Fig. 2: function state per output position before onset
# ==========================================================================

def fig2_output_position_dependence() -> None:
    """Fig. 2: function state and accuracy per output position from epoch 20 to onset (Sec. 4.1).

    Three maps over the output position j = 1, ..., 11 (horizontal) and the
    normalized training progress (vertical), 0 at epoch 20 and 1 at the
    generalization onset of each run, at 11 equally spaced stored
    checkpoints.  Every cell is the mean over the 24 N = 1,280 runs with a
    retreat event at equal progress.  Here the quantities belong to the
    single output j, not to the long-range mean:

        (A) K_f^(j) minus its value at epoch 20, so the bottom row is zero,
            on a diverging scale symmetric about zero;
        (B) w_star^(j), on [0, 1];
        (C) the accuracy at output j over all 2,048 inputs, on [0.5, 1],
            chance level being one half.

    The line in each panel is its frontier, the largest j that meets the
    criterion of the panel at that progress: j_diff, j_target and j_acc,
    with the criteria of ``analysis.output_position_dependence``.  Where no
    position qualifies, the table cell is empty and the line has a gap.

    Reads   results/output_position_dependence/per_seed.npz (positions,
            progress, and K_f, w_star, accuracy of shape (seed, progress,
            position)) and frontiers.csv (columns j_diff, j_target, j_acc)
    Writes  figure/fig2_output_position_dependence.pdf
    """
    source = RESULTS / "output_position_dependence"
    payload = read_npz(source / "per_seed.npz")
    rows = read_rows(source / "frontiers.csv")
    frontiers = tuple(
        np.asarray([float(row[name]) if row[name] else np.nan for row in rows])
        for name in ("j_diff", "j_target", "j_acc")
    )

    positions = payload["positions"]
    progress = payload["progress"]
    kf = payload["K_f"].mean(axis=0)          # mean over runs, shape (progress, position)
    delta = kf - kf[0]
    wstar = payload["w_star"].mean(axis=0)
    accuracy = payload["accuracy"].mean(axis=0)

    limit = float(np.nanmax(np.abs(delta)))
    panels = (
        (delta, "RdBu_r", -limit, limit,
         r"(A) Differentiation: $\Delta K_f^{(j)}$ from epoch 20",
         r"$\Delta K_f^{(j)}$", r"$j_{\rm diff}$"),
        (wstar, "viridis", 0.0, 1.0,
         r"(B) Target alignment: $w_\star^{(j)}$",
         r"$w_\star^{(j)}$", r"$j_{\rm target}$"),
        (accuracy, "viridis", 0.5, 1.0,
         "(C) Predictive performance: per-output full-space accuracy",
         "accuracy", r"$j_{\rm acc}$"),
    )

    figure_style({
        "font.size": 8.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
    })
    fig, axes = plt.subplots(3, 1, figsize=(7.46, 7.77), constrained_layout=True)

    # Progress increases upward, so row 0 (epoch 20) is the bottom row; the
    # extent reaches half a cell beyond the first and the last centre.
    half = 0.5 / (len(progress) - 1)
    extent = (
        positions[0] - 0.5, positions[-1] + 0.5,
        progress[0] - half, progress[-1] + half,
    )
    for axis, line, (values, cmap, low, high, title, label, frontier_label) in zip(
        axes, frontiers, panels
    ):
        image = axis.imshow(
            values, cmap=cmap, vmin=low, vmax=high, aspect="auto",
            origin="lower", extent=extent, interpolation="nearest",
        )
        bar = fig.colorbar(image, ax=axis, fraction=0.036, pad=0.02)
        bar.set_label(label, fontsize=7.5)
        bar.ax.tick_params(labelsize=6.8)
        axis.set_title(title, loc="left", fontsize=9.0, pad=4)
        axis.set_xticks(positions)
        axis.set_yticks(progress[::2], [f"{p:.1f}" for p in progress[::2]])
        axis.set_ylabel("normalized\ntraining progress", fontsize=8.0)
        axis.tick_params(labelsize=7.2)
        axis.plot(
            line, progress, color="white", lw=1.6,
            marker="o", ms=3.4, markerfacecolor="white",
            markeredgecolor=INK, markeredgewidth=0.8,
            label=frontier_label,
        )
        axis.legend(
            frameon=True, fontsize=6.8, loc="lower right",
            facecolor=INK, edgecolor="none", labelcolor="white", framealpha=0.85,
        )

    axes[-1].set_xlabel("parity span / output position $j$", fontsize=8.5)
    save(fig, "fig2_output_position_dependence")


# ==========================================================================
# Fig. 3: accuracy and function state along training at N = 64 and N = 1,280
# ==========================================================================

SMALL_LIMIT = 1000          # optimizer steps shown in panel A
SMALL_EXAMPLE = 503         # seed of the bold N = 64 run
FIRST_SAVED_EPOCH = 20      # the N = 1,280 networks are stored from this epoch on
LAST_EPOCH = 220            # last epoch of the N = 1,280 runs
TRAJECTORY_1 = 20           # seed of the bold N = 1,280 run "trajectory 1"; its first retreat is dashed
TRAJECTORY_2 = 14           # seed of the bold N = 1,280 run "trajectory 2"
# Colours of Fig. 3, with RUN_BLUE for the held-out accuracy and the faint
# N = 1,280 runs and INK for K_f - 1 in the inset.
TRAJECTORY_PURPLE = "#5B4EA3"   # trajectory 1
TRAJECTORY_ORANGE = "#D96822"   # training-set accuracy at N = 64; trajectory 2
STATE_GRAY = "#3f4448"          # K_f - 1 at N = 64
INSET_TEAL = "#138A72"          # held-out A_blk in the inset
RULE_GRAY = "#777777"           # reference lines
# Two text placements of Fig. 3, in points: the legend of panel A sits
# LEGEND_A_SHIFT to the left of the upper right corner of its anchor box, and
# the second line of the inset label at tau = 1 starts INSET_TAU1_LINE[0]
# from the tick with its baseline INSET_TAU1_LINE[1] from the lower edge of
# the inset.
LEGEND_A_SHIFT = 20.96875
INSET_TAU1_LINE = (-11.78125, -16.4)


def runs_by_seed(rows: list[dict[str, str]], x_key: str, y_keys: tuple[str, ...],
                 limit: Optional[int] = None) -> dict[int, list[dict]]:
    """Rows grouped by seed in order of first appearance, each group sorted by ``x_key``.

    A row keeps ``x_key`` as an integer and ``y_keys`` as floats; rows with
    ``x_key`` above ``limit`` are left out.
    """
    runs: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        x = int(row[x_key])
        if limit is None or x <= limit:
            runs[int(row["seed"])].append({x_key: x, **{key: float(row[key]) for key in y_keys}})
    for values in runs.values():
        values.sort(key=lambda row: row[x_key])
    return runs


def draw_runs(axis, runs: dict[int, list[dict]], x_key: str, y_key: str, colour: str,
              alpha: float, bold: Optional[int] = None) -> None:
    """One faint line per run (width 0.8, round caps) in the order of ``runs``; the run ``bold`` width 2.1 on top."""
    for seed, rows in runs.items():
        heavy = seed == bold
        axis.plot([row[x_key] for row in rows], [row[y_key] for row in rows], color=colour,
                  lw=2.1 if heavy else 0.8, alpha=1.0 if heavy else alpha,
                  zorder=3 if heavy else 1, solid_capstyle="round")


def onset_aligned_inset(axis, means: dict[str, np.ndarray], accuracy: np.ndarray, n_seeds: int) -> None:
    """The inset of panel B: K_f - 1 and held-out A_blk against tau, mean over the 24 runs with interquartile band.

    tau runs from 0 at epoch 20 to 1 at the generalization onset of each run;
    at tau = 0, 0.1, ..., 1 each run is read at its stored epoch
    round(20 + tau (onset - 20)), so runs that reach onset at different epochs
    can be averaged.  ``means`` is ``onset_aligned_means.csv``, whose K_f - 1
    columns give the black line and band (left axis); ``accuracy`` (runs x 11)
    is the held-out A_blk at the same epochs, whose mean and quartiles give
    the teal line and band (right axis).  Both vertical axes scale to their
    data.  ``n_seeds`` is the number of runs, given in the title.
    """
    tau = means["tau"]
    inset = axis.inset_axes([0.055, 0.52, 0.58, 0.44])
    inset.plot(tau, means["K_f_excess_mean"], color=INK, marker="o", ms=2.6, lw=1.2, label=r"$K_f-1$")
    inset.fill_between(tau, means["K_f_excess_q25"], means["K_f_excess_q75"],
                       color=INK, alpha=0.14, lw=0)
    twin = inset.twinx()
    twin.plot(tau, accuracy.mean(axis=0), color=INSET_TEAL, marker="s", ms=2.6, lw=1.2,
              label=r"$A_{\rm blk}$")
    twin.fill_between(tau, np.quantile(accuracy, 0.25, axis=0), np.quantile(accuracy, 0.75, axis=0),
                      color=INSET_TEAL, alpha=0.16, lw=0)
    inset.set_title(f"event-aligned mean ({n_seeds} seeds)", fontsize=6.2, pad=2.5)
    inset.set_ylabel(r"$K_f-1$", fontsize=5.8, labelpad=1)
    twin.set_ylabel(r"held-out $A_{\rm blk}$", fontsize=5.8, color=INSET_TEAL, labelpad=1)
    twin.tick_params(colors=INSET_TEAL, labelsize=5.4)
    inset.tick_params(labelsize=5.4)
    inset.set_xticks([0, 0.5, 1])
    inset.set_xticklabels([r"$\tau=0$" "\n" "epoch 20", r"$\tau=0.5$", r"$\tau=1$"], fontsize=5.4)
    # The second line of the label at tau = 1, at its place INSET_TAU1_LINE.
    inset.annotate("gen. onset", xy=(1, 0), xycoords=inset.get_xaxis_transform(),
                   xytext=INSET_TAU1_LINE, textcoords="offset points",
                   ha="left", va="baseline", fontsize=5.4)
    inset.set_xlim(-0.03, 1.03)
    inset.grid(alpha=0.12)
    handles = inset.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    inset.legend(handles, [r"$K_f-1$", r"$A_{\rm blk}$"], frameon=False, fontsize=5.6,
                 loc="upper left", ncol=2, handlelength=1.4, columnspacing=1.0)
    inset.spines["top"].set_visible(False)
    twin.spines["top"].set_visible(False)
    twin.spines["right"].set_visible(True)


def fig3_training_trajectories() -> None:
    """Fig. 3: accuracy and K_f - 1 along training at N = 64 and N = 1,280 (Secs. 4.2, 4.3).

    Top row block accuracy, bottom row the excess effective Schmidt rank
    K_f - 1, which is zero for a single conditional mode.  The two panels of
    a row share their vertical scale.  Faint lines are single runs (width
    0.8), bold lines (width 2.1) the highlighted runs, each in one colour in
    both rows.

    (A) N = 64, the 24 runs of the small-data protocol (seeds 500-523) at
        their saved steps up to 1,000 (every 10 steps to 200, every 50 to
        1,000; one optimizer step per epoch), with the run of seed 503 bold.
        Above, the held-out block accuracy A_blk(S) (blue), S the 1,984
        strings outside the training set, and the training-set block
        accuracy (orange); below, K_f - 1 (gray).  The dashed line is the
        interpolation checkpoint of the bold run, its first saved step with
        training-set block accuracy 1 (step 500).
    (B) N = 1,280, the 24 runs with a retreat event, from epoch 20 to epoch
        220, with seed 20 (trajectory 1, purple) and seed 14 (trajectory 2,
        orange) bold.  Above, the held-out block accuracy A_blk(S) of every
        stored network, S the 1,085 strings outside the training set; the
        dotted line is A_blk = 1.  Below, K_f - 1:
        faint from epoch 20 to onset for every run, bold at every stored
        epoch for seeds 20 and 14.  No network is stored before epoch 20,
        which the shaded margin marks.  The dashed line is the first retreat
        of trajectory 1, epoch 141 (``retreat_epoch`` of its first event).

    The inset of panel B is the onset-aligned mean of K_f - 1 and of the
    held-out A_blk over the 24 runs (``onset_aligned_inset``).

    Reads   results/training_trajectories/small_data.csv (seed, saved_step,
                A_block, training_block_accuracy, K_f_excess),
                large_data_to_onset.csv, large_data_seed20.csv,
                large_data_seed14.csv (seed, epoch, K_f_excess)
            results/state_vs_accuracy/heldout_accuracy.csv (seed, epoch,
                A_block_heldout), onset_aligned_means.csv
            results/retreat_events/events.csv (acquisition_epoch and
                retreat_epoch of the first event of each run)
    Writes  figure/fig3_training_trajectories.pdf
    """
    tables = RESULTS / "training_trajectories"
    small = runs_by_seed(read_rows(tables / "small_data.csv"), "saved_step",
                         ("A_block", "training_block_accuracy", "K_f_excess"), SMALL_LIMIT)
    to_onset = runs_by_seed(read_rows(tables / "large_data_to_onset.csv"), "epoch", ("K_f_excess",))
    bold_state = {
        seed: runs_by_seed(read_rows(tables / f"large_data_seed{seed}.csv"), "epoch", ("K_f_excess",))[seed]
        for seed in (TRAJECTORY_1, TRAJECTORY_2)
    }
    heldout = {
        (int(row["seed"]), int(row["epoch"])): float(row["A_block_heldout"])
        for row in read_rows(RESULTS / "state_vs_accuracy" / "heldout_accuracy.csv")
    }
    first_event: dict[int, dict[str, str]] = {}
    for row in read_rows(RESULTS / "retreat_events" / "events.csv"):
        first_event.setdefault(int(row["seed"]), row)
    seeds = sorted(first_event)
    onset_rows = read_rows(RESULTS / "state_vs_accuracy" / "onset_aligned_means.csv")
    # Every column at full precision; n_seeds is the same in every row.
    onset_means = {
        key: np.asarray([float(row[key]) for row in onset_rows])
        for key in onset_rows[0] if key != "n_seeds"
    }
    n_seeds = int(onset_rows[0]["n_seeds"])
    # Held-out A_blk of every run at the epochs of the inset, one row per run.
    inset_accuracy = []
    for seed in seeds:
        onset = int(first_event[seed]["acquisition_epoch"])
        wanted = [round(FIRST_SAVED_EPOCH + tau * (onset - FIRST_SAVED_EPOCH)) for tau in onset_means["tau"]]
        inset_accuracy.append([heldout[(seed, int(epoch))] for epoch in wanted])

    figure_style({
        "font.size": 8.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.35), sharey="row", constrained_layout=True)
    (small_acc, large_acc), (small_rank, large_rank) = axes

    # --- A: the N = 64 protocol on its optimizer-step axis -----------------
    draw_runs(small_acc, small, "saved_step", "A_block", RUN_BLUE, 0.22, SMALL_EXAMPLE)
    draw_runs(small_acc, small, "saved_step", "training_block_accuracy", TRAJECTORY_ORANGE, 0.18,
                SMALL_EXAMPLE)
    # Legend entries at full strength, with no data.
    small_acc.plot([], [], color=RUN_BLUE, lw=2.1, label="held-out block accuracy")
    small_acc.plot([], [], color=TRAJECTORY_ORANGE, lw=2.1, label="training-set block accuracy")
    interpolation = next(row["saved_step"] for row in small[SMALL_EXAMPLE]
                         if row["training_block_accuracy"] == 1.0)
    for axis in (small_acc, small_rank):
        axis.axvline(interpolation, color=RULE_GRAY, ls="--", lw=0.85)
    small_acc.set_title(r"(A) Small data: $N=64$")
    small_acc.legend(frameon=False, fontsize=6.8, loc="upper right", bbox_to_anchor=(1.0, 0.95),
                     bbox_transform=small_acc.transAxes
                     + ScaledTranslation(-LEGEND_A_SHIFT / 72, 0, fig.dpi_scale_trans))
    draw_runs(small_rank, small, "saved_step", "K_f_excess", STATE_GRAY, 0.24, SMALL_EXAMPLE)
    small_rank.text(0.98, 0.95, "highlighted path at interpolation", transform=small_rank.transAxes,
                    ha="right", va="top", fontsize=6.8, color="#444444")
    small_rank.set_xlabel("epoch (= optimizer step)")
    for axis in (small_acc, small_rank):
        axis.set_xlim(0, SMALL_LIMIT)

    # --- B: the N = 1,280 protocol on its epoch axis -----------------------
    epochs = list(range(FIRST_SAVED_EPOCH, LAST_EPOCH + 1))
    for seed in seeds:
        if seed not in (TRAJECTORY_1, TRAJECTORY_2):
            large_acc.plot(epochs, [heldout[(seed, epoch)] for epoch in epochs],
                           color=RUN_BLUE, lw=0.8, alpha=0.16, zorder=1)
    for seed, colour, label in ((TRAJECTORY_1, TRAJECTORY_PURPLE, "trajectory 1"),
                                (TRAJECTORY_2, TRAJECTORY_ORANGE, "trajectory 2")):
        large_acc.plot(epochs, [heldout[(seed, epoch)] for epoch in epochs], color=colour,
                       lw=2.1, zorder=3, label=label, solid_capstyle="round")
    large_acc.axhline(1.0, color=RULE_GRAY, ls=":", lw=0.7)
    retreat = int(first_event[TRAJECTORY_1]["retreat_epoch"])
    for axis in (large_acc, large_rank):
        axis.axvline(retreat, color=RULE_GRAY, ls="--", lw=0.85)
    large_acc.set_title(r"(B) Large data: $N=1280$")
    large_acc.legend(frameon=False, fontsize=6.8, loc="upper left", title="held-out block accuracy",
                     title_fontsize=6.8, alignment="left")
    draw_runs(large_rank, to_onset, "epoch", "K_f_excess", RUN_BLUE, 0.16)
    for seed, colour in ((TRAJECTORY_1, TRAJECTORY_PURPLE), (TRAJECTORY_2, TRAJECTORY_ORANGE)):
        large_rank.plot([row["epoch"] for row in bold_state[seed]],
                        [row["K_f_excess"] for row in bold_state[seed]], color=colour, lw=2.1, zorder=3)
    large_rank.axvspan(0, FIRST_SAVED_EPOCH - 0.5, color="#888888", alpha=0.09, lw=0)
    large_rank.set_xlabel("epoch")
    for axis in (large_acc, large_rank):
        axis.set_xlim(0, LAST_EPOCH)

    # --- shared framing -----------------------------------------------------
    for axis in (small_acc, large_acc):
        axis.set_ylim(-0.03, 1.04)
        axis.grid(alpha=0.14)
        axis.tick_params(axis="x", labelbottom=False)
    # sharey="row" gives each row one scale, so the label belongs to the left
    # column alone.
    small_acc.set_ylabel("block accuracy")
    for axis in (small_rank, large_rank):
        axis.set_ylim(-0.22, 9.3)
        axis.axhline(0, color=RULE_GRAY, ls="--", lw=0.7)
        axis.grid(alpha=0.14)
    small_rank.set_ylabel(r"excess effective Schmidt rank $K_f-1$")
    onset_aligned_inset(large_rank, onset_means, np.asarray(inset_accuracy), n_seeds)

    save(fig, "fig3_training_trajectories")


# ==========================================================================
# Fig. 4: function state against block accuracy
# ==========================================================================

ACCURACY_BAND = (0.18, 0.32)   # the accuracy band shaded in Fig. 4


def fig4_state_vs_accuracy() -> None:
    """Fig. 4: function state against block accuracy at every stored checkpoint (Sec. 4.4).

    One point per stored checkpoint of the 24 N = 1,280 runs with a retreat
    event, epochs 20 to 220, against the block accuracy A_blk over all 2,048
    inputs:

        (A) K_f - 1 on a logarithmic axis; the checkpoints at the exact
            single-mode floor K_f - 1 = 0 cannot be shown there and are left
            out of this panel only;
        (B) Gamma_1;
        (C) w_star, with the line w_star = A_blk.

    Blue marks the checkpoints before the generalization onset of their run,
    orange those at or after it.  The grey band is the accuracy band
    0.18 <= A_blk <= 0.32, inside which the states spread widely at one
    accuracy.  The points are rasterized inside the PDF.

    Reads   results/state_vs_accuracy/checkpoints.npz (A_block, after_onset,
            K_f, Gamma_1, w_star, one entry per checkpoint)
    Writes  figure/fig4_state_vs_accuracy.pdf
    """
    payload = read_npz(RESULTS / "state_vs_accuracy" / "checkpoints.npz")
    accuracy = payload["A_block"]
    after = payload["after_onset"].astype(bool)
    panels = (
        (payload["K_f"] - 1.0, True,
         r"excess effective Schmidt rank $K_f-1$", "A"),
        (payload["Gamma_1"], False, r"crossing-gauge $\Gamma_1$", "B"),
        (payload["w_star"], False, r"target-mode population $w_\star$", "C"),
    )

    figure_style({
        "font.size": 7.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
    })
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.9), constrained_layout=True)

    for axis, (values, log_scale, ylabel, letter) in zip(axes, panels):
        axis.axvspan(*ACCURACY_BAND, color="#EDEDED", linewidth=0, zorder=0)
        for mask, color, label in (
            (~after, RUN_BLUE, "before onset"),
            (after, RUN_ORANGE, "after onset"),
        ):
            x, y = accuracy[mask], values[mask]
            if log_scale:
                # A logarithmic axis cannot show the exact single-mode floor,
                # where K_f - 1 is identically zero.
                keep = y > 0.0
                x, y = x[keep], y[keep]
            axis.scatter(
                x, y, s=3.2, c=color, alpha=0.45, linewidths=0,
                rasterized=True, label=label,
            )
        if log_scale:
            axis.set_yscale("log")
        axis.set_xlabel(r"full-space block accuracy $A_{\rm blk}$", fontsize=7.8)
        axis.set_ylabel(ylabel, fontsize=7.8)
        axis.set_xlim(-0.02, 1.02)
        axis.grid(alpha=0.12)
        axis.tick_params(labelsize=7.0)
        axis.text(
            0.02, 1.02, f"({letter})", transform=axis.transAxes,
            fontsize=8.6, fontweight="bold", va="bottom", color=INK,
        )

    axes[2].plot([0, 1], [0, 1], color=RETREAT_RED, ls="--", lw=0.9, zorder=3)
    axes[2].text(
        0.62, 0.50, r"$w_\star=A_{\rm blk}$", color=RETREAT_RED, fontsize=6.6,
        rotation=38, rotation_mode="anchor", va="bottom", ha="left",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles, labels, frameon=False, fontsize=6.6, loc="lower left",
        markerscale=2.6, handletextpad=0.4,
    )
    save(fig, "fig4_state_vs_accuracy")


# ==========================================================================
# Fig. 5: three branches from one starting model at the largest addition
# ==========================================================================

LARGEST_ADDITION = ADDITION_SIZES[-1]  # 1,216 added inputs
# Colours of Fig. 5 that no other figure uses.
FIG5_GRAY = "#707985"
FIG5_LIGHT = "#D6DCE2"
FIG5_PALE_BLUE = "#E7F0F7"
FIG5_CONTROL = "#4C4C4C"
# Branch name in the tables, legend entry, colour; drawn in this order.
BRANCH_STYLES = (
    ("distinct", "distinct arm: +1,216 new data points", TEAL),
    ("duplicate", "duplicate arm: +1,216 repeats", CORAL),
    ("control", "control: original $N=64$", FIG5_CONTROL),
)
# Branch and update of a spectrum, legend entry, colour, line style, marker.
SPECTRUM_STYLES = (
    ("distinct", 0, "common start (update 0)", INK, "--", "o"),
    ("distinct", 25, "distinct early (update 25)", ORANGE, "-", "s"),
    ("distinct", 800, "distinct endpoint (update 800)", TEAL, "-", "o"),
    ("duplicate", 800, "duplicate endpoint (update 800)", CORAL, "-", "o"),
)
# Field of paired_contrasts.csv, axis label, colour.
CONTRAST_STYLES = (
    ("K_f", r"$\Delta K_f=K_f^{dist}-K_f^{dup}$", TEAL),
    ("Gamma_1", r"$\Gamma_1^{dist}-\Gamma_1^{dup}$", PURPLE),
    ("A_block", r"$A_{blk}^{dist}-A_{blk}^{dup}$", ORANGE),
)


def read_largest_addition_branches() -> tuple[dict, dict, dict, dict]:
    """The four tables of ``analysis.largest_addition_branches`` keyed for drawing.

    Returns the per-starting-model traces of K_f - 1 by branch (seed ->
    (updates, values)), the means and quartiles by (branch, field), the mean
    spectra by (branch, update), and the paired contrasts by field; every
    entry carries its own list of updates, since the control branch is stored
    at 7 updates and the others at 11.
    """
    tables = RESULTS / "largest_addition_branches"
    traces: dict[str, dict[int, tuple[list[int], list[float]]]] = {branch: {} for branch, _legend, _colour in BRANCH_STYLES}
    for row in read_rows(tables / "branch_traces.csv"):
        steps, values = traces[row["arm"]].setdefault(int(row["seed"]), ([], []))
        steps.append(int(row["step"]))
        values.append(float(row["K_f_exc"]))

    means: dict[tuple[str, str], dict[str, list[float]]] = {}
    for row in read_rows(tables / "branch_means.csv"):
        entry = means.setdefault((row["arm"], row["field"]), {"step": [], "mean": [], "q25": [], "q75": []})
        entry["step"].append(int(row["step"]))
        for key in ("mean", "q25", "q75"):
            entry[key].append(float(row[key]))

    spectra: dict[tuple[str, int], list[float]] = {}
    for row in read_rows(tables / "spectra.csv"):
        spectra.setdefault((row["arm"], int(row["step"])), []).append(float(row["eigenvalue"]))

    contrasts: dict[str, dict[str, list[float]]] = {}
    for row in read_rows(tables / "paired_contrasts.csv"):
        entry = contrasts.setdefault(row["field"], {"step": [], "mean": [], "q25": [], "q75": []})
        entry["step"].append(int(row["step"]))
        for key in ("mean", "q25", "q75"):
            entry[key].append(float(row[key]))
    return traces, means, spectra, contrasts


def update_axis(axis, steps: np.ndarray) -> None:
    """Symmetric-log axis of continuation updates, linear below 25, one tick per saved update."""
    axis.set_xscale("symlog", linthresh=25)
    axis.set_xticks(steps)
    axis.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())


def branch_schematic(axis) -> None:
    """The branch protocol, inset into panel D and drawn from its constants alone.

    The branch-point model fits the 64 base inputs; it continues on
    N = 1,280 after 1,216 unseen inputs are added; the continued model is
    evaluated on the added subset (1,216) and on the held-out subset (768).
    """
    inset = axis.inset_axes([0.045, 0.37, 0.42, 0.56])
    inset.set_xlim(0, 1)
    inset.set_ylim(0, 1)
    inset.axis("off")
    inset.add_patch(Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=FIG5_LIGHT, linewidth=0.7, zorder=0))
    box = dict(boxstyle="round,pad=0.02,rounding_size=0.025", facecolor=FIG5_PALE_BLUE, edgecolor=BLUE, linewidth=0.8)
    arrow = dict(arrowstyle="-|>", mutation_scale=7, color=FIG5_GRAY, linewidth=0.8)
    inset.add_patch(FancyBboxPatch((0.12, 0.76), 0.76, 0.18, **box))
    inset.text(0.50, 0.85, "branch-point model\nfits original $N=64$", ha="center", va="center", fontsize=6.5)
    inset.add_patch(FancyArrowPatch((0.50, 0.75), (0.50, 0.64), **arrow))
    inset.text(0.55, 0.695, "+1,216 unseen", ha="left", va="center", fontsize=5.9, color=FIG5_GRAY)
    inset.add_patch(FancyBboxPatch((0.12, 0.45), 0.76, 0.18, **box))
    inset.text(0.50, 0.54, "continue training\non $N=1{,}280$", ha="center", va="center", fontsize=6.5)
    inset.add_patch(FancyArrowPatch((0.50, 0.44), (0.50, 0.33), **arrow))
    inset.add_patch(FancyBboxPatch((0.08, 0.05), 0.84, 0.26, boxstyle="round,pad=0.02,rounding_size=0.025",
                                   facecolor="white", edgecolor=FIG5_GRAY, linewidth=0.8))
    for y, label, colour, size in (
        (0.255, "same model evaluated on", INK, 5.9),
        (0.175, "added training subset (1,216)", BLUE, 5.8),
        (0.095, "held-out subset (768)", ORANGE, 6.2),
    ):
        inset.text(0.50, y, label, ha="center", va="center", fontsize=size, color=colour)


def fig5_starting_model_branches() -> None:
    """Fig. 5: three branches from one starting model at the largest addition (Sec. 5.1).

    The branches of a starting model share its weights, the optimizer protocol,
    the number of updates and the minibatch stream, and differ only in the
    data added to the 64 base inputs: 1,216 distinct inputs (distinct),
    1,216 repeated base inputs (duplicate), or nothing (control).  The
    horizontal axes of panels A, C and D are the saved updates on a
    symmetric-log axis, linear below 25: 11 updates 0, ..., 800 for the
    distinct and duplicate branches, the 7 updates 0, 25, ..., 800 of the
    records for the control branch.

    (A) K_f - 1 along the updates: one faint trace per starting model, and the
        mean over the 25 starting models, branch by branch.
    (B) Mean spectra of rho_A (averaged over the long-range positions and the
        starting models) against eigenvalue rank, on a logarithmic axis, at
        four points: the common branch point (update 0), the distinct branch
        at update 25, and the distinct and duplicate branches at update 800.
    (C) Distinct-minus-duplicate contrasts formed within a starting model and
        then averaged, one row each for K_f, Gamma_1 and A_block, with the
        interquartile range over starting models shaded.
    (D) Block accuracy of the distinct branch on two disjoint subsets of the
        2,048 inputs, the 1,216 it added and the 768 it never trains on,
        with interquartile bands, and the protocol schematic
        (``branch_schematic``).

    Reads   results/largest_addition_branches/branch_traces.csv, branch_means.csv,
            spectra.csv, paired_contrasts.csv (``read_largest_addition_branches``)
    Writes  figure/fig5_starting_model_branches.pdf
    """
    traces, means, spectra, contrasts = read_largest_addition_branches()
    figure_style({
        "font.family": "DejaVu Sans",
        "font.size": 8.3,
        "axes.titlesize": 9.3,
        "axes.labelsize": 8.4,
        "legend.fontsize": 7.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(10.2, 7.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, hspace=0.18, wspace=0.18)

    # --- A: excess effective Schmidt rank along the continuation ----------
    ax = fig.add_subplot(grid[0, 0])
    for branch, legend, colour in BRANCH_STYLES:
        for steps, values in traces[branch].values():
            ax.plot(np.asarray(steps), np.asarray(values), color=colour, alpha=0.10, lw=0.55, marker="o", ms=1.2)
        entry = means[(branch, "K_f_exc")]
        ax.plot(np.asarray(entry["step"]), np.asarray(entry["mean"]), color=colour, lw=2.0,
                marker="o", ms=3.4, label=legend)
    steps = np.asarray(means[("distinct", "K_f_exc")]["step"])
    update_axis(ax, steps)
    ax.tick_params(axis="x", labelsize=6.4)
    ax.set_xlabel("continuation updates")
    ax.set_ylabel(r"excess effective Schmidt rank $K_f-1$")
    ax.set_title("(A) Matched continuation trajectories")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(alpha=0.15)

    # --- B: spectra at four checkpoints ------------------------------------
    ax = fig.add_subplot(grid[0, 1])
    ranks = np.arange(1, 33)
    for branch, step, label, colour, linestyle, marker in SPECTRUM_STYLES:
        is_common_start = step == 0
        ax.plot(ranks, np.asarray(spectra[(branch, step)]), color=colour,
                lw=2.3 if is_common_start else 1.8, ls=linestyle,
                marker=marker, ms=3.5 if is_common_start else 2.5,
                markerfacecolor="white" if is_common_start else colour,
                markeredgewidth=0.9 if is_common_start else 0.5,
                zorder=5 if is_common_start else 2, label=label)
    ax.set_yscale("log")
    ax.set_xlim(1, 32)
    ax.set_xlabel("conditional eigenvalue rank")
    ax.set_ylabel(r"mean eigenvalue $\lambda_r$")
    ax.set_title("(B) Full conditional spectrum")
    ax.legend(frameon=False, fontsize=6.5)
    ax.grid(alpha=0.15)

    # --- C: within-starting-model distinct minus duplicate, one row per quantity
    contrast_grid = grid[1, 0].subgridspec(3, 1, hspace=0.08)
    for index, (field, ylabel, colour) in enumerate(CONTRAST_STYLES):
        axis = fig.add_subplot(contrast_grid[index, 0])
        entry = contrasts[field]
        axis.axhline(0, color="#777777", ls=":", lw=0.7)
        axis.plot(steps, np.asarray(entry["mean"]), color=colour, lw=1.8, marker="o", ms=3.0)
        axis.fill_between(steps, np.asarray(entry["q25"]), np.asarray(entry["q75"]), color=colour,
                          alpha=0.14, linewidth=0)
        update_axis(axis, steps)
        axis.set_ylabel(ylabel, fontsize=7)
        axis.grid(alpha=0.12)
        if index == 0:
            axis.set_title("(C) Within-seed distinct-vs-duplicate contrasts", pad=2)
        if index < 2:
            axis.tick_params(axis="x", labelbottom=False)
        else:
            axis.tick_params(axis="x", labelsize=6.4)
            axis.set_xlabel("continuation updates")

    # --- D: the distinct branch on two disjoint subsets --------------------
    ax = fig.add_subplot(grid[1, 1])
    for field, label, colour in (
        ("A_block_added", f"added training subset ({LARGEST_ADDITION:,})", BLUE),
        ("A_block_untrained", "held-out subset (768)", ORANGE),
    ):
        entry = means[("distinct", field)]
        ax.plot(steps, np.asarray(entry["mean"]), color=colour, lw=2.0, marker="o", ms=3.4, label=label)
        ax.fill_between(steps, np.asarray(entry["q25"]), np.asarray(entry["q75"]), color=colour,
                        alpha=0.13, linewidth=0)
    update_axis(ax, steps)
    ax.tick_params(axis="x", labelsize=6.4)
    ax.set_ylim(-0.02, 0.65)
    ax.set_xlabel("continuation updates")
    ax.set_ylabel("block accuracy")
    ax.set_title("(D) Same model, two evaluation subsets")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.grid(alpha=0.15)
    branch_schematic(ax)

    save(fig, "fig5_starting_model_branches")


# ==========================================================================
# Fig. 6: endpoint contrasts across added-data size
# ==========================================================================

SMALL_ADDITIONS = (16, 32, 64)     # the three small additions, shaded
FIG6_PALE_ORANGE = "#F8ECE4"


def fig6_added_data_size() -> None:
    """Fig. 6: distinct-minus-duplicate contrasts at update 800 across added-data size (Sec. 5.2).

    At each of the six additions (+16 to +1,216), the contrast of the
    distinct and the duplicate branch of one starting model at update 800,
    averaged over the 25 starting models, with the central 95 per cent
    bootstrap interval over starting models as bars:

        (A) Delta K_f = -C_K: positive at +16, +32, +64 (differentiation),
            negative from +192 on (recoherence);
        (B) Delta w_star, positive throughout.

    The three small additions are shaded in both panels, and the additions
    are plotted at equal spacing in their order.

    Reads   results/branch_endpoint_state/contrasts.csv (``dose``; C_K_mean,
            _ci_low, _ci_high; delta_w_star_mean, _ci_low, _ci_high)
    Writes  figure/fig6_added_data_size.pdf
    """
    data = read_frame(RESULTS / "branch_endpoint_state" / "contrasts.csv",
                      float_precision="round_trip").sort_values("dose")
    x = np.arange(len(data))
    labels = [f"+{int(value)}" for value in data.dose]
    delta_k = -data.C_K_mean.to_numpy(float)
    k_low = -data.C_K_ci_high.to_numpy(float)
    k_high = -data.C_K_ci_low.to_numpy(float)
    delta_w = data.delta_w_star_mean.to_numpy(float)
    w_low = data.delta_w_star_ci_low.to_numpy(float)
    w_high = data.delta_w_star_ci_high.to_numpy(float)
    shaded = [i for i, addition in enumerate(data.dose) if addition in SMALL_ADDITIONS]

    figure_style({
        "font.family": "DejaVu Sans",
        "font.size": 8.3,
        "axes.titlesize": 9.3,
        "axes.labelsize": 8.4,
        "legend.fontsize": 7.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.65))
    for ax in axes:
        ax.axhline(0, color=INK, lw=0.9)
        ax.axvspan(min(shaded) - 0.45, max(shaded) + 0.45, color=FIG6_PALE_ORANGE, alpha=0.72, lw=0)
        ax.set_xticks(x, labels)
        ax.set_xlabel("added unseen data points")
        ax.grid(axis="y", alpha=0.18, linewidth=0.6)

    axes[0].errorbar(x, delta_k, yerr=[delta_k - k_low, k_high - delta_k], color=PURPLE,
                     marker="o", markersize=5, lw=1.8, capsize=2.6)
    axes[0].set_ylabel(r"$\Delta K_f=K_f^{\rm dist}-K_f^{\rm dup}$")
    axes[0].set_title("added-data-size sign change")
    # The two regimes of panel A, named where each one occurs.
    axes[0].text(1.0, max(delta_k) * 0.82, "broader spectrum", ha="center", color=ORANGE, fontsize=7.5)
    axes[0].text(4.2, min(delta_k) * 0.82, "spectral reconcentration", ha="center", color=BLUE, fontsize=7.5)

    axes[1].errorbar(x, delta_w, yerr=[delta_w - w_low, w_high - delta_w], color=TEAL,
                     marker="o", markersize=5, lw=1.8, capsize=2.6)
    axes[1].set_ylabel(r"$\Delta w_\star=w_\star^{\rm dist}-w_\star^{\rm dup}$")
    axes[1].set_title("target-mode population")
    for ax, letter in zip(axes, "AB"):
        ax.text(-0.13, 1.09, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
    fig.suptitle("Added distinct data produce a nonmonotonic state response",
                 y=1.02, fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig6_added_data_size")


# ==========================================================================
# Fig. 7: percolation of the constraint graph against the learned state
# ==========================================================================

# The grid of added-input numbers of ``analysis.constraint_percolation``.
GRAPH_GRID = (0, 16, 32, 64, 96, 128, 160, 192, 256, 384, 576, 768, 1024, 1216)


def set_addition_axis(ax) -> None:
    """Symmetric-log axis of added inputs, linear below 8, shared by panels B, C and D."""
    ax.set_xscale("symlog", linthresh=8)
    ax.set_xlim(-1, 1400)
    ax.grid(alpha=0.16, lw=0.6)


def fig7_constraint_percolation_vs_state() -> None:
    """Fig. 7: monotone percolation of the constraint graph against the nonmonotone learned state (Sec. 6).

    The graph is that of the distinct branches of the branching study: the
    64 base inputs plus the first n inputs of the one random ordering of the
    1,984 inputs outside the base set (the data selection).

    (A) Components of the graph at n = 0, 64, 192, 576.  Each bar is split
        into the components of the whole graph, largest first and dark,
        with widths equal to the fraction of the 32 left substrings each
        holds; segments of at least 0.055 (two or more substrings) are
        labelled with the number.  The text at the right gives the number
        c of components among all 96 vertices and d_free = c - 1.
    (B) Probability over 10,000 random orderings that the left vertices are
        connected and that the whole graph is connected, against n.
    (C) f_L over the random orderings, mean and 5 to 95 per cent band, and
        f_L of the data selection.
    (D) K_perc of the data selection, the K_f that follows from the graph
        alone, against the means over the starting models of K_f at
        update 800 in the distinct and in the duplicate branches.

    Panels B to D share a symmetric-log axis in n (``set_addition_axis``).

    Reads   results/constraint_percolation/selection.csv (by ``dose``: components,
                d_free, f_left, K_perc_primary), selection_components.csv
                (``dose``, left_substrings, largest first),
                random_orderings_summary.csv (by ``dose``: p_left_connected,
                p_connected, f_left_mean, f_left_q05, f_left_q95)
            results/branch_endpoint_state/contrasts.csv (by ``dose``: K_f_dist_mean,
                K_f_dup_mean)
    Writes  figure/fig7_constraint_percolation_vs_state.pdf
    """
    figure_style({
        "font.family": "DejaVu Sans",
        "font.size": 8.3,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.2,
        "legend.fontsize": 6.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })
    graph_tables = RESULTS / "constraint_percolation"
    primary = read_frame(graph_tables / "selection.csv").set_index("dose")
    components = read_frame(graph_tables / "selection_components.csv")
    ensemble = read_frame(graph_tables / "random_orderings_summary.csv").set_index("dose")
    learned = read_frame(RESULTS / "branch_endpoint_state" / "contrasts.csv").set_index("dose")
    pale_teal = "#D7ECE7"
    sky = "#0072B2"
    vermilion = "#D55E00"

    fig, axes = plt.subplots(2, 2, figsize=(9.72, 7.16), constrained_layout=True)

    ax = axes[0, 0]
    shown_additions = (0, 64, 192, 576)
    y = np.arange(len(shown_additions))
    for row, addition in enumerate(shown_additions):
        left = 0.0
        sizes = components[components.dose == addition].left_substrings.tolist()
        for index, size in enumerate(sizes):
            width = size / N_LEFT
            color = TEAL if index == 0 else pale_teal
            ax.barh(row, width, left=left, height=0.48, color=color, edgecolor="white", linewidth=0.45)
            if width >= 0.055:
                ax.text(left + width / 2, row, str(size), ha="center", va="center", fontsize=7.0)
            left += width
        state = primary.loc[addition]
        ax.text(
            1.035,
            row,
            rf"$c={int(state.components)},\ d_{{\rm free}}={int(state.d_free)}$",
            ha="left",
            va="center",
            fontsize=7.0,
        )
    ax.set_yticks(y, ["+0", "+64", "+192", "+576"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.23)
    ax.set_xlabel("left-context component mass (labels give sizes)")
    ax.set_title("Representative constraint components")
    ax.grid(axis="x", alpha=0.15, lw=0.6)

    graph = ensemble.loc[list(GRAPH_GRID)]
    ax = axes[0, 1]
    ax.plot(GRAPH_GRID, graph.p_left_connected, color=TEAL, marker="o", ms=3.4, lw=1.8, label="left graph connected")
    ax.plot(GRAPH_GRID, graph.p_connected, color=sky, marker="s", ms=3.1, lw=1.7, label="full graph connected")
    ax.set_ylim(-0.03, 1.04)
    ax.set_xlabel("added unseen data points")
    ax.set_ylabel("ensemble connectivity probability")
    ax.set_title("Connectivity rises monotonically")
    set_addition_axis(ax)
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1, 0]
    ax.fill_between(GRAPH_GRID, graph.f_left_q05, graph.f_left_q95, color=pale_teal, label="ensemble 5--95%")
    ax.plot(GRAPH_GRID, graph.f_left_mean, color=TEAL, lw=1.9, label="ensemble mean")
    ax.plot(GRAPH_GRID, primary.loc[list(GRAPH_GRID)].f_left, color=INK, marker="o", ms=3.0, lw=1.1, label="matched-branch data selection")
    ax.set_ylim(0.48, 1.025)
    ax.set_xlabel("added unseen data points")
    ax.set_ylabel(r"identifiable left-pair fraction $f_L$")
    ax.set_title("Phase-identifiable support expands")
    set_addition_axis(ax)
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1, 1]
    graph_primary = primary.loc[[0, *ADDITION_SIZES]]
    ax.plot(
        [0, *ADDITION_SIZES],
        graph_primary.K_perc_primary,
        color=INK,
        linestyle=(0, (5, 3)),
        marker="D",
        markerfacecolor="white",
        markeredgecolor=INK,
        markersize=3.4,
        lw=1.7,
        label=r"graph-only $K_{\rm perc}$",
        zorder=3,
    )
    ax.plot(ADDITION_SIZES, learned.loc[list(ADDITION_SIZES)].K_f_dist_mean, color=sky, marker="o", ms=4.0, lw=2.0, label=r"learned $K_f$, distinct")
    ax.plot(ADDITION_SIZES, learned.loc[list(ADDITION_SIZES)].K_f_dup_mean, color=vermilion, marker="s", ms=3.7, lw=1.9, label=r"learned $K_f$, duplicate")
    ax.set_xlabel("number of added data points")
    ax.set_ylabel(r"effective Schmidt rank $K_f$")
    ax.set_title("Graph-only prediction and learned state")
    set_addition_axis(ax)
    ax.legend(frameon=False, loc="upper right")

    for label, ax in zip("ABCD", axes.flat):
        ax.text(-0.13, 1.06, label, transform=ax.transAxes, fontweight="bold", fontsize=11)

    fig.suptitle(
        "Constraint percolation expands support while the model response remains nonmonotonic",
        fontsize=10.5,
        fontweight="bold",
    )
    save(fig, "fig7_constraint_percolation_vs_state")


# ==========================================================================
# Fig. 8: order of the responses at fixed support
# ==========================================================================

# field, colour, axis label, title, direction of the response, and the first
# saved update at which the 95 per cent interval lies entirely on that side
# of zero.  The six updates are read off contrasts.csv, not derived here.
TIMING_PANELS = (
    ("Q_in", TEAL, r"$\Delta Q_{\rm in}$", "correlation magnitude", "increase", 5),
    ("K_f", PURPLE, r"$\Delta K_f$", "effective Schmidt rank", "decrease", 5),
    ("Gamma_1", BLUE, r"$\Delta\Gamma_1$", "uniform-mode coherence", "increase", 5),
    ("R_in", ORANGE, r"$\Delta R_{\rm in}$", "target alignment", "increase", 200),
    ("w_star", INK, r"$\Delta w_\star$", "target-mode population", "increase", 200),
    ("A_block", CORAL, r"$\Delta A_{\rm blk}$", "full-space block accuracy", "increase", 400),
)


def fig8_fixed_identifiable_support() -> None:
    """Fig. 8: order of the responses to added data at fixed support (Sec. 7).

    The distinct branch with 1,216 added inputs has a connected constraint
    graph, f_L = 1, from update 0, so the identifiable support does not
    change along it.  Six panels give the distinct-minus-duplicate contrast
    of one quantity at the 11 saved updates: (A) Q_in, (B) K_f,
    (C) Gamma_1, (D) R_in, (E) w_star, (F) full-space block accuracy.  Points are means
    over the 25 starting models, bars the central 95 per cent bootstrap
    intervals over starting models.  The update axis is linear up to 5 and
    logarithmic beyond.

    Each title names the first saved update at which the interval lies
    entirely on the stated side of zero: update 5 for the rise of Q_in, the
    fall of K_f and the rise of Gamma_1, update 200 for the rise of R_in and
    of w_star, update 400 for the rise of block accuracy.  These are the
    literals of ``TIMING_PANELS``.

    Reads   results/fixed_support_timing/contrasts.csv (update and, per field,
            delta_<field>_mean, _ci_low, _ci_high)
    Writes  figure/fig8_fixed_identifiable_support.pdf
    """
    figure_style({
        "font.family": "DejaVu Sans",
        "font.size": 8.3,
        "axes.titlesize": 9.1,
        "axes.labelsize": 8.2,
        "legend.fontsize": 7.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    # The round-trip parser returns the values the analysis wrote, to the last digit.
    time_summary = read_frame(RESULTS / "fixed_support_timing" / "contrasts.csv",
                              float_precision="round_trip")
    updates = time_summary["update"].to_numpy()

    fig, axes = plt.subplots(3, 2, figsize=(7.1, 7.0), sharex=True, constrained_layout=True)
    for panel, (axis, (field, color, ylabel, title, direction, first_update)) in enumerate(zip(axes.flat, TIMING_PANELS)):
        mean = time_summary[f"delta_{field}_mean"].to_numpy()
        low = time_summary[f"delta_{field}_ci_low"].to_numpy()
        high = time_summary[f"delta_{field}_ci_high"].to_numpy()
        axis.errorbar(updates, mean, yerr=np.vstack((mean - low, high - mean)), color=color, marker="o", lw=1.8, capsize=2.5)
        axis.axhline(0, color=GRAY, ls=":", lw=0.8)
        axis.set_ylabel(ylabel)
        axis.set_title(
            f"{title}\n(first detected {direction}: update {first_update})",
            loc="left",
            fontsize=8.4,
        )
        axis.set_xscale("symlog", linthresh=5)
        axis.grid(axis="y", alpha=0.13)
        axis.text(-0.14, 1.08, chr(ord("A") + panel), transform=axis.transAxes, fontweight="bold", fontsize=10)
    for axis in axes[-1]:
        shown_updates = (0, 5, 10, 25, 100, 200, 400, 800)
        axis.set_xticks(shown_updates, [str(value) for value in shown_updates], rotation=28, ha="right")
        axis.set_xlabel("continuation update; fixed $+1{,}216$ training set")
    fig.suptitle("Correlation magnitude precedes target alignment and accuracy", fontsize=10, fontweight="bold")

    save(fig, "fig8_fixed_identifiable_support")


# ==========================================================================
# Fig. 9: splitting and merging of the conditional functions
# ==========================================================================

def draw_ribbon(ax, y0, y1, x0, x1, width, color, alpha) -> None:
    """Band of constant height ``width`` from (x0, y0) to (x1, y1), its edges cubic Bezier curves."""
    verts = [(x0, y0), (x0 + .25, y0), (x1 - .25, y1), (x1, y1), (x1, y1 + width), (x1 - .25, y1 + width), (x0 + .25, y0 + width), (x0, y0 + width), (x0, y0)]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha))


def fig9_conditional_function_change() -> None:
    """Fig. 9: change of the conditional functions during retreat and recovery (Sec. 8.2).

    Each checkpoint has its own clustering of the 32 gauged conditional
    functions (``common.cluster_conditional_functions``), so the cluster labels at
    two checkpoints are unrelated, and B = H[T(e1) | T(e0)] measures how a
    partition at e0 splits at e1.

    (A) One retreat event, the one whose splitting entropy is nearest the
        median over events.  Left bars are the clusters T1, T2, ... at
        last-high, right bars the clusters T1', T2', ... at retreat, with
        heights proportional to the fraction of the 32 left substrings they
        hold; each band runs from a last-high to a retreat cluster with the
        fraction it carries as height, fainter below 0.04.  The clusters
        are those of output position 8; the B written in the panel is the
        mean over the four long-range outputs.
    (B) B against the change 1 - R_4 of the leading four-dimensional
        subspace, for every retreat event (filled) and its control interval
        (open), crosses at the median of each group.
    (C) For each event with a recovery checkpoint, a line joining the
        splitting H[T(recovery) | T(retreat)] and the merging
        H[T(retreat) | T(recovery)]; black bars at the medians.

    Reads   results/conditional_function_splitting/splitting_by_event.csv
                (one row per event: seed, B_retreat, R4_retreat, B_control,
                R4_control, B_forward_recovery, B_merge_recovery, and the 0/1 flag of a
                recovery checkpoint) and
                example_clusters.csv (the 32 labels of the event of panel A)
    Writes  figure/fig9_conditional_function_change.pdf
    """
    tables = RESULTS / "conditional_function_splitting"
    # Only event_id is text.
    mechanism = [
        {key: value if key == "event_id" else float(value) for key, value in row.items()}
        for row in read_rows(tables / "splitting_by_event.csv")
    ]
    example = sorted(read_rows(tables / "example_clusters.csv"), key=lambda row: int(row["left_substring"]))
    med_seed = int(example[0]["seed"])
    a0 = np.array([int(row["last_high_cluster"]) for row in example])
    a1 = np.array([int(row["retreat_cluster"]) for row in example])
    n0 = int(example[0]["n_last_high_clusters"])
    n1 = int(example[0]["n_retreat_clusters"])

    figure_style({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})

    # Fraction of each last-high cluster that goes to each retreat cluster.
    trans = np.zeros((n0, n1))
    for i in range(n0):
        mask = a0 == i
        trans[i] = np.bincount(a1[mask], minlength=n1) / mask.sum()
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.5), constrained_layout=True)
    ax = axes[0]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("One conditional-cluster transition")
    left_sizes = np.bincount(a0, minlength=n0) / len(a0)
    right_sizes = np.bincount(a1, minlength=n1) / len(a1)
    scale = .82
    ly = .09 + scale * np.cumsum(np.r_[0, left_sizes])
    ry = .09 + scale * np.cumsum(np.r_[0, right_sizes])
    child_colors = plt.cm.Set2(np.linspace(.05, .95, max(n1, 2)))
    for i, size in enumerate(left_sizes):
        ax.add_patch(Rectangle((.08, ly[i]), .10, scale*size, facecolor="#22577a", edgecolor="#14344a", linewidth=.7))
        ax.text(.02, ly[i]+scale*size/2, f"T{i+1}", va="center", fontsize=7, color="#17324a")
    for j, size in enumerate(right_sizes):
        ax.add_patch(Rectangle((.82, ry[j]), .10, scale*size, facecolor=child_colors[j], edgecolor="#5f2525", linewidth=.7))
        ax.text(.94, ry[j]+scale*size/2, f"T{j+1}' {100*size:.0f}%", va="center", fontsize=7, color="#5f2525")
    # Bands are stacked inside each bar in order of (i, j); a cursor per bar
    # tracks the height already used.
    left_cursor = ly[:-1].copy()
    right_cursor = ry[:-1].copy()
    for i in range(n0):
        for j in range(n1):
            frac = trans[i, j] * left_sizes[i]
            if frac <= 0:
                continue
            width = scale * frac
            draw_ribbon(ax, left_cursor[i], right_cursor[j], .18, .82, width, child_colors[j], alpha=.78 if frac >= .04 else .52)
            left_cursor[i] += width
            right_cursor[j] += width
    ax.text(.08, .02, "last-high", color=RUN_BLUE)
    ax.text(.82, .02, "retreat", color=RETREAT_RED)
    representative_b = next(r["B_retreat"] for r in mechanism if int(r["seed"]) == med_seed)
    ax.text(.50, .96, rf"$B={representative_b:.2f}$ bits", transform=ax.transAxes, ha="center", va="top", fontsize=7.5, weight="bold")

    # Splitting entropy against the change of the leading subspace.
    ax = axes[1]
    bx = np.array([r["B_retreat"] for r in mechanism])
    by = 1 - np.array([r["R4_retreat"] for r in mechanism])
    cx = np.array([r["B_control"] for r in mechanism])
    cy = 1 - np.array([r["R4_control"] for r in mechanism])
    ax.scatter(cx, cy, facecolors="none", edgecolors=CONTROL_GREEN, alpha=.65, s=22, linewidths=.8, label="control interval")
    ax.scatter(bx, by, color=RETREAT_RED, alpha=.80, s=22, linewidths=0, label="retreat")
    ax.scatter(np.median(cx), np.median(cy), marker="X", facecolors="white", edgecolors=CONTROL_GREEN, s=60, linewidths=1.3)
    ax.scatter(np.median(bx), np.median(by), marker="X", color=RETREAT_RED, s=60)
    ax.set_xlabel(r"splitting entropy $B$ (bits)")
    ax.set_ylabel(r"subspace change $1-R_4$")
    ax.set_title("Splitting and subspace change separate retreat")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(alpha=.14)

    # The two conditional entropies between retreat and recovery.
    ax = axes[2]
    rec = [r for r in mechanism if r["recovered"]]
    forward = np.array([r["B_forward_recovery"] for r in rec])
    merge = np.array([r["B_merge_recovery"] for r in rec])
    for forward_value, merge_value in zip(forward, merge):
        ax.plot([0, 1], [forward_value, merge_value], color="#8c939a", alpha=.35, lw=.8, zorder=1)
    ax.scatter(np.zeros(len(forward)), forward, color=RETREAT_RED, s=15, alpha=.8, zorder=2)
    ax.scatter(np.ones(len(merge)), merge, color=RECOVERY_GREEN, s=15, alpha=.8, zorder=2)
    for i, values in enumerate((forward, merge)):
        ax.hlines(np.median(values), i-.18, i+.18, color="k", lw=2, zorder=3)
    label_y = max(float(np.max(forward)), float(np.max(merge))) + .12
    ax.text(0, label_y, r"$H[T(e_{\rm rec})\mid T(e_{\rm ret})]$", ha="center", va="bottom", fontsize=8)
    ax.text(1, label_y, r"$H[T(e_{\rm ret})\mid T(e_{\rm rec})]$", ha="center", va="bottom", fontsize=8)
    ax.set_xticks([0, 1], ["splitting", "merging"])
    ax.set_xlim(-.35, 1.35)
    ax.set_ylim(-.05, label_y + .18)
    ax.set_ylabel("conditional entropy (bits)")
    ax.set_title("Recovery is many-to-one")
    ax.grid(alpha=.14, axis="y")
    for label, panel in zip("ABC", axes):
        panel.text(.01, .88 if label == "C" else .98, label, transform=panel.transAxes, fontsize=11, fontweight="bold", va="top")

    save(fig, "fig9_conditional_function_change")


# ==========================================================================
# Fig. 10: scalar return, pair-pattern return and incomplete function return
# ==========================================================================

STAGES = ("last_high", "retreat", "recovery")


def fig10_retreat_recovery_fixed_graph() -> None:
    """Fig. 10: retreat and recovery of the 20 events with a recovery checkpoint (Sec. 8.3).

    The N = 1,280 training graph is connected (f_L = 1), so the support is
    fixed and every change here belongs to the learned state.

    (A) Mean over the 20 events of the off-diagonal C_h^2 and of w_star at
        last-high, retreat and recovery.
    (B) Histogram (density) at each of the three checkpoints of the Hamming
        distance between the conditional functions of two left substrings,
        as a fraction of their length 64, over all pairs a < a', the four
        long-range outputs and the 20 events, in 32 bins on [0, 1].  The
        peaks at 0 and 1 belong to h(a, b) close to u(a) v(b); retreat
        moves weight toward one half.
    (C) One point per event, averaged over the four long-range outputs: the
        pair-pattern correlation of C_h^2 between last-high and recovery
        against the function fidelity F_h(last-high, recovery), coloured by
        the function return ratio
        theta_h = [1 - F_h(last-high, recovery)] / [1 - F_h(last-high, retreat)].

    Reads   results/retreat_recurrence/recovered_stage_means.csv (stage, metric,
                mean) and event_summary.csv (C2_pair_pattern_pearson, F_h,
                eta_function_one_minus_F = theta_h)
            results/retreat_pair_correlations/pair_matrices.npz (hamming[event,
                checkpoint, output, 32, 32]) and sign_tables.npz (available
                [event, checkpoint]: which events have a recovery checkpoint)
    Writes  figure/fig10_retreat_recovery_fixed_graph.pdf
    """
    figure_style({
        "font.family": "DejaVu Sans",
        "font.size": 8.3,
        "axes.titlesize": 9.2,
        "axes.labelsize": 8.3,
        "legend.fontsize": 6.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    recurrence = RESULTS / "retreat_recurrence"
    pair_correlations = RESULTS / "retreat_pair_correlations"
    # The round-trip parser returns every number of the two tables exactly.
    fixed20 = read_frame(recurrence / "recovered_stage_means.csv", float_precision="round_trip")
    event = read_frame(recurrence / "event_summary.csv", float_precision="round_trip")
    hamming = read_npz(pair_correlations / "pair_matrices.npz")["hamming"].astype(float)
    available = read_npz(pair_correlations / "sign_tables.npz")["available"].astype(bool)
    recovered = np.flatnonzero(available[:, 2])
    upper = np.triu_indices(hamming.shape[-1], 1)

    fig, axs = plt.subplots(1, 3, figsize=(11.8, 3.55))
    x = np.arange(len(STAGES))
    for metric, color, marker, label in (("mean_offdiag_C2", PURPLE, "o", r"mean $C_h^2$"), ("w_star", TEAL, "s", r"$w_\star$")):
        subset = fixed20[fixed20.metric == metric].set_index("stage").loc[list(STAGES)]
        axs[0].plot(x, subset["mean"], color=color, marker=marker, lw=2.3, label=label)
    axs[0].set_xticks(x, ["last-high", "retreat", "recovery"])
    axs[0].set_ylabel("mean across 20 events")
    axs[0].set_title("Scalar organization reverses")
    axs[0].legend(frameon=False)
    axs[0].grid(axis="y", alpha=0.14)

    bins = np.linspace(0, 1, 33)
    for stage_index, label, color in ((0, "last-high", TEAL), (1, "retreat", ORANGE), (2, "recovery", BLUE)):
        values = hamming[recovered, stage_index][:, :, upper[0], upper[1]].ravel()
        axs[1].hist(values, bins=bins, density=True, histtype="step", lw=1.8, color=color, label=label)
    axs[1].set_xlabel("pairwise Hamming distance as a fraction of the whole length")
    axs[1].set_ylabel("density")
    axs[1].set_title(r"Recoherent $\to$ differentiated $\to$ recoherent")
    axs[1].legend(frameon=False)

    scatter = axs[2].scatter(
        event.C2_pair_pattern_pearson,
        event.F_h,
        c=event.eta_function_one_minus_F,
        cmap="viridis_r",
        s=48,
        edgecolor="white",
        linewidth=0.6,
    )
    axs[2].axhline(1, color=LIGHT_GRAY, lw=0.8)
    axs[2].set_xlabel(r"pair-pattern correlation $r(e_{\rm high},e_{\rm rec})$")
    axs[2].set_ylabel("projective function fidelity\n" + r"$F_h(\mathrm{last\!-\!high},\mathrm{recovery})$")
    axs[2].set_title("Similar pair pattern, incomplete function return")
    axs[2].grid(alpha=0.14)
    colorbar = fig.colorbar(scatter, ax=axs[2], fraction=0.047, pad=0.03)
    colorbar.set_label(r"function return ratio $\theta_h$")
    for index, label in enumerate("ABC"):
        axs[index].text(-0.13, 1.11, label, transform=axs[index].transAxes,
                        fontweight="bold", fontsize=11, va="top")
    fig.tight_layout()

    save(fig, "fig10_retreat_recovery_fixed_graph")


# ==========================================================================
# Fig. 11: robustness of the retreat and recovery result
# ==========================================================================

COHERENCE_THRESHOLDS = (0.85, 0.90, 0.95)   # thresholds on Gamma(n) in threshold_robustness.csv


def numeric(value: str):
    """``value`` as a float where it is a number, else the string itself."""
    try:
        return float(value)
    except ValueError:
        return value


def fig11_retreat_recovery_robustness() -> None:
    """Fig. 11: robustness of the retreat and recovery result (App. B.2).

    (A) Median over the first retreat events of K_f, Gamma_1 and w_star at
        the four epochs tau = -3, ..., 0 ending at retreat, with the
        interquartile range shaded; n in the title is the number of runs,
        one first event each.
    (B) Splitting entropy from last-high to retreat and across the control
        interval, one point per event, with the threshold on the cluster
        coherence Gamma(n) set to 0.85, 0.90 and 0.95.
    (C) The conditional entropy between retreat and recovery in both orders
        at the same thresholds, for the events with a recovery checkpoint:
        splitting H[T(recovery) | T(retreat)] and merging
        H[T(retreat) | T(recovery)].

    The figure uses the Matplotlib defaults throughout.

    Reads   results/conditional_function_splitting/approach_to_retreat.csv
                (seed, tau, C_func_PR = K_f, C1 = Gamma_1, w_star) and
                threshold_robustness.csv (threshold, B_retreat, B_control,
                H_recovery_given_retreat, H_retreat_given_recovery; the
                last two read "nan" for events without a recovery
                checkpoint, and panel C keeps the rows where the first is
                finite)
    Writes  figure/fig11_retreat_recovery_robustness.pdf
    """
    tables = RESULTS / "conditional_function_splitting"
    balanced_rows = [{k: numeric(v) for k, v in row.items()}
                     for row in read_rows(tables / "approach_to_retreat.csv")]
    robustness_rows = [{k: numeric(v) for k, v in row.items()}
                       for row in read_rows(tables / "threshold_robustness.csv")]

    figure_style()
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.9), constrained_layout=True)
    for key, color, label in (("C_func_PR", RETREAT_RED, r"$K_f$"), ("C1", RUN_ORANGE, r"$\Gamma_1$"), ("w_star", RUN_BLUE, r"$w_\star$")):
        by_tau = defaultdict(list)
        for row in balanced_rows:
            by_tau[int(row["tau"])].append(float(row[key]))
        taus = sorted(by_tau)
        med = [np.median(by_tau[t]) for t in taus]
        lo = [np.quantile(by_tau[t], .25) for t in taus]
        hi = [np.quantile(by_tau[t], .75) for t in taus]
        axes[0].plot(taus, med, marker="o", color=color, label=label)
        axes[0].fill_between(taus, lo, hi, color=color, alpha=.14)
    axes[0].axvline(0, color="#333", ls="--", lw=.7)
    axes[0].set_title(f"(A) Approach to retreat (n={len(set(r['seed'] for r in balanced_rows))})")
    axes[0].set_xlabel("epoch relative to retreat")
    axes[0].legend(frameon=False, fontsize=7)
    axes[0].grid(alpha=.14)
    for threshold in COHERENCE_THRESHOLDS:
        rows_t = [r for r in robustness_rows if r["threshold"] == threshold]
        axes[1].scatter([threshold] * len(rows_t), [r["B_control"] for r in rows_t], facecolors="none", edgecolors=CONTROL_GREEN, s=12, alpha=.45, label="control\ninterval" if threshold == COHERENCE_THRESHOLDS[0] else None)
        axes[1].scatter([threshold] * len(rows_t), [r["B_retreat"] for r in rows_t], color=RETREAT_RED, s=12, alpha=.55, label="retreat" if threshold == COHERENCE_THRESHOLDS[0] else None)
    axes[1].set_title("(B) Splitting entropy")
    axes[1].set_xlabel("cluster-selection threshold\n" + r"on $\Gamma(n)$")
    axes[1].set_ylabel(r"$B$ (bits)")
    axes[1].legend(frameon=False, fontsize=6.5, loc="center left", bbox_to_anchor=(.16, .55))
    axes[1].grid(alpha=.14)
    for threshold in COHERENCE_THRESHOLDS:
        rows_t = [r for r in robustness_rows if r["threshold"] == threshold and np.isfinite(r["H_recovery_given_retreat"])]
        axes[2].scatter([threshold] * len(rows_t), [r["H_recovery_given_retreat"] for r in rows_t], color=RETREAT_RED, s=12, alpha=.5, label="splitting\n" + r"$H[T(e_{\rm rec})\mid T(e_{\rm ret})]$" if threshold == COHERENCE_THRESHOLDS[0] else None)
        axes[2].scatter([threshold] * len(rows_t), [r["H_retreat_given_recovery"] for r in rows_t], color=RECOVERY_GREEN, marker="s", s=12, alpha=.5, label="merging\n" + r"$H[T(e_{\rm ret})\mid T(e_{\rm rec})]$" if threshold == COHERENCE_THRESHOLDS[0] else None)
    axes[2].set_title("(C) Recovery direction")
    axes[2].set_xlabel("cluster-selection threshold\n" + r"on $\Gamma(n)$")
    axes[2].set_ylabel(r"$B$ (bits)")
    axes[2].legend(frameon=False, fontsize=6, loc="center left", bbox_to_anchor=(.07, .55))
    axes[2].grid(alpha=.14)

    # First pass of the constrained layout; ``savefig`` performs the second.
    fig.canvas.draw()
    save(fig, "fig11_retreat_recovery_robustness")


# ==========================================================================
# Command line
# ==========================================================================

# (name, function, paper location, analysis items whose tables it reads), in
# the order of the paper.  Every figure writes figure/<name>.pdf.
FIGURES: list[tuple[str, Callable[[], None], str, tuple[str, ...]]] = [
    ("fig1_support_state_performance", fig1_support_state_performance, "Fig. 1, Sec. 2", ()),
    ("fig2_output_position_dependence", fig2_output_position_dependence, "Fig. 2, Sec. 4.1", ("output_position_dependence",)),
    ("fig3_training_trajectories", fig3_training_trajectories, "Fig. 3, Secs. 4.2, 4.3",
     ("training_trajectories", "retreat_events", "state_vs_accuracy")),
    ("fig4_state_vs_accuracy", fig4_state_vs_accuracy, "Fig. 4, Sec. 4.4", ("state_vs_accuracy",)),
    ("fig5_starting_model_branches", fig5_starting_model_branches, "Fig. 5, Sec. 5.1",
     ("largest_addition_branches",)),
    ("fig6_added_data_size", fig6_added_data_size, "Fig. 6, Sec. 5.2", ("branch_endpoint_state",)),
    ("fig7_constraint_percolation_vs_state", fig7_constraint_percolation_vs_state, "Fig. 7, Sec. 6",
     ("constraint_percolation", "branch_endpoint_state")),
    ("fig8_fixed_identifiable_support", fig8_fixed_identifiable_support, "Fig. 8, Sec. 7", ("fixed_support_timing",)),
    ("fig9_conditional_function_change", fig9_conditional_function_change, "Fig. 9, Sec. 8.2",
     ("conditional_function_splitting",)),
    ("fig10_retreat_recovery_fixed_graph", fig10_retreat_recovery_fixed_graph, "Fig. 10, Sec. 8.3",
     ("retreat_recurrence", "retreat_pair_correlations")),
    ("fig11_retreat_recovery_robustness", fig11_retreat_recovery_robustness, "Fig. 11, App. B.2",
     ("conditional_function_splitting",)),
]


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Draw the figures of the paper; each writes one PDF under figure/.",
        epilog="Run with --list for the figures.",
    )
    parser.add_argument("names", nargs="*", metavar="NAME", help="figures to draw in this process")
    parser.add_argument("--list", action="store_true", help="list the figures and stop")
    parser.add_argument("--all", action="store_true",
                        help="draw every figure, each in its own process")
    args = parser.parse_args(argv)
    figures = {name: function for name, function, _paper, _needs in FIGURES}

    if args.list:
        for name, function, paper, needs in FIGURES:
            summary = (function.__doc__ or "").strip().splitlines()[0]
            tables = ", ".join(f"results/{item}/" for item in needs) or "no table"
            print(f"{name:38s} {paper:24s} -> figure/{name}.pdf\n"
                  f"{'':38s} {summary}\n{'':38s} reads {tables}")
        return 0
    if args.all:
        if args.names:
            parser.error("--all takes no figure names")
        # One process per figure, so that no Matplotlib or module state of one
        # figure reaches the next.
        environment = dict(os.environ)
        for name in THREAD_VARIABLES:
            environment.setdefault(name, "1")
        for name, _function, _paper, _needs in FIGURES:
            print(f"[figures] {name}", flush=True)
            start = time.time()
            status = subprocess.run([sys.executable, str(Path(__file__).resolve()), name],
                                    cwd=ROOT, env=environment).returncode
            if status:
                print(f"{name} exited with status {status}", file=sys.stderr)
                return status
            print(f"    done in {time.time() - start:.0f} s", flush=True)
        return 0
    if not args.names:
        parser.error("name one or more figures, or use --list or --all")
    unknown = [name for name in args.names if name not in figures]
    if unknown:
        parser.error(f"unknown figure {unknown[0]!r}; see --list")
    for name in args.names:
        figures[name]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
