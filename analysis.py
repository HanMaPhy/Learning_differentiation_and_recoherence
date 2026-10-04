#!/usr/bin/env python3
"""Every analysis of the paper: from the trained networks in data/ to the tables in results/.

One function per analysis.  Each reads data/ and, where noted, tables written
by an item listed before it, and writes one directory under results/ named
after it.  Nothing here trains a network; the forward passes evaluate stored
weights on the 2,048 inputs of the full binary space.  The observables (K_f,
Gamma_1, w_star, the constraint graph, the pair decomposition, the
clustering of conditional functions) are defined in ``common.py``, whose
docstring gives the physics.

    python3 analysis.py --list               every item: name, paper location, what it writes
    python3 analysis.py NAME [NAME ...]      run the named items in this process
    python3 analysis.py --all                every item in dependency order, each in its own process

The analyses are run with one BLAS thread: the order of floating-point
accumulation in a matrix product depends on the thread count, and with one
thread the tables are the same on every run.  This file sets OMP_NUM_THREADS,
MKL_NUM_THREADS, OPENBLAS_NUM_THREADS, VECLIB_MAXIMUM_THREADS and
NUMEXPR_NUM_THREADS to 1 unless they are set already.  ``--all`` takes about
half an hour on one core, most of it in ``state_vs_accuracy`` and
``training_trajectories``.

THE THREE DATA SETS
-------------------
Large-data runs (N = 1,280; Secs. 3, 4.1, 4.3, 4.4, 8).  30 networks trained for 220
epochs on the same 1,280 inputs (963 distinct), with weights stored at every
epoch from 20 on and a monitoring block accuracy at every epoch (from epoch 20
for seeds 6, 7 and 8, evaluated from their stored weights).  Generalization
onset is the first epoch with monitoring block accuracy >= 0.8.  A retreat is
the first later epoch with accuracy < 0.5; the first retreat of each of the 24
seeds that have one is the unit of the retreat analyses, at its last-high
checkpoint (the last epoch >= 0.8 before the retreat), its retreat checkpoint
and, in 20 of them, the first recovery checkpoint (>= 0.8 again).

Small-data runs (N = 64; Sec. 4.2).  24 runs with the function state stored at
52 optimizer steps.

Branching study (Secs. 5, 6, 7, App. B.1).  25 starting models trained on the
64 base inputs, each continued for 800 updates on the base inputs plus n
distinct inputs, plus n repeated base inputs, or on the base inputs alone.
Every contrast is distinct minus duplicate, formed within a starting model and
then averaged over starting models, with a bootstrap interval over starting
models.

Paper location                         function
--------------                         --------
Table 1, Sec. 4.3                      retreat_events
Sec. 8 (input of the retreat tables)   event_sign_tables
Fig. 2, Sec. 4.1                       output_position_dependence
Fig. 3, Secs. 4.2, 4.3                 training_trajectories
Figs. 3B, 4, Secs. 4.3, 4.4            state_vs_accuracy
Fig. 5, Sec. 5.1                       largest_addition_branches
Sec. 5.1, App. B.1                     branch_contrasts
Fig. 6, Sec. 5.2                       branch_endpoint_state
Fig. 7, Sec. 6                         constraint_percolation, branch_endpoint_state
Fig. 8, Sec. 7                         fixed_support_timing
Sec. 3, Sec. 8.1                       retreat_pair_correlations
Fig. 9, Sec. 8.2, Fig. 11, App. B.2    conditional_function_splitting
Fig. 10, Secs. 8.1, 8.3                retreat_recurrence
"""

from __future__ import annotations

import os

# One BLAS thread for every analysis, set before numpy and torch are imported.
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
for _name in THREAD_VARIABLES:
    os.environ.setdefault(_name, "1")

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

from common import (
    ADDITION_SIZES,
    ARCHITECTURE,
    BASE_N,
    BRANCHES,
    CUT,
    ENDPOINT_UPDATE,
    EVENTS,
    EVENT_SIGN_TABLES,
    L,
    LARGE_DATA_TRAJECTORIES,
    LONG_RANGE_POSITIONS,
    N_LEFT,
    N_NODES,
    N_STRINGS,
    OFFDIAG_COUNT,
    RESULTS,
    ROOT,
    BRANCHING_STUDY,
    SAVED_UPDATES,
    SMALL_DATA_TRAJECTORIES,
    TRAJECTORIES,
    ChainTransformer,
    UnionFind,
    all_events,
    base_training_codes,
    bootstrap,
    cluster_conditional_functions,
    constraint_graph,
    distinct_branch_codes,
    effective_rank,
    first_events,
    full_binary_space,
    load_checkpoint_sign_tables,
    load_endpoint_sign_tables,
    load_sign_table,
    long_range_target_signs,
    pair_averages,
    pair_matrices,
    rank_and_coherence,
    read_csv,
    reduced_state,
    sign_table_path,
    splitting_entropy,
    stage_epochs,
    subspace_overlap,
    target_population,
    target_signs,
    training_codes,
)

# ==========================================================================
# Helpers shared by several analyses
# ==========================================================================

def write_rows(path: Path, rows: list[dict]) -> None:
    """CSV with the keys of the first row as header (csv module, CRLF line ends); nothing for no rows."""
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def large_data_network() -> ChainTransformer:
    """An N = 1,280 network to load stored weights into."""
    return ChainTransformer(L=L, **ARCHITECTURE, dropout=0.0)


def output_signs(model: ChainTransformer, path: Path, inputs: torch.Tensor) -> np.ndarray:
    """Raw output signs of the stored network ``path`` on ``inputs``, int8 (n, L).

    +1 for a logit >= 0, -1 otherwise, before any target gauge.  The weights are
    loaded into ``model`` and evaluated in eval mode without gradients.
    """
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        logits = model(inputs).numpy()
    return np.where(logits >= 0.0, 1, -1).astype(np.int8)


def seed_checkpoint_dir(event: dict[str, str]) -> Path:
    """Directory of the stored epochs of the run of one event, from its retreat checkpoint path."""
    return ROOT / event["retreat_checkpoint"].rpartition("/ep")[0]


# ==========================================================================
# retreat_events: events of the N = 1,280 runs (Table 1, Sec. 4.3)
# ==========================================================================

ONSET_THRESHOLD = 0.8
RETREAT_THRESHOLD = 0.5
CHECKPOINT_PATTERN = re.compile(r"seed(?P<seed>\d+)/ep(?P<epoch>\d+)\.pt$")
MONITORING = LARGE_DATA_TRAJECTORIES / "monitoring_seeds06_08.csv"
# Column order of trajectories.csv, ahead of ``source`` and ``checkpoint_available``.
MONITORING_COLUMNS = ("block_acc", "seed", "epoch", "loss")


def repo_relative(path: Path) -> str:
    """``path`` relative to the repository root, without resolving the data link."""
    return str(path.relative_to(ROOT))


def monitoring_trajectories() -> pd.DataFrame:
    """All recorded (seed, epoch) rows of the monitoring curves, sorted by seed and epoch.

    The four training-time tables monitoring_seeds00_05.csv,
    monitoring_seeds09_15.csv, monitoring_seeds16_22.csv and
    monitoring_seeds23_29.csv cover 27 seeds at epochs 1 to 220;
    ``monitoring_seeds06_08.csv`` holds seeds 6, 7 and 8 at the stored epochs
    20 to 220, evaluated on the same monitoring inputs.  The tables are read
    in this order.  A (seed, epoch) that appears twice is an error.
    """
    files = sorted(path for path in LARGE_DATA_TRAJECTORIES.glob("monitoring_seeds*.csv")
                   if path != MONITORING)
    if not files:
        raise FileNotFoundError(f"no monitoring tables under {repo_relative(LARGE_DATA_TRAJECTORIES)}")
    frames = []
    for path in files + [MONITORING]:
        frame = pd.read_csv(path)
        if not set(MONITORING_COLUMNS).issubset(frame.columns):
            raise ValueError(f"{path} is missing {set(MONITORING_COLUMNS) - set(frame.columns)}")
        frame = frame[list(MONITORING_COLUMNS)].copy()
        frame["source"] = repo_relative(path)
        frames.append(frame)
    trajectories = pd.concat(frames, ignore_index=True)
    duplicates = trajectories.duplicated(["seed", "epoch"], keep=False)
    if duplicates.any():
        repeated = trajectories.loc[duplicates, ["seed", "epoch", "source"]]
        raise ValueError(f"Duplicate seed/epoch rows found:\n{repeated.to_string(index=False)}")
    return trajectories.sort_values(["seed", "epoch"]).reset_index(drop=True)


def index_checkpoints() -> dict[tuple[int, int], Path]:
    """Map (seed, epoch) to the stored weights ``data/large_data_trajectories/checkpoints/seed<k>/ep<e>.pt``."""
    index: dict[tuple[int, int], Path] = {}
    for path in sorted(LARGE_DATA_TRAJECTORIES.glob("checkpoints/seed*/ep*.pt")):
        match = CHECKPOINT_PATTERN.search(str(path))
        if match is None:
            continue
        seed = int(match.group("seed"))
        epoch = int(match.group("epoch"))
        key = (seed, epoch)
        if key in index:
            raise ValueError(f"Duplicate checkpoint for seed={seed}, epoch={epoch}")
        index[key] = path
    return index


def checkpoint_path(checkpoint_index: dict[tuple[int, int], Path], seed: int, epoch: int) -> str:
    """Repository-relative path of one stored checkpoint, empty when none is stored."""
    path = checkpoint_index.get((seed, epoch))
    return repo_relative(path) if path is not None else ""


def detect_events(
    trajectories: pd.DataFrame, checkpoint_index: dict[tuple[int, int], Path]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Events of every seed, and the onsets with no retreat up to the last epoch.

    Scanning the epochs of one seed in order: while inactive, the first epoch
    with block accuracy >= 0.8 is the onset; after it, the first epoch with
    accuracy < 0.5 is the retreat and closes the event, and the detector
    becomes inactive again.  For every event: the peak (first maximum between
    onset and retreat), last-high (last epoch >= 0.8 before the retreat), the
    epoch before the retreat, the first epoch >= 0.8 after the retreat
    (``recovery_epoch``, None if there is none up to epoch 220), the gaps, the
    depths and the path of each stored checkpoint.
    """
    events: list[dict[str, Any]] = []
    censored: list[dict[str, Any]] = []
    for seed, group in trajectories.groupby("seed", sort=True):
        group = group.sort_values("epoch").reset_index(drop=True)
        onset_index: int | None = None
        event_number = 0
        for row_index, row in group.iterrows():
            accuracy = float(row["block_acc"])
            if onset_index is None:
                if accuracy >= ONSET_THRESHOLD:
                    onset_index = int(row_index)
                continue
            if accuracy >= RETREAT_THRESHOLD:
                continue

            event_number += 1
            high_segment = group.iloc[onset_index:row_index]
            peak_index = int(high_segment["block_acc"].idxmax())
            high_rows = high_segment[high_segment["block_acc"] >= ONSET_THRESHOLD]
            last_high_index = int(high_rows.index[-1])
            previous_index = int(row_index) - 1
            recovery_rows = group.iloc[int(row_index) + 1 :]
            recovery_rows = recovery_rows[recovery_rows["block_acc"] >= ONSET_THRESHOLD]
            recovery_epoch = int(recovery_rows.iloc[0]["epoch"]) if len(recovery_rows) else None

            onset = group.loc[onset_index]
            peak = group.loc[peak_index]
            last_high = group.loc[last_high_index]
            previous = group.loc[previous_index]
            retreat = row
            event = {
                "event_id": f"seed{int(seed):02d}_event{event_number:02d}",
                "seed": int(seed),
                "event_number_within_seed": event_number,
                "acquisition_epoch": int(onset["epoch"]),
                "acquisition_acc": float(onset["block_acc"]),
                "peak_epoch": int(peak["epoch"]),
                "peak_acc": float(peak["block_acc"]),
                "last_high_epoch": int(last_high["epoch"]),
                "last_high_acc": float(last_high["block_acc"]),
                "previous_epoch": int(previous["epoch"]),
                "previous_acc": float(previous["block_acc"]),
                "retreat_epoch": int(retreat["epoch"]),
                "retreat_acc": float(retreat["block_acc"]),
                "recovery_epoch": recovery_epoch,
                "epochs_acquisition_to_retreat": int(retreat["epoch"] - onset["epoch"]),
                "epochs_peak_to_retreat": int(retreat["epoch"] - peak["epoch"]),
                "epochs_last_high_to_retreat": int(retreat["epoch"] - last_high["epoch"]),
                "retreat_depth_from_peak": float(peak["block_acc"] - retreat["block_acc"]),
                "retreat_depth_from_last_high": float(last_high["block_acc"] - retreat["block_acc"]),
                "recovered_by_epoch_220": recovery_epoch is not None,
            }
            for label in ("acquisition", "peak", "last_high", "previous", "retreat"):
                epoch = int(event[f"{label}_epoch"])
                path = checkpoint_path(checkpoint_index, int(seed), epoch)
                event[f"{label}_checkpoint"] = path
                event[f"{label}_checkpoint_available"] = bool(path)
            events.append(event)
            onset_index = None

        if onset_index is not None:
            onset = group.loc[onset_index]
            censored.append(
                {
                    "seed": int(seed),
                    "acquisition_epoch": int(onset["epoch"]),
                    "acquisition_acc": float(onset["block_acc"]),
                    "final_epoch": int(group.iloc[-1]["epoch"]),
                    "final_acc": float(group.iloc[-1]["block_acc"]),
                }
            )
    return events, censored


def control_candidates(
    group: pd.DataFrame,
    checkpoint_index: dict[tuple[int, int], Path],
    seed: int,
    event_start: int,
    event_end: int,
) -> list[tuple[float, int, int]]:
    """Intervals of the same seed with the same epoch gap, disjoint from the event, all epochs >= 0.8.

    Both ends must have a stored checkpoint.  Sorted by the distance of the
    midpoint from the midpoint of the event interval, then by start epoch.
    """
    lag = event_end - event_start
    indexed = group.set_index("epoch").sort_index()
    candidates = []
    event_midpoint = 0.5 * (event_start + event_end)
    for start in indexed.index:
        start = int(start)
        end = start + lag
        if end not in indexed.index:
            continue
        if not (end < event_start or start > event_end):
            continue
        window = indexed.loc[start:end]
        if len(window) != lag + 1 or not bool((window["block_acc"] >= ONSET_THRESHOLD).all()):
            continue
        if (seed, start) not in checkpoint_index or (seed, end) not in checkpoint_index:
            continue
        distance = abs(0.5 * (start + end) - event_midpoint)
        candidates.append((float(distance), start, end))
    return sorted(candidates)


def attach_controls(
    events: list[dict[str, Any]],
    trajectories: pd.DataFrame,
    checkpoint_index: dict[tuple[int, int], Path],
) -> None:
    """Add the ``local`` and ``peak`` control interval to every event, in place.

    ``local`` has the epoch gap of last-high -> retreat, ``peak`` that of
    peak -> retreat.
    The nearest candidate not taken by another event of the same seed is
    chosen; when every candidate is taken the nearest one is reused and
    ``*_reused`` is True.  The columns are empty when the seed has none.
    """
    groups = {
        int(seed): group.sort_values("epoch").reset_index(drop=True)
        for seed, group in trajectories.groupby("seed")
    }
    used: dict[tuple[int, str], set[tuple[int, int]]] = {}
    for event in events:
        seed = int(event["seed"])
        for kind, start_key in (("local", "last_high_epoch"), ("peak", "peak_epoch")):
            event_start = int(event[start_key])
            event_end = int(event["retreat_epoch"])
            candidates = control_candidates(groups[seed], checkpoint_index, seed, event_start, event_end)
            used_key = (seed, kind)
            used.setdefault(used_key, set())
            fresh = [candidate for candidate in candidates if candidate[1:] not in used[used_key]]
            selected = fresh[0] if fresh else (candidates[0] if candidates else None)
            prefix = f"{kind}_control"
            if selected is None:
                event[f"{prefix}_start_epoch"] = None
                event[f"{prefix}_end_epoch"] = None
                event[f"{prefix}_start_acc"] = None
                event[f"{prefix}_end_acc"] = None
                event[f"{prefix}_reused"] = None
                event[f"{prefix}_start_checkpoint"] = ""
                event[f"{prefix}_end_checkpoint"] = ""
                continue
            _, start, end = selected
            reused = (start, end) in used[used_key]
            used[used_key].add((start, end))
            indexed = groups[seed].set_index("epoch")
            event[f"{prefix}_start_epoch"] = start
            event[f"{prefix}_end_epoch"] = end
            event[f"{prefix}_start_acc"] = float(indexed.loc[start, "block_acc"])
            event[f"{prefix}_end_acc"] = float(indexed.loc[end, "block_acc"])
            event[f"{prefix}_reused"] = reused
            event[f"{prefix}_start_checkpoint"] = checkpoint_path(checkpoint_index, seed, start)
            event[f"{prefix}_end_checkpoint"] = checkpoint_path(checkpoint_index, seed, end)


def summarize_events(events: list[dict[str, Any]], censored: list[dict[str, Any]]) -> dict[str, Any]:
    """The event rule in words and the counts over events and seeds (``summary.json``)."""
    event_seeds = sorted({int(event["seed"]) for event in events})
    local_lags = Counter(int(event["epochs_last_high_to_retreat"]) for event in events)
    peak_lags = Counter(int(event["epochs_peak_to_retreat"]) for event in events)
    return {
        "protocol": {
            "acquisition_threshold": ONSET_THRESHOLD,
            "retreat_threshold": RETREAT_THRESHOLD,
            "event_definition": (
                "After the first block_acc >= 0.8 while inactive, the first later "
                "block_acc < 0.5 closes an event; the detector then resets."
            ),
            "peak_definition": "First maximum block accuracy between acquisition and retreat.",
            "local_pair_definition": "Last epoch with block_acc >= 0.8 before first block_acc < 0.5.",
            "control_interval_definition": (
                "Same seed and identical epoch gap; every intervening recorded accuracy >= 0.8; "
                "control interval is disjoint from the event pair."
            ),
            "monitoring_list": "4,096 inputs drawn with seed 777",
        },
        "events": {
            "event_count": len(events),
            "event_seed_count": len(event_seeds),
            "event_seeds": event_seeds,
            "events_per_seed": {
                str(seed): count
                for seed, count in sorted(Counter(event["seed"] for event in events).items())
            },
            "seeds_with_multiple_events": sorted(
                seed for seed, count in Counter(event["seed"] for event in events).items() if count > 1
            ),
            "censored_acquisition_count": len(censored),
            "censored_acquisition_seed_count": len({row["seed"] for row in censored}),
            "all_event_anchor_checkpoints_available": all(
                all(
                    bool(event[f"{label}_checkpoint_available"])
                    for label in ("acquisition", "peak", "last_high", "retreat")
                )
                for event in events
            ),
            "local_lag_histogram": {str(key): value for key, value in sorted(local_lags.items())},
            "peak_lag_histogram": {str(key): value for key, value in sorted(peak_lags.items())},
            "mean_peak_to_retreat_depth": float(np.mean([event["retreat_depth_from_peak"] for event in events])),
            "median_peak_to_retreat_depth": float(np.median([event["retreat_depth_from_peak"] for event in events])),
            "recovered_by_epoch_220_count": sum(bool(event["recovered_by_epoch_220"]) for event in events),
            "local_matched_control_count": sum(event["local_control_start_epoch"] is not None for event in events),
            "peak_matched_control_count": sum(event["peak_control_start_epoch"] is not None for event in events),
            "local_control_reuse_count": sum(bool(event["local_control_reused"]) for event in events),
            "peak_control_reuse_count": sum(bool(event["peak_control_reused"]) for event in events),
        },
    }


def retreat_events() -> None:
    """Retreat events of the N = 1,280 runs and their control intervals (Table 1, Sec. 4.3).

    Reads the monitoring curves (block accuracy and loss on 4,096 inputs drawn
    from seed 777, once per epoch) of the 30 runs, finds every event in which
    a run that generalized loses the function again (``detect_events``) and
    names the checkpoints the retreat analyses compare.  No network is
    evaluated; checkpoint files are listed, not opened.

    A change between two checkpoints of an event is compared with the change
    over a control interval of the same seed in which nothing happens
    (``attach_controls``): the same epoch gap, no overlap with the event, all
    accuracies >= 0.8, stored checkpoints at both ends, midpoint nearest the
    event's.  An onset with no retreat by epoch 220 is not an event; the
    summary counts those.

    Reads   data/large_data_trajectories/monitoring_seeds*.csv and the list of
            data/large_data_trajectories/checkpoints/seed<k>/ep<e>.pt
    Writes  results/retreat_events/
              events.csv        one row per event: epochs, accuracies,
                                checkpoint paths, both control intervals
              trajectories.csv  every recorded (seed, epoch): block_acc, loss,
                                source table, whether a checkpoint is stored
              summary.json      the rule and the counts
    Every path in the tables is relative to the repository root.  A few seconds.
    """
    out = EVENTS.parent
    out.mkdir(parents=True, exist_ok=True)
    trajectories = monitoring_trajectories()
    checkpoint_index = index_checkpoints()
    trajectories["checkpoint_available"] = [
        (int(row.seed), int(row.epoch)) in checkpoint_index
        for row in trajectories.itertuples(index=False)
    ]
    trajectories.to_csv(TRAJECTORIES, index=False)

    events, censored = detect_events(trajectories, checkpoint_index)
    attach_controls(events, trajectories, checkpoint_index)
    write_rows(EVENTS, events)

    summary = summarize_events(events, censored)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {len(events)} events from {summary['events']['event_seed_count']} seeds to "
          f"{repo_relative(out)}")


# ==========================================================================
# event_sign_tables: sign tables of the event checkpoints (Sec. 8)
# ==========================================================================

FINAL_EPOCH = 220
FIRST_STORED_EPOCH = 20
APPROACH_WINDOW = (-3, -2, -1, 0)
EVENT_EPOCH_COLUMNS = (
    "acquisition_epoch", "peak_epoch", "last_high_epoch", "previous_epoch",
    "retreat_epoch", "recovery_epoch", "local_control_start_epoch",
    "local_control_end_epoch", "peak_control_start_epoch",
    "peak_control_end_epoch",
)


def required_checkpoints(rows: list[dict[str, str]]) -> dict[tuple[int, int], Path]:
    """Map every (seed, epoch) the retreat analyses read to the seed's checkpoint directory.

    For every event: each epoch the event table names and the last epoch,
    220.  For the first event of each seed also the four epochs
    tau = -3, -2, -1, 0 ending at the retreat (the approach to retreat of
    ``conditional_function_splitting``).  Epochs below 20 have no stored
    weights and are left out.
    """
    first_ids = {row["event_id"] for row in first_events()}
    needed: dict[tuple[int, int], Path] = {}
    for row in rows:
        seed = int(row["seed"])
        seed_dir = seed_checkpoint_dir(row)
        epochs = {FINAL_EPOCH}
        for column in EVENT_EPOCH_COLUMNS:
            value = row.get(column, "")
            if value not in ("", "None"):
                epochs.add(int(float(value)))
        if row["event_id"] in first_ids:
            retreat = int(row["retreat_epoch"])
            epochs |= {retreat + tau for tau in APPROACH_WINDOW}
        for epoch in epochs:
            if FIRST_STORED_EPOCH <= epoch <= FINAL_EPOCH:
                needed[(seed, epoch)] = seed_dir
    return needed


def event_sign_tables() -> None:
    """Sign tables over all 2,048 inputs at every checkpoint the retreat analyses use (Sec. 8).

    A sign table is the function a checkpoint represents: one forward pass
    over the full binary space gives sign(logit_j(x)) for every input x and
    output j.
    The retreat analyses work from these tables and never open a checkpoint.

    Each file holds ``signs``, int8 (2048, 11): the RAW output sign, +1 for a
    logit >= 0, with the identically-zero output position 0 dropped, so column
    j - 1 is output position j (the layout ``common.load_sign_table`` reads).
    The target gauge is not applied: ``retreat_pair_correlations`` and
    ``conditional_function_splitting`` form sign(logit) * t themselves to get
    w_star, and a gauged table would be gauged twice there.

    Reads   results/retreat_events/events.csv (``retreat_events``) and the
            checkpoints data/large_data_trajectories/checkpoints/seed<k>/ep<e>.pt
    Writes  results/event_sign_tables/seedSS_epEEE.npz, 322 files.
    About a minute.
    """
    _codes, all_inputs, _targets = full_binary_space()
    rows = all_events()
    needed = required_checkpoints(rows)
    EVENT_SIGN_TABLES.mkdir(parents=True, exist_ok=True)
    print(f"checkpoints required: {len(needed)}", flush=True)

    model = large_data_network()
    start = time.time()
    written = 0
    missing: list[tuple[int, int]] = []
    for (seed, epoch), seed_dir in sorted(needed.items()):
        source = seed_dir / f"ep{epoch}.pt"
        if not source.exists():
            missing.append((seed, epoch))
            continue
        sign_table = output_signs(model, source, all_inputs)[:, 1:]
        np.savez_compressed(sign_table_path(seed, epoch), signs=sign_table)
        written += 1
        if written % 50 == 0:
            print(f"  {written} written  {time.time() - start:.0f}s", flush=True)
    print(f"done: {written} written, {len(missing)} missing .pt in {time.time() - start:.0f}s", flush=True)
    if missing:
        print(f"  missing: {missing}", flush=True)


# ==========================================================================
# training_trajectories: K_f and Gamma_1 along training (Fig. 3, Secs. 4.2, 4.3)
# ==========================================================================

TRAJECTORY_OUT = RESULTS / "training_trajectories"
SMALL_DATA = SMALL_DATA_TRAJECTORIES / "trajectories.json"
EXAMPLE_SEEDS = (20, 14)                     # the highlighted runs of Fig. 3B
BEHAVIOR_WINDOW = (0.60, 0.80)               # A_block window of the robustness estimate


def long_range_state(signs: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """K_f and Gamma_1 of one network, each averaged over the four long-range positions."""
    observables = [rank_and_coherence(signs[:, position] * target[:, position]) for position in LONG_RANGE_POSITIONS]
    return float(np.mean([item[0] for item in observables])), float(np.mean([item[1] for item in observables]))


def small_data_rows() -> list[dict]:
    """One row per run and saved step of the 24 N = 64 runs; writes ``small_data.csv``.

    The state at each saved step is stored in the record itself; ``A_block``
    is the held-out block accuracy A_blk(S) of Eq. (17).  ``stage`` is
    ``interpolation`` from the first saved step with training block accuracy
    1 on.
    """
    payload = json.loads(SMALL_DATA.read_text())
    rows: list[dict] = []
    for run in payload["runs"]:
        interpolated = False
        for item in run["history"]:
            interpolated = interpolated or float(item["training_block_accuracy"]) >= 1.0
            rows.append({
                "dataset_size": 64,
                "seed": int(run["seed"]),
                "epoch": "",
                "saved_step": int(item["step"]),
                "A_block": float(item["A_block"]),
                "heldout_bit_accuracy": float(item["heldout_bit_accuracy"]),
                "training_accuracy": float(item["training_bit_accuracy"]),
                "training_bit_accuracy": float(item["training_bit_accuracy"]),
                "training_block_accuracy": float(item["training_block_accuracy"]),
                "K_f": float(item["K_f"]),
                "K_f_excess": float(item["K_f_excess"]),
                "Gamma_1": float(item["Gamma_1"]),
                "D_SGD": "",
                "stage": "interpolation" if interpolated else "pre-interpolation",
                "checkpoint_available": True,
                "source_checkpoint": str(SMALL_DATA.relative_to(ROOT)),
            })
    write_rows(TRAJECTORY_OUT / "small_data.csv", rows)
    return rows


def interpolation_summary(rows: list[dict]) -> dict:
    """Medians over the 24 N = 64 runs at the first saved step with training block accuracy 1 (Sec. 4.2)."""
    table = pd.DataFrame(rows)
    first = (
        table[table.training_block_accuracy >= 1.0]
        .sort_values(["seed", "saved_step"])
        .groupby("seed", as_index=False)
        .first()
    )
    if len(first) != 24:
        raise RuntimeError(f"only {len(first)} seeds reached interpolation")
    return {
        "n_seeds": len(first),
        "K_f_median": float(first.K_f.median()),
        "Gamma_1_median": float(first.Gamma_1.median()),
        "A_block_median": float(first.A_block.median()),
        "interpolation_step_median": float(first.saved_step.median()),
    }


def seed_checkpoints(events: list[dict[str, str]]) -> dict[tuple[int, int], Path]:
    """Map (seed, epoch) to the stored weights, for the seeds of the given events."""
    index: dict[tuple[int, int], Path] = {}
    for event in events:
        run_root = (ROOT / event["acquisition_checkpoint"]).parents[2]
        seed = int(event["seed"])
        for path in sorted((run_root / "checkpoints" / f"seed{seed}").glob("ep*.pt")):
            match = CHECKPOINT_PATTERN.search(str(path))
            if match is None:
                continue
            index[(seed, int(match.group("epoch")))] = path
    return index


def large_data_to_onset() -> tuple[list[dict], dict]:
    """State of every N = 1,280 run with an event at each stored epoch from 20 to its onset.

    Onset is the ``acquisition_epoch`` of the first event of the run.
    ``A_block`` is the monitoring block accuracy of that epoch.  Writes
    ``large_data_to_onset.csv``; returns the rows and the coverage record.
    """
    events = first_events()
    event_by_seed = {int(row["seed"]): row for row in events}
    checkpoint_index = seed_checkpoints(events)
    monitoring_accuracy = {
        (int(row["seed"]), int(row["epoch"])): float(row["block_acc"])
        for row in read_csv(TRAJECTORIES)
    }
    _codes, all_inputs, _targets = full_binary_space()
    target = target_signs()
    model = large_data_network()
    rows: list[dict] = []
    start = time.time()
    for (seed, epoch), path in sorted(checkpoint_index.items()):
        onset = int(event_by_seed[seed]["acquisition_epoch"])
        if epoch < 20 or epoch > onset:
            continue
        k_f, gamma_1 = long_range_state(output_signs(model, path, all_inputs), target)
        rows.append({
            "dataset_size": 1280,
            "seed": seed,
            "epoch": epoch,
            "saved_step": epoch,
            "A_block": monitoring_accuracy[(seed, epoch)],
            "heldout_bit_accuracy": "",
            "training_accuracy": "",
            "training_bit_accuracy": "",
            "training_block_accuracy": "",
            "K_f": k_f,
            "K_f_excess": k_f - 1.0,
            "Gamma_1": gamma_1,
            "D_SGD": "",
            "stage": "pre-acquisition" if epoch < onset else "acquisition",
            "checkpoint_available": True,
            "source_checkpoint": str(path.relative_to(ROOT)),
        })
        if epoch == onset:
            print(f"  seed {seed:02d}  onset {onset:3d}  "
                  f"{len(rows):5d} checkpoints  {time.time() - start:6.1f}s", flush=True)
    write_rows(TRAJECTORY_OUT / "large_data_to_onset.csv", rows)
    coverage = {
        "n_seeds": len(events),
        "n_rows": len(rows),
        "epochs_missing_function_state": {str(seed): list(range(1, 20)) for seed in sorted(event_by_seed)},
        "checkpoint_grid": "integer epochs 20..first acquisition inclusive; no interpolation",
    }
    return rows, coverage


def large_data_full_run(seed: int) -> list[dict]:
    """State of one N = 1,280 run at every stored epoch, onset and beyond; writes ``large_data_seed<k>.csv``.

    These rows draw the highlighted trajectories of Fig. 3B and enter no
    statistic.  The rows follow the sorted file names of the checkpoints.
    """
    event = next(row for row in first_events() if int(row["seed"]) == seed)
    run_root = (ROOT / event["acquisition_checkpoint"]).parents[2]
    paths = []
    for path in sorted((run_root / "checkpoints" / f"seed{seed}").glob("ep*.pt")):
        match = CHECKPOINT_PATTERN.search(str(path))
        if match is not None:
            paths.append((int(match.group("epoch")), path))
    monitoring_accuracy = {
        int(row["epoch"]): float(row["block_acc"])
        for row in read_csv(TRAJECTORIES)
        if int(row["seed"]) == seed
    }
    _codes, all_inputs, _targets = full_binary_space()
    target = target_signs()
    model = large_data_network()
    rows: list[dict] = []
    start = time.time()
    for epoch, path in paths:
        if epoch < 20 or epoch not in monitoring_accuracy:
            continue
        k_f, gamma_1 = long_range_state(output_signs(model, path, all_inputs), target)
        rows.append({
            "dataset_size": 1280,
            "seed": seed,
            "epoch": epoch,
            "saved_step": epoch,
            "A_block": monitoring_accuracy[epoch],
            "K_f": k_f,
            "K_f_excess": k_f - 1.0,
            "Gamma_1": gamma_1,
            "stage": "pre-acquisition" if epoch < int(event["acquisition_epoch"]) else "post-acquisition",
            "checkpoint_available": True,
            "source_checkpoint": str(path.relative_to(ROOT)),
        })
    write_rows(TRAJECTORY_OUT / f"large_data_seed{seed}.csv", rows)
    print(f"  seed {seed:02d}  all saved epochs  {len(rows):5d} checkpoints  "
          f"{time.time() - start:6.1f}s", flush=True)
    return rows


def peak_to_onset(rows: list[dict]) -> dict:
    """Drop of K_f from before onset to onset, per run and over runs (Sec. 4.3); writes ``peak_to_onset.csv``.

    Per run: the checkpoint before onset with the largest K_f, the state at
    onset and the differences.  The robustness estimate uses instead the mean
    state over the checkpoints before onset with 0.60 <= A_block < 0.80.
    Medians, means, positive fractions and bootstrap 95 per cent intervals
    over runs, 5,000 resamples each from one generator, the K_f-maximum
    resamples drawn first.
    """
    by_seed: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_seed[int(row["seed"])].append(row)
    summary_rows: list[dict] = []
    window_low, window_high = BEHAVIOR_WINDOW
    for seed, values in sorted(by_seed.items()):
        values.sort(key=lambda row: int(row["epoch"]))
        pre = values[:-1]
        max_row = max(pre, key=lambda row: float(row["K_f"])) if pre else values[0]
        onset = values[-1]
        window_rows = [row for row in pre if window_low <= float(row["A_block"]) < window_high]
        if not window_rows:
            raise ValueError(
                f"seed {seed} has no checkpoint before onset in "
                f"the A_block window [{window_low}, {window_high})"
            )
        window_kf = float(np.mean([float(row["K_f"]) for row in window_rows]))
        window_gamma = float(np.mean([float(row["Gamma_1"]) for row in window_rows]))
        summary_rows.append({
            "dataset_size": 1280,
            "seed": seed,
            "t_maxK_pre": int(max_row["epoch"]),
            "K_f_max_pre": float(max_row["K_f"]),
            "K_f_excess_max_pre": float(max_row["K_f_excess"]),
            "Gamma_1_at_maxK_pre": float(max_row["Gamma_1"]),
            "t_acquisition": int(onset["epoch"]),
            "A_block_acquisition": float(onset["A_block"]),
            "K_f_acquisition": float(onset["K_f"]),
            "K_f_excess_acquisition": float(onset["K_f_excess"]),
            "Gamma_1_acquisition": float(onset["Gamma_1"]),
            "delta_K_f_compression": float(max_row["K_f"]) - float(onset["K_f"]),
            "delta_excess_compression": float(max_row["K_f_excess"]) - float(onset["K_f_excess"]),
            "delta_Gamma_1": float(onset["Gamma_1"]) - float(max_row["Gamma_1"]),
            "behavior_window_low": window_low,
            "behavior_window_high": window_high,
            "n_behavior_window_checkpoints": len(window_rows),
            "K_f_behavior_window_mean": window_kf,
            "Gamma_1_behavior_window_mean": window_gamma,
            "delta_K_f_window_compression": window_kf - float(onset["K_f"]),
            "delta_Gamma_1_window": float(onset["Gamma_1"]) - window_gamma,
        })
    write_rows(TRAJECTORY_OUT / "peak_to_onset.csv", summary_rows)
    delta = np.array([row["delta_excess_compression"] for row in summary_rows])
    gamma = np.array([row["delta_Gamma_1"] for row in summary_rows])
    window_delta = np.array([row["delta_K_f_window_compression"] for row in summary_rows])
    window_gamma = np.array([row["delta_Gamma_1_window"] for row in summary_rows])
    rng = np.random.default_rng(20260822)
    boot_delta, boot_fraction = [], []
    for _ in range(5000):
        sample = delta[rng.integers(0, len(delta), size=len(delta))]
        boot_delta.append(float(sample.mean()))
        boot_fraction.append(float(np.mean(sample > 0.0)))
    boot_window_delta = []
    boot_window_fraction = []
    for _ in range(5000):
        sample = window_delta[rng.integers(0, len(window_delta), size=len(window_delta))]
        boot_window_delta.append(float(sample.mean()))
        boot_window_fraction.append(float(np.mean(sample > 0.0)))
    return {
        "n_seeds": len(summary_rows),
        "K_f_max_pre_median": float(np.median([row["K_f_max_pre"] for row in summary_rows])),
        "K_f_acquisition_median": float(np.median([row["K_f_acquisition"] for row in summary_rows])),
        "excess_compression_median": float(np.median(delta)),
        "excess_compression_mean": float(delta.mean()),
        "excess_compression_mean_ci95": [float(np.quantile(boot_delta, .025)), float(np.quantile(boot_delta, .975))],
        "excess_compression_positive_fraction": float(np.mean(delta > 0.0)),
        "excess_compression_positive_fraction_ci95": [float(np.quantile(boot_fraction, .025)), float(np.quantile(boot_fraction, .975))],
        "Gamma_1_change_median": float(np.median(gamma)),
        "behavior_window": {
            "A_block": [window_low, window_high],
            "K_f_window_mean_median": float(np.median([row["K_f_behavior_window_mean"] for row in summary_rows])),
            "K_f_acquisition_median": float(np.median([row["K_f_acquisition"] for row in summary_rows])),
            "window_compression_median": float(np.median(window_delta)),
            "window_compression_mean": float(window_delta.mean()),
            "window_compression_mean_ci95": [float(np.quantile(boot_window_delta, .025)), float(np.quantile(boot_window_delta, .975))],
            "window_compression_positive_fraction": float(np.mean(window_delta > 0.0)),
            "window_compression_positive_fraction_ci95": [float(np.quantile(boot_window_fraction, .025)), float(np.quantile(boot_window_fraction, .975))],
            "Gamma_1_window_change_median": float(np.median(window_gamma)),
        },
    }


def training_trajectories() -> None:
    """K_f, Gamma_1 and block accuracy along the N = 64 and N = 1,280 runs (Fig. 3, Secs. 4.2, 4.3).

    Every K_f and Gamma_1 is taken from the sign table of a stored network
    over all 2,048 inputs, across the cut d_L = 5, averaged over the output
    positions 8-11.  Nothing is interpolated between stored checkpoints.

    N = 64 (Fig. 3A, Sec. 4.2): the 24 runs (seeds 500-523) with 52 saved
    steps each, flattened into one table; the state at interpolation is the
    first saved step with training block accuracy 1, and its medians are the
    values of Sec. 4.2.

    N = 1,280 (Fig. 3B, Sec. 4.3): for the 24 runs with an event, every stored
    epoch from 20 to onset; epochs 1-19 have a monitoring accuracy and no
    stored weights.  Seeds 20 and 14 are evaluated at every stored epoch up to
    220 for the highlighted traces.  ``peak_to_onset`` reduces the rows to
    the K_f drop from before onset to onset.

    The per-checkpoint tables share one set of columns, empty where a column
    does not apply (``epoch`` for N = 64, the training accuracies for
    N = 1,280, ``D_SGD`` in both).

    Reads   data/small_data_trajectories/trajectories.json, the stored N = 1,280
            weights,
            results/retreat_events/events.csv and trajectories.csv
    Writes  results/training_trajectories/
              small_data.csv, large_data_to_onset.csv, large_data_seed20.csv,
              large_data_seed14.csv, peak_to_onset.csv, summary.json
              (``interpolation``, ``peak_to_onset``, ``coverage``)
    2,299 forward passes, about 8 minutes.
    """
    TRAJECTORY_OUT.mkdir(parents=True, exist_ok=True)
    small_rows = small_data_rows()
    interpolation = interpolation_summary(small_rows)
    large_rows, large_coverage = large_data_to_onset()
    for seed in EXAMPLE_SEEDS:
        large_data_full_run(seed)
    compression = peak_to_onset(large_rows)
    payload = {
        "interpolation": interpolation,
        "peak_to_onset": compression,
        "coverage": large_coverage,
    }
    (TRAJECTORY_OUT / "summary.json").write_text(json.dumps(payload, indent=2))
    print(f"wrote {TRAJECTORY_OUT.relative_to(ROOT)}")


# ==========================================================================
# output_position_dependence: the three frontiers (Fig. 2, Sec. 4.1)
# ==========================================================================

POSITION_OUT = RESULTS / "output_position_dependence"
N_PROGRESS = 11                           # checkpoints per run, equally spaced in progress 0 to 1
ALL_POSITIONS = tuple(range(1, 12))       # output positions j = 1..11
DIFF_THRESHOLD = 0.25                     # j_diff: K_f^(j) at least this above its epoch-20 value
TARGET_KF_MAX = 1.25                      # j_target: K_f^(j) <= 1.25 and ...
TARGET_WSTAR_MIN = 0.80                   # ... w_star^(j) >= 0.80
ACC_THRESHOLD = 0.95                      # j_acc: A^(j) >= 0.95


def progress_epochs(onset: int) -> list[int]:
    """``N_PROGRESS`` epochs equally spaced in normalized progress from epoch 20 (0) to onset (1), rounded."""
    progress = np.linspace(0.0, 1.0, N_PROGRESS)
    epochs = np.rint(FIRST_STORED_EPOCH + progress * (onset - FIRST_STORED_EPOCH)).astype(int)
    return [int(e) for e in epochs]


def position_seed_means(payload: dict[str, np.ndarray]) -> None:
    """Write ``seed_means.csv``: seed means of K_f, its change since progress 0, w_star and accuracy, six decimals."""
    kf = payload["K_f"].mean(axis=0)
    delta = kf - kf[0]
    wstar = payload["w_star"].mean(axis=0)
    accuracy = payload["accuracy"].mean(axis=0)
    with (POSITION_OUT / "seed_means.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["progress", "position", "K_f", "delta_K_f", "w_star", "accuracy"])
        for ri, progress in enumerate(payload["progress"]):
            for pi, position in enumerate(payload["positions"]):
                writer.writerow([
                    f"{progress:.2f}", int(position),
                    f"{kf[ri, pi]:.6f}", f"{delta[ri, pi]:.6f}",
                    f"{wstar[ri, pi]:.6f}", f"{accuracy[ri, pi]:.6f}",
                ])


def frontier(mask: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Largest position satisfying ``mask`` in each row of progress; NaN where none does."""
    out = np.full(mask.shape[0], np.nan)
    for ri in range(mask.shape[0]):
        hits = np.flatnonzero(mask[ri])
        if hits.size:
            out[ri] = positions[hits[-1]]
    return out


def write_frontiers(payload: dict[str, np.ndarray]) -> None:
    """Write ``frontiers.csv`` from the seed means and print the three frontiers."""
    positions = payload["positions"]
    progress = payload["progress"]
    kf = payload["K_f"].mean(axis=0)
    delta = kf - kf[0]
    wstar = payload["w_star"].mean(axis=0)
    accuracy = payload["accuracy"].mean(axis=0)

    names = ("j_diff", "j_target", "j_acc")
    frontiers = (
        frontier(delta >= DIFF_THRESHOLD, positions),
        frontier((kf <= TARGET_KF_MAX) & (wstar >= TARGET_WSTAR_MIN), positions),
        frontier(accuracy >= ACC_THRESHOLD, positions),
    )
    with (POSITION_OUT / "frontiers.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["progress", *names])
        for ri, point in enumerate(progress):
            writer.writerow([f"{point:.2f}"] + ["" if np.isnan(line[ri]) else int(line[ri]) for line in frontiers])
    for name, line in zip(names, frontiers):
        rendered = ", ".join("-" if np.isnan(v) else str(int(v)) for v in line)
        print(f"  {name:<9} over normalized progress: {rendered}")


def output_position_dependence() -> None:
    """Function state and accuracy of each output position between epoch 20 and onset (Fig. 2, Sec. 4.1).

    For each of the 24 N = 1,280 runs with an event, 11 stored checkpoints
    equally spaced in normalized progress from epoch 20 (progress 0) to the
    onset of that run (progress 1) are evaluated on all 2,048 inputs.  At
    output position j = 1..11, the parity of the first j bits:

        K_f^(j)     effective Schmidt rank of the sign table across d_L = 5
        w_star^(j)  target-mode population of the target-gauged sign table
        A^(j)       accuracy at output j over all inputs

    The seed means at equal progress give three frontiers, each the largest
    j that meets a criterion: j_diff (K_f^(j) at least 0.25 above its value
    at epoch 20), j_target (K_f^(j) <= 1.25 and w_star^(j) >= 0.80) and j_acc
    (A^(j) >= 0.95).  Differentiation reaches the long-range outputs before
    they are predicted.

    Reads   results/retreat_events/events.csv and the stored weights it names
    Writes  results/output_position_dependence/
              per_seed.npz    seeds, positions, progress, epochs and K_f,
                              w_star, accuracy of shape (seed, progress, position)
              seed_means.csv  one row per (progress, position)
              frontiers.csv   the three frontiers; empty where no position qualifies
    264 forward passes, about a minute.
    """
    target = target_signs()
    _codes, all_inputs, _targets = full_binary_space()
    events = {int(row["seed"]): row for row in first_events()}
    seeds = sorted(events)
    model = large_data_network()

    shape = (len(seeds), N_PROGRESS, len(ALL_POSITIONS))
    kf = np.full(shape, np.nan)
    wstar = np.full(shape, np.nan)
    accuracy = np.full(shape, np.nan)
    epochs_used = np.zeros((len(seeds), N_PROGRESS), dtype=np.int32)

    start = time.time()
    for si, seed in enumerate(seeds):
        row = events[seed]
        seed_dir = seed_checkpoint_dir(row)
        onset = int(float(row["acquisition_epoch"]))
        for ri, epoch in enumerate(progress_epochs(onset)):
            source = seed_dir / f"ep{epoch}.pt"
            if not source.exists():
                raise SystemExit(f"missing checkpoint: {source}")
            # Truth-gauged sign table with the identically-zero output column
            # dropped: column j-1 holds output position j.
            sign_table = (output_signs(model, source, all_inputs) * target)[:, 1:]
            epochs_used[si, ri] = epoch
            for pi, position in enumerate(ALL_POSITIONS):
                column = sign_table[:, position - 1]
                kf[si, ri, pi] = effective_rank(column)
                wstar[si, ri, pi] = target_population(column)
                accuracy[si, ri, pi] = float(np.mean(column > 0))
        print(f"  seed {seed:02d}  onset {onset:3d}  {time.time() - start:5.1f}s", flush=True)

    payload = {
        "seeds": np.asarray(seeds, dtype=np.int32),
        "positions": np.asarray(ALL_POSITIONS, dtype=np.int32),
        "progress": np.linspace(0.0, 1.0, N_PROGRESS),
        "epochs": epochs_used,
        "K_f": kf,
        "w_star": wstar,
        "accuracy": accuracy,
    }
    POSITION_OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(POSITION_OUT / "per_seed.npz", **payload)
    position_seed_means(payload)
    print(f"wrote {(POSITION_OUT / 'per_seed.npz').relative_to(ROOT)}  ({time.time() - start:.0f}s)")
    write_frontiers(payload)


# ==========================================================================
# state_vs_accuracy: function state at fixed accuracy (Fig. 4, Sec. 4.4, Fig. 3B)
# ==========================================================================

ACCURACY_OUT = RESULTS / "state_vs_accuracy"
LAST_EPOCH = 220
TAU_POINTS = 11
HELD_OUT_TRAINING_N = 1280     # the training inputs whose complement is the held-out set S
WINDOW_COLUMNS = (
    "window", "n_checkpoints",
    "K_f_excess_min", "K_f_excess_max",
    "Gamma_1_min", "Gamma_1_max",
    "w_star_min", "w_star_max",
    "Gamma_1_sd_within_seed", "Gamma_1_sd_between_seeds",
)


def long_range_state_with_target(signs: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    """K_f, Gamma_1 and w_star of one network, each averaged over the four long-range positions."""
    kf: list[float] = []
    gamma: list[float] = []
    wstar: list[float] = []
    for position in LONG_RANGE_POSITIONS:
        sign_table = signs[:, position] * target[:, position]
        k, g = rank_and_coherence(sign_table)
        kf.append(k)
        gamma.append(g)
        wstar.append(target_population(sign_table))
    return float(np.mean(kf)), float(np.mean(gamma)), float(np.mean(wstar))


def all_checkpoint_states() -> dict[str, np.ndarray]:
    """K_f, Gamma_1, w_star and block accuracy of every stored checkpoint, epochs 20 to 220.

    Writes ``checkpoints.npz`` (full precision) and ``checkpoints.csv`` (six
    decimals), whose ``A_block`` is the full-space block accuracy over all
    2,048 inputs and all 12 output positions; ``after_onset`` marks the
    checkpoints at or after the onset epoch of the run.  It also writes
    ``heldout_accuracy.csv``, the held-out block accuracy A_blk(S) of Eq. (17)
    at the same checkpoints (Fig. 3B): S is the set of the 1,085 strings that
    are not among the 1,280 training inputs (963 distinct), and a string counts
    when its D = 11 outputs j = 1, ..., 11 are all right (position 0 has the
    constant target 0 and is not counted).  Full precision, a multiple of
    1/1,085.
    """
    _codes, all_inputs, _targets = full_binary_space()
    target = target_signs()
    events = {int(row["seed"]): row for row in first_events()}
    seeds = sorted(events)
    model = large_data_network()
    held_out = np.ones(N_STRINGS, dtype=bool)
    held_out[training_codes(HELD_OUT_TRAINING_N).numpy()] = False

    records: list[tuple[int, int, float, float, float, float, int]] = []
    heldout_rows: list[dict] = []
    start = time.time()
    for seed in seeds:
        row = events[seed]
        seed_dir = seed_checkpoint_dir(row)
        onset = int(float(row["acquisition_epoch"]))
        for epoch in range(FIRST_STORED_EPOCH, LAST_EPOCH + 1):
            source = seed_dir / f"ep{epoch}.pt"
            if not source.exists():
                continue
            signs = output_signs(model, source, all_inputs)
            kf, gamma, wstar = long_range_state_with_target(signs, target)
            accuracy = float(np.mean((signs == target).all(axis=1)))
            records.append((seed, epoch, kf, gamma, wstar, accuracy, int(epoch >= onset)))
            all_right = (signs[:, 1:] == target[:, 1:]).all(axis=1)
            heldout_rows.append({"seed": seed, "epoch": epoch,
                                 "A_block_heldout": float(np.mean(all_right[held_out]))})
        print(f"  seed {seed:02d}  onset {onset:3d}  {len(records):5d} checkpoints  "
              f"{time.time() - start:6.1f}s", flush=True)

    array = np.asarray(records, dtype=np.float64)
    payload = {
        "seed": array[:, 0].astype(np.int32),
        "epoch": array[:, 1].astype(np.int32),
        "K_f": array[:, 2],
        "Gamma_1": array[:, 3],
        "w_star": array[:, 4],
        "A_block": array[:, 5],
        "after_onset": array[:, 6].astype(np.int8),
    }
    ACCURACY_OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ACCURACY_OUT / "checkpoints.npz", **payload)
    with (ACCURACY_OUT / "checkpoints.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "epoch", "A_block", "K_f", "K_f_excess", "Gamma_1", "w_star", "after_onset"])
        for seed, epoch, kf, gamma, wstar, accuracy, after in records:
            writer.writerow([
                seed, epoch, f"{accuracy:.6f}", f"{kf:.6f}", f"{kf - 1.0:.6f}",
                f"{gamma:.6f}", f"{wstar:.6f}", after,
            ])
    write_rows(ACCURACY_OUT / "heldout_accuracy.csv", heldout_rows)
    print(f"wrote {(ACCURACY_OUT / 'checkpoints.npz').relative_to(ROOT)}  "
          f"({len(records)} checkpoints, {time.time() - start:.0f}s)")
    return payload


def accuracy_windows(payload: dict[str, np.ndarray]) -> None:
    """Ranges of the state at fixed accuracy (Sec. 4.4); writes ``accuracy_windows.csv`` and prints them.

    Minimum and maximum of K_f - 1, Gamma_1 and w_star over the checkpoints
    with 0.18 <= A_block < 0.22, with 0.88 <= A_block < 0.92 and with
    A_block > 0.99.  For the last window the spread of Gamma_1 is split into
    the mean over runs of the standard deviation inside a run and the
    standard deviation of the run means, both with the n - 1 normalization.
    Also prints the means at onset and at epoch 20.
    """
    accuracy = payload["A_block"]
    excess = payload["K_f"] - 1.0
    gamma = payload["Gamma_1"]
    wstar = payload["w_star"]
    table: list[dict] = []

    def ranges(mask: np.ndarray, label: str) -> dict:
        row = dict.fromkeys(WINDOW_COLUMNS, "")
        row["window"] = label
        row["n_checkpoints"] = int(mask.sum())
        if mask.any():
            for name, values in (("K_f_excess", excess), ("Gamma_1", gamma), ("w_star", wstar)):
                row[f"{name}_min"] = float(values[mask].min())
                row[f"{name}_max"] = float(values[mask].max())
        table.append(row)
        return row

    def window(low: float, high: float, label: str) -> None:
        mask = (accuracy >= low) & (accuracy < high)
        row = ranges(mask, f"{low:.2f}<=A_block<{high:.2f}")
        if not mask.any():
            print(f"  {label}: no checkpoints")
            return
        print(
            f"  {label} (n={row['n_checkpoints']}): "
            f"K_f-1 {row['K_f_excess_min']:.2f}-{row['K_f_excess_max']:.2f}   "
            f"Gamma_1 {row['Gamma_1_min']:.3f}-{row['Gamma_1_max']:.3f}   "
            f"w_star {row['w_star_min']:.3f}-{row['w_star_max']:.3f}"
        )

    print("ranges at fixed block accuracy:")
    window(0.18, 0.22, "0.18<=A<0.22 ")
    window(0.88, 0.92, "0.88<=A<0.92 ")

    high = accuracy > 0.99
    row = ranges(high, "A_block>0.99")
    if high.any():
        print(
            f"  A>0.99       (n={row['n_checkpoints']}): "
            f"K_f-1 max {row['K_f_excess_max']:.3f}   "
            f"w_star {row['w_star_min']:.3f}-{row['w_star_max']:.3f}   "
            f"Gamma_1 {row['Gamma_1_min']:.3f}-{row['Gamma_1_max']:.3f}"
        )
        seeds = payload["seed"][high]
        within = [
            float(np.std(gamma[high][seeds == s], ddof=1))
            for s in np.unique(seeds)
            if (seeds == s).sum() > 1
        ]
        between = [float(np.mean(gamma[high][seeds == s])) for s in np.unique(seeds)]
        row["Gamma_1_sd_within_seed"] = float(np.mean(within))
        row["Gamma_1_sd_between_seeds"] = float(np.std(between, ddof=1))
        print(
            f"  Gamma_1 spread over A>0.99: within-seed sd "
            f"{row['Gamma_1_sd_within_seed']:.3f}, between-seed sd "
            f"{row['Gamma_1_sd_between_seeds']:.3f}"
        )

    with (ACCURACY_OUT / "accuracy_windows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=WINDOW_COLUMNS)
        writer.writeheader()
        writer.writerows(table)

    onset = payload["after_onset"].astype(bool)
    first_after = []
    for s in np.unique(payload["seed"]):
        mask = (payload["seed"] == s) & onset
        if mask.any():
            index = np.flatnonzero(mask)[0]
            first_after.append((payload["K_f"][index], gamma[index]))
    if first_after:
        values = np.asarray(first_after)
        print(f"  at onset: mean K_f {values[:, 0].mean():.2f}   mean Gamma_1 {values[:, 1].mean():.3f}")
    start = payload["epoch"] == FIRST_STORED_EPOCH
    if start.any():
        print(f"  at epoch 20: mean Gamma_1 {gamma[start].mean():.3f}")


def onset_aligned(payload: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """K_f - 1 and A_block of every run on the onset-aligned axis tau (inset of Fig. 3B).

    Each run reaches onset at its own epoch, so the epoch axis cannot be
    averaged directly.  tau runs from 0 at epoch 20 to 1 at the onset of the
    run; at each of 11 values the nearest stored epoch of the run is taken.
    Returns tau, the two arrays of shape (runs, 11) and the number of runs.
    """
    seed, epoch = payload["seed"], payload["epoch"]
    K_f, A_block = payload["K_f"], payload["A_block"]
    onset = {int(row["seed"]): int(row["acquisition_epoch"]) for row in first_events()}

    tau = np.linspace(0.0, 1.0, TAU_POINTS)
    excess, accuracy = [], []
    for model in sorted(set(seed.tolist())):
        if model not in onset:
            continue
        mask = seed == model
        epochs, ranks, accuracies = epoch[mask], K_f[mask], A_block[mask]
        wanted = np.round(FIRST_STORED_EPOCH + tau * (onset[model] - FIRST_STORED_EPOCH)).astype(int)
        index = [int(np.argmin(np.abs(epochs - value))) for value in wanted]
        excess.append(ranks[index] - 1.0)
        accuracy.append(accuracies[index])
    return tau, np.asarray(excess), np.asarray(accuracy), len(excess)


def onset_aligned_means(payload: dict[str, np.ndarray]) -> None:
    """Write ``onset_aligned_means.csv`` (mean and quartiles over runs, full precision) and print the means."""
    tau, excess, accuracy, n_seeds = onset_aligned(payload)
    columns = (
        tau,
        excess.mean(axis=0),
        np.quantile(excess, 0.25, axis=0), np.quantile(excess, 0.75, axis=0),
        accuracy.mean(axis=0),
        np.quantile(accuracy, 0.25, axis=0), np.quantile(accuracy, 0.75, axis=0),
    )
    with (ACCURACY_OUT / "onset_aligned_means.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "tau", "K_f_excess_mean", "K_f_excess_q25", "K_f_excess_q75",
            "A_block_mean", "A_block_q25", "A_block_q75", "n_seeds",
        ])
        # Full-precision values: the inset of Fig. 3B is drawn from this table.
        for values in zip(*columns):
            writer.writerow([float(value) for value in values] + [n_seeds])
    print(f"onset-aligned mean over {n_seeds} runs")
    print("  tau    K_f-1   A_blk")
    for point, rank, value in zip(tau, excess.mean(axis=0), accuracy.mean(axis=0)):
        print(f"  {point:.1f}   {rank:6.3f}   {value:6.3f}")


def onset_aligned_summary(payload: dict[str, np.ndarray]) -> dict:
    """Means over runs on the onset-aligned axis and the K_f peak of each run (Sec. 4.3).

    Each run from epoch 20 to its onset epoch is placed on
    tau = (epoch - 20) / (onset - 20) and interpolated linearly in tau.  The
    mean over runs is then piecewise linear with corners at the stored tau of
    the runs, so it is evaluated on the union of those tau and its maximum is
    exact.  Reported:

        K_f_mean               the mean K_f at tau = 0, its maximum, the tau of
                               the maximum, and the mean K_f at tau = 1
        A_blk_heldout_mean     the mean held-out A_blk(S) at the same three tau
        A_blk_heldout_rises    the number of runs in which A_blk(S) rises from
                               tau = 0 to the tau of the maximum, and from there
                               to tau = 1
        K_f_rises_then_falls   the number of runs whose K_f exceeds its epoch-20
                               value at a later epoch and is below its largest
                               value at onset
        peaks                  per run the epoch from 20 to onset with the
                               largest K_f: the range and median of its tau, the
                               mean of those K_f, the means of Gamma_1 there and
                               at onset, the mean K_f at onset, and the numbers
                               of runs in which K_f falls and Gamma_1 rises from
                               the peak to onset

    Writes ``onset_aligned_summary.json``.
    """
    heldout = {
        (int(row["seed"]), int(row["epoch"])): float(row["A_block_heldout"])
        for row in csv.DictReader((ACCURACY_OUT / "heldout_accuracy.csv").open(newline="", encoding="utf-8"))
    }
    onset = {int(row["seed"]): int(row["acquisition_epoch"]) for row in first_events()}
    runs = []
    for model in sorted(set(payload["seed"].tolist())):
        if model not in onset:
            continue
        mask = (payload["seed"] == model) & (payload["epoch"] <= onset[model])
        order = np.argsort(payload["epoch"][mask])
        epochs = payload["epoch"][mask][order].astype(int)
        runs.append({
            "tau": (epochs - FIRST_STORED_EPOCH) / (onset[model] - FIRST_STORED_EPOCH),
            "K_f": payload["K_f"][mask][order],
            "Gamma_1": payload["Gamma_1"][mask][order],
            "A_blk": np.asarray([heldout[(model, int(epoch))] for epoch in epochs]),
        })
    grid = np.unique(np.concatenate([run["tau"] for run in runs]))

    def on_grid(key: str) -> np.ndarray:
        return np.asarray([np.interp(grid, run["tau"], run[key]) for run in runs])

    rank, accuracy = on_grid("K_f"), on_grid("A_blk")
    peak = int(np.argmax(rank.mean(axis=0)))
    at = (0, peak, len(grid) - 1)
    largest = [int(np.argmax(run["K_f"])) for run in runs]
    summary = {
        "n_runs": len(runs),
        "K_f_mean": {"tau_0": float(rank.mean(axis=0)[0]), "max": float(rank.mean(axis=0)[peak]),
                     "tau_at_max": float(grid[peak]), "tau_1": float(rank.mean(axis=0)[-1])},
        "A_blk_heldout_mean": dict(zip(("tau_0", "tau_at_max", "tau_1"),
                                       (float(accuracy.mean(axis=0)[index]) for index in at))),
        "A_blk_heldout_rises": {"tau_0_to_tau_at_max": int((accuracy[:, peak] > accuracy[:, 0]).sum()),
                                "tau_at_max_to_tau_1": int((accuracy[:, -1] > accuracy[:, peak]).sum())},
        "K_f_rises_then_falls": int(sum(
            bool((run["K_f"][1:] > run["K_f"][0]).any()) and bool(run["K_f"][-1] < run["K_f"].max())
            for run in runs)),
        "peaks": {
            "tau_min": float(min(run["tau"][index] for run, index in zip(runs, largest))),
            "tau_max": float(max(run["tau"][index] for run, index in zip(runs, largest))),
            "tau_median": float(np.median([run["tau"][index] for run, index in zip(runs, largest)])),
            "K_f_mean": float(np.mean([run["K_f"][index] for run, index in zip(runs, largest)])),
            "Gamma_1_mean": float(np.mean([run["Gamma_1"][index] for run, index in zip(runs, largest)])),
            "Gamma_1_mean_at_onset": float(np.mean([run["Gamma_1"][-1] for run in runs])),
            "K_f_mean_at_onset": float(np.mean([run["K_f"][-1] for run in runs])),
            "K_f_falls": int(sum(run["K_f"][-1] < run["K_f"][index] for run, index in zip(runs, largest))),
            "Gamma_1_rises": int(sum(run["Gamma_1"][-1] > run["Gamma_1"][index]
                                     for run, index in zip(runs, largest))),
        },
    }
    (ACCURACY_OUT / "onset_aligned_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    mean_rank, mean_accuracy = summary["K_f_mean"], summary["A_blk_heldout_mean"]
    print(f"onset-aligned means over {len(runs)} runs: K_f {mean_rank['tau_0']:.2f} -> "
          f"{mean_rank['max']:.2f} at tau {mean_rank['tau_at_max']:.2f} -> {mean_rank['tau_1']:.2f}; "
          f"held-out A_blk {mean_accuracy['tau_0']:.3f} -> {mean_accuracy['tau_at_max']:.3f} -> "
          f"{mean_accuracy['tau_1']:.3f}")
    return summary


def state_vs_accuracy() -> None:
    """Function state against block accuracy at every stored N = 1,280 checkpoint (Figs. 3B, 4, Secs. 4.3, 4.4).

    For each of the 24 runs with an event, every stored checkpoint from epoch
    20 to 220 is evaluated on all 2,048 inputs: K_f and Gamma_1 of the sign
    table, w_star of the target-gauged table (the quantity that separates a
    return to the target mode from a return to any single mode), each
    averaged over positions 8-11, the full-space block accuracy A_block,
    the fraction of inputs whose whole output string is correct, and the
    held-out block accuracy A_blk(S) of Eq. (17) on the 1,085 strings
    outside the training set.  Many function states share one accuracy;
    ``accuracy_windows`` gives their ranges, ``onset_aligned_means`` the
    means over runs against progress to onset (the K_f - 1 curve of the inset of Fig. 3B) and
    ``onset_aligned_summary`` the numbers of Sec. 4.3.  The held-out
    accuracies are the accuracy curves and the inset accuracy of Fig. 3B.

    Reads   results/retreat_events/events.csv and the stored weights it names
    Writes  results/state_vs_accuracy/
              checkpoints.npz, checkpoints.csv, heldout_accuracy.csv,
              accuracy_windows.csv, onset_aligned_means.csv,
              onset_aligned_summary.json
    4,824 forward passes, about 17 minutes.
    """
    payload = all_checkpoint_states()
    accuracy_windows(payload)
    onset_aligned_means(payload)
    onset_aligned_summary(payload)


# ==========================================================================
# largest_addition_branches: the branches at the largest addition (Fig. 5)
# ==========================================================================

LARGEST_ADDITION_OUT = RESULTS / "largest_addition_branches"
LARGEST_ADDITION = 1216
DISTINCT = f"random_plus{LARGEST_ADDITION}"
DUPLICATE = f"duplicate_plus{LARGEST_ADDITION}"
CONTROL = "continuation"
# name in the tables of data/branching_study/, name in the tables written here
LARGEST_ADDITION_BRANCHES = ((DISTINCT, "distinct"), (DUPLICATE, "duplicate"), (CONTROL, "control"))
EARLY_UPDATE = 25
# field written here, column of checkpoints_added1216.csv
CONTRAST_FIELDS = (("K_f", "K_f"), ("Gamma_1", "Gamma_1"), ("A_block", "full_A_block"))
SUBSET_FIELDS = (("A_block_added", "injected_A_block"), ("A_block_untrained", "unseen_A_block"))
# branch and update of the four spectra of panel B; the branch point is the
# distinct branch at update 0, whose sign table every branch shares
SPECTRUM_POINTS = (("distinct", 0), ("distinct", EARLY_UPDATE), ("distinct", ENDPOINT_UPDATE),
                   ("duplicate", ENDPOINT_UPDATE))


def branch_array(table: pd.DataFrame, branch: str, field: str, seeds: list[int], steps: list[int]) -> np.ndarray:
    """One field of one branch as an array (starting model, update), rows in ``seeds`` order."""
    return (table[table.arm == branch].pivot(index="seed", columns="step", values=field)
            .loc[seeds, steps].to_numpy(float))


def rho_spectrum(sign_tables: np.ndarray) -> np.ndarray:
    """Eigenvalues of rho_A in descending order, averaged over the long-range positions.

    ``sign_tables`` holds one sign table per long-range position.  The
    eigenvalues of each reduced state are clipped at zero and normalized to
    unit sum before the average.
    """
    spectra = []
    for sign_table in sign_tables:
        values = np.clip(np.linalg.eigvalsh(reduced_state(sign_table)), 0.0, None)
        values /= values.sum()
        spectra.append(values[::-1])
    return np.mean(spectra, axis=0)


def largest_addition_branches() -> None:
    """The three branches of each starting model at the largest addition (Fig. 5, Sec. 5.1).

    The branches leave one starting model and share the optimizer protocol,
    the number of updates and the minibatch stream; they differ only in the
    data added to the 64 base inputs: 1,216 distinct inputs, 1,216 repeated
    base inputs, or nothing.  The four panels:

        A  K_f - 1 along the updates, one trace per starting model and branch,
           and the mean over starting models: the distinct and duplicate
           branches at the 11 updates of checkpoints_added1216.csv, the control
           branch at the 7 updates of trajectory_metrics.csv
        B  the spectrum of rho_A (from the sign tables, averaged over the four
           long-range positions) averaged over starting models: at the branch
           point, for the distinct branch at update 25, and for the distinct
           and the duplicate branch at update 800
        C  distinct-minus-duplicate contrasts in K_f, Gamma_1 and A_block,
           formed within a starting model, with the interquartile range
        D  block accuracy of the distinct branch on the 1,216 inputs it added
           and on the 768 inputs it never trains on

    Means and quartiles are taken over the 25 starting models, in ascending
    seed order.

    Reads   data/branching_study/checkpoints_added1216.csv,
            trajectory_metrics.csv (the control branch), checkpoint_sign_tables.npz
    Writes  results/largest_addition_branches/
              branch_traces.csv     arm, seed, step, K_f_exc (panel A, drawing order)
              branch_means.csv      arm, field, step, mean, q25, q75
              spectra.csv           arm, step, rank, eigenvalue (panel B)
              paired_contrasts.csv  field, step, mean, q25, q75, n_positive, n
    and prints the endpoint means, the contrasts at update 25 and the subset
    accuracies of the distinct branch.
    """
    added = pd.read_csv(BRANCHING_STUDY / "checkpoints_added1216.csv")
    records = pd.read_csv(BRANCHING_STUDY / "trajectory_metrics.csv")
    control = records[records.arm == CONTROL]
    seeds = sorted(int(seed) for seed in added.seed.unique())
    steps = sorted(int(step) for step in added.step.unique())
    control_steps = sorted(int(step) for step in control.step.unique())
    LARGEST_ADDITION_OUT.mkdir(parents=True, exist_ok=True)

    def source(branch: str) -> tuple[pd.DataFrame, list[int]]:
        return (control, control_steps) if branch == CONTROL else (added, steps)

    traces = []
    for branch, short in LARGEST_ADDITION_BRANCHES:
        table, branch_steps = source(branch)
        values = branch_array(table, branch, "K_f_exc", seeds, branch_steps)
        for seed, trace in zip(seeds, values):
            for step, value in zip(branch_steps, trace):
                traces.append({"arm": short, "seed": seed, "step": step, "K_f_exc": float(value)})
    write_rows(LARGEST_ADDITION_OUT / "branch_traces.csv", traces)

    means = []
    wanted = [(branch, short, "K_f_exc", "K_f_exc") for branch, short in LARGEST_ADDITION_BRANCHES]
    wanted += [(DISTINCT, "distinct", field, column) for field, column in SUBSET_FIELDS]
    for branch, short, field, column in wanted:
        table, branch_steps = source(branch)
        values = branch_array(table, branch, column, seeds, branch_steps)
        for step, m, lo, hi in zip(branch_steps, values.mean(axis=0), np.quantile(values, 0.25, axis=0),
                                   np.quantile(values, 0.75, axis=0)):
            means.append({"arm": short, "field": field, "step": step,
                          "mean": float(m), "q25": float(lo), "q75": float(hi)})
    write_rows(LARGEST_ADDITION_OUT / "branch_means.csv", means)

    signs, _metrics, sign_seeds = load_checkpoint_sign_tables()
    if sign_seeds != seeds:
        raise SystemExit("the sign tables and checkpoints_added1216.csv list different starting models")
    spectra = []
    for branch, step in SPECTRUM_POINTS:
        b, u = BRANCHES.index(branch), SAVED_UPDATES.index(step)
        values = np.asarray([rho_spectrum(signs[i, b, u]) for i in range(len(seeds))]).mean(axis=0)
        for rank, value in enumerate(values, start=1):
            spectra.append({"arm": branch, "step": step, "rank": rank, "eigenvalue": float(value)})
    write_rows(LARGEST_ADDITION_OUT / "spectra.csv", spectra)

    contrasts = []
    for field, column in CONTRAST_FIELDS:
        values = (branch_array(added, DISTINCT, column, seeds, steps)
                  - branch_array(added, DUPLICATE, column, seeds, steps))
        for index, step in enumerate(steps):
            contrasts.append({
                "field": field,
                "step": step,
                "mean": float(values.mean(axis=0)[index]),
                "q25": float(np.quantile(values, 0.25, axis=0)[index]),
                "q75": float(np.quantile(values, 0.75, axis=0)[index]),
                "n_positive": int((values[:, index] > 0).sum()),
                "n": len(seeds),
            })
    write_rows(LARGEST_ADDITION_OUT / "paired_contrasts.csv", contrasts)

    last = steps[-1]
    print(f"branch point step {steps[0]}, endpoint step {last}, {len(seeds)} starting models")
    for branch, label in LARGEST_ADDITION_BRANCHES:
        table, branch_steps = source(branch)
        values = branch_array(table, branch, "K_f_exc", seeds, branch_steps)[:, -1]
        print(f"  {label:<13} endpoint K_f-1 mean {values.mean():.3f}")
    early = steps.index(EARLY_UPDATE)
    for (field, column), name in zip(CONTRAST_FIELDS, ("K_f", "Gamma_1", "A_blk")):
        values = (branch_array(added, DISTINCT, column, seeds, steps)
                  - branch_array(added, DUPLICATE, column, seeds, steps))[:, early]
        print(f"  update {EARLY_UPDATE} contrast in {name:<8} {values.mean():+.4f} "
              f"({int((values > 0).sum())}/{len(values)} positive)")
    for label, (_field, column) in zip(("added 1,216", "never trained 768"), SUBSET_FIELDS):
        values = branch_array(added, DISTINCT, column, seeds, steps)
        print(f"  distinct branch on {label:<18} update {EARLY_UPDATE}: {values[:, early].mean():.4f}"
              f"   update {last}: {values[:, -1].mean():.4f}")


# ==========================================================================
# branch_contrasts: contrasts along the branches (Sec. 5.1, App. B.1)
# ==========================================================================

BRANCH_CONTRASTS_OUT = RESULTS / "branch_contrasts"
BRANCH_ARMS = {"random": "distinct", "duplicate": "duplicate"}   # arm prefix, branch


def branching_rows(name: str) -> list[dict[str, str]]:
    """Rows of one table of data/branching_study/."""
    with (BRANCHING_STUDY / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def branch_contrast(distinct: dict[int, dict[str, str]], duplicate: dict[int, dict[str, str]],
                    column: str, name: str) -> dict[str, float]:
    """Branch means and distinct-minus-duplicate contrast of one column, paired by starting model."""
    seeds = sorted(set(distinct) & set(duplicate))
    a = np.asarray([float(distinct[seed][column]) for seed in seeds])
    b = np.asarray([float(duplicate[seed][column]) for seed in seeds])
    delta = a - b
    return {
        f"{name}_distinct_mean": float(a.mean()),
        f"{name}_duplicate_mean": float(b.mean()),
        f"delta_{name}_mean": float(delta.mean()),
        f"delta_{name}_negative": int((delta < 0).sum()),
        f"delta_{name}_positive": int((delta > 0).sum()),
    }


def checkpoint_contrasts() -> list[dict]:
    """Every addition at the seven updates of ``trajectory_metrics.csv`` (K_f and A_block)."""
    cells: dict[tuple[int, int, str], dict[int, dict[str, str]]] = defaultdict(dict)
    for row in branching_rows("trajectory_metrics.csv"):
        prefix, _, addition = row["arm"].partition("_plus")
        if prefix not in BRANCH_ARMS:
            continue
        cells[(int(addition), int(row["step"]), BRANCH_ARMS[prefix])][int(row["seed"])] = row
    rows = []
    for addition in sorted({key[0] for key in cells}):
        for update in sorted({key[1] for key in cells}):
            distinct = cells[(addition, update, "distinct")]
            duplicate = cells[(addition, update, "duplicate")]
            rows.append({
                "dose": addition,
                "update": update,
                "n_starts": len(set(distinct) & set(duplicate)),
                **branch_contrast(distinct, duplicate, "K_f", "K_f"),
                **branch_contrast(distinct, duplicate, "A_block", "A_block"),
            })
    return rows


def added1216_contrasts() -> list[dict]:
    """The addition of 1,216 inputs at the eleven updates of ``checkpoints_added1216.csv``."""
    cells: dict[tuple[int, str], dict[int, dict[str, str]]] = defaultdict(dict)
    for row in branching_rows("checkpoints_added1216.csv"):
        cells[(int(row["step"]), row["arm"])][int(row["seed"])] = row
    rows = []
    for update in sorted({key[0] for key in cells}):
        distinct = cells[(update, f"random_plus{LARGEST_ADDITION}")]
        duplicate = cells[(update, f"duplicate_plus{LARGEST_ADDITION}")]
        seeds = sorted(set(distinct) & set(duplicate))
        rows.append({
            "update": update,
            "n_starts": len(seeds),
            **branch_contrast(distinct, duplicate, "K_f", "K_f"),
            **branch_contrast(distinct, duplicate, "Gamma_1", "Gamma_1"),
            **branch_contrast(distinct, duplicate, "full_A_block", "A_block"),
            "A_block_added_distinct_mean": float(np.mean([float(distinct[seed]["injected_A_block"]) for seed in seeds])),
            "A_block_untrained_distinct_mean": float(np.mean([float(distinct[seed]["unseen_A_block"]) for seed in seeds])),
        })
    return rows


def ordering_contrasts() -> list[dict]:
    """K_f contrast at update 800 for every addition and ordering of ``independent_orderings.csv``."""
    cells: dict[tuple[int, int, str], dict[int, float]] = defaultdict(dict)
    for row in branching_rows("independent_orderings.csv"):
        family, addition = row["arm"].split("_plus")
        cells[(int(addition), int(row["pool_seed"]), family)][int(row["seed"])] = float(row["K_f_end"])
    rows = []
    for addition in sorted({key[0] for key in cells}):
        for ordering in sorted({key[1] for key in cells}):
            distinct = cells[(addition, ordering, "random")]
            duplicate = cells[(addition, ordering, "duplicate")]
            seeds = sorted(set(distinct) & set(duplicate))
            delta = np.asarray([distinct[seed] - duplicate[seed] for seed in seeds])
            rows.append({
                "dose": addition,
                "ordering_seed": ordering,
                "n_starts": len(seeds),
                "delta_K_f_mean": float(delta.mean()),
                "delta_K_f_max": float(delta.max()),
                "delta_K_f_negative": int((delta < 0).sum()),
            })
    return rows


def branch_contrasts() -> None:
    """Contrasts along the branches, the numbers of Sec. 5.1 and App. B.1.

    In every table a contrast is distinct minus duplicate, within a starting
    model, then averaged over starting models.

        trajectory_metrics.csv     starting model x addition (16 to 1,216) x branch
                                   x update (0, 25, 50, 100, 200, 400, 800): K_f and
                                   A_block on all inputs.  At update 0 both
                                   branches are the starting model: the branch-point
                                   means of Sec. 5.1.
        checkpoints_added1216.csv  starting model x branch x the 11 saved updates for
                                   +1,216: K_f, Gamma_1, A_block on all inputs, on
                                   the 1,216 added and on the 768 never-trained
                                   inputs (Sec. 5.1 quotes update 25).
        independent_orderings.csv  8 starting models x 3 orderings of the inputs
                                   outside the base set x (192, 576, 1,216) x
                                   branch at update 800 (App. B.1: number of
                                   negative K_f contrasts among the 24 pairs).

    Reads   data/branching_study/{trajectory_metrics, checkpoints_added1216,
            independent_orderings}.csv
    Writes  results/branch_contrasts/
              checkpoint_contrasts.csv   per addition and update
              added1216_contrasts.csv    per update
              independent_orderings.csv  per addition and ordering: mean and
                                         largest K_f contrast, number negative
    and prints the quoted values.
    """
    checkpoints = checkpoint_contrasts()
    added = added1216_contrasts()
    orderings = ordering_contrasts()
    BRANCH_CONTRASTS_OUT.mkdir(parents=True, exist_ok=True)
    write_rows(BRANCH_CONTRASTS_OUT / "checkpoint_contrasts.csv", checkpoints)
    write_rows(BRANCH_CONTRASTS_OUT / "added1216_contrasts.csv", added)
    write_rows(BRANCH_CONTRASTS_OUT / "independent_orderings.csv", orderings)

    start = next(r for r in checkpoints if r["dose"] == LARGEST_ADDITION and r["update"] == 0)
    early = next(r for r in checkpoints if r["dose"] == LARGEST_ADDITION and r["update"] == EARLY_UPDATE)
    print(f"trajectory_metrics.csv, {start['n_starts']} starting models")
    print(f"  branch points (update 0): K_f {start['K_f_distinct_mean']:.3f}, "
          f"A_blk {start['A_block_distinct_mean']:.3f}")
    print(f"  +{LARGEST_ADDITION}, update {EARLY_UPDATE}, distinct - duplicate: "
          f"K_f {early['delta_K_f_mean']:+.3f}, A_blk {early['delta_A_block_mean']:+.3f}")
    early = next(r for r in added if r["update"] == EARLY_UPDATE)
    untrained = N_STRINGS - BASE_N - LARGEST_ADDITION
    print(f"checkpoints_added1216.csv, {early['n_starts']} starting models")
    print(f"  update {EARLY_UPDATE}, distinct - duplicate: "
          f"K_f {early['delta_K_f_mean']:+.3f}, Gamma_1 {early['delta_Gamma_1_mean']:+.3f}, "
          f"A_blk {early['delta_A_block_mean']:+.3f}")
    print(f"  update {EARLY_UPDATE}, distinct branch: "
          f"A_blk {early['A_block_added_distinct_mean']:.3f} on the {LARGEST_ADDITION:,} added inputs, "
          f"{early['A_block_untrained_distinct_mean']:.3f} on the {untrained} never trained")
    n_orderings = len({r["ordering_seed"] for r in orderings})
    print(f"independent_orderings.csv, {n_orderings} orderings, update {ENDPOINT_UPDATE}")
    for addition in sorted({r["dose"] for r in orderings}):
        selected = [r for r in orderings if r["dose"] == addition]
        negative = sum(r["delta_K_f_negative"] for r in selected)
        total = sum(r["n_starts"] for r in selected)
        print(f"  +{addition:>4}: K_f contrast negative in {negative} of {total} "
              f"combinations of starting model and ordering")


# ==========================================================================
# constraint_percolation: the constraint graph as data are added (Fig. 7, Sec. 6)
# ==========================================================================

GRAPH_OUT = RESULTS / "constraint_percolation"
GRAPH_GRID = (0, 16, 32, 64, 96, 128, 160, 192, 256, 384, 576, 768, 1024, 1216)
N_GRAPH_DRAWS = 10_000
GRAPH_DRAW_SEED = 20260824


def k_perc(f_left: float) -> float:
    """K_perc = 1 / (1/32 + (31/32) f_L), the effective rank the graph alone allows."""
    return 1.0 / (1.0 / N_LEFT + (N_LEFT - 1) / N_LEFT * f_left)


def selection_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Graph of the data selection of the study at every count of ``GRAPH_GRID``.

    Returns the table of component counts, f_L and K_perc, and the number of
    left substrings in each component, largest first.
    """
    rows = []
    component_rows = []
    for addition in GRAPH_GRID:
        state, left_roots = constraint_graph(distinct_branch_codes(addition))
        rows.append({"dose": addition, **state, "K_perc_primary": k_perc(float(state["f_left"]))})
        sizes = sorted(np.unique(left_roots, return_counts=True)[1].tolist(), reverse=True)
        for index, size in enumerate(sizes):
            component_rows.append({"dose": addition, "component": index + 1, "left_substrings": size})
    return pd.DataFrame(rows), pd.DataFrame(component_rows)


def graph_ensemble(n_draws: int) -> pd.DataFrame:
    """Component counts and f_L along ``n_draws`` random nested orderings of added inputs.

    Each ordering is a uniform sample without replacement of 1,216 inputs
    outside the base set (one ``rng.choice`` per ordering, all from one
    generator seeded with ``GRAPH_DRAW_SEED``); a count of the grid takes its
    first entries.  The edges are added to one union-find structure in order,
    so the graphs of one ordering are nested.
    """
    base = np.unique(base_training_codes().numpy().astype(np.int64))
    unseen = np.setdiff1d(np.arange(N_STRINGS, dtype=np.int64), base, assume_unique=True)
    base_left = base & (N_LEFT - 1)
    base_right = base >> CUT
    rng = np.random.default_rng(GRAPH_DRAW_SEED)
    rows: list[dict] = []
    for draw in range(n_draws):
        sampled = rng.choice(unseen, size=max(GRAPH_GRID), replace=False)
        sampled_left = sampled & (N_LEFT - 1)
        sampled_right = sampled >> CUT
        union = UnionFind(N_NODES)
        for left, right in zip(base_left, base_right):
            union.union(int(left), N_LEFT + int(right))
        previous = 0
        for addition in GRAPH_GRID:
            for index in range(previous, addition):
                union.union(int(sampled_left[index]), N_LEFT + int(sampled_right[index]))
            roots = np.asarray([union.find(node) for node in range(N_NODES)], dtype=int)
            full_counts = np.unique(roots, return_counts=True)[1]
            left_counts = np.unique(roots[:N_LEFT], return_counts=True)[1]
            f_left = float(np.sum(left_counts * (left_counts - 1)) / OFFDIAG_COUNT)
            components = int(len(full_counts))
            rows.append({
                "draw": draw,
                "dose": addition,
                "components": components,
                "d_free": components - 1,
                "connected": int(components == 1),
                "c_left": int(len(left_counts)),
                "d_free_left": int(len(left_counts) - 1),
                "left_connected": int(len(left_counts) == 1),
                "f_left": f_left,
            })
            previous = addition
        if (draw + 1) % 1000 == 0:
            print(f"random orderings complete: {draw + 1}/{n_draws}", flush=True)
    return pd.DataFrame(rows)


def summarize_graph_ensemble(draws: pd.DataFrame) -> pd.DataFrame:
    """Per count: mean, median, 5 and 95 per cent quantiles of f_L, connection probabilities, K_perc at the mean f_L."""
    rows = []
    for addition, group in draws.groupby("dose"):
        f = group.f_left.to_numpy(float)
        rows.append({
            "dose": int(addition),
            "n_draws": int(len(group)),
            "f_left_mean": float(f.mean()),
            "f_left_median": float(np.median(f)),
            "f_left_q05": float(np.quantile(f, 0.05)),
            "f_left_q95": float(np.quantile(f, 0.95)),
            "p_left_connected": float(group.left_connected.mean()),
            "p_connected": float(group.connected.mean()),
            "d_free_left_mean": float(group.d_free_left.mean()),
            "d_free_mean": float(group.d_free.mean()),
            "c_left_mean": float(group.c_left.mean()),
            "K_perc_ensemble_mean_f": k_perc(float(f.mean())),
        })
    return pd.DataFrame(rows).sort_values("dose")


def half_point(table: pd.DataFrame, field: str) -> float:
    """Added-input count at which ``field`` crosses 0.5, by linear interpolation between grid counts."""
    ordered = table.sort_values("dose")
    below = ordered[ordered[field] < 0.5]
    above = ordered[ordered[field] >= 0.5]
    if below.empty or above.empty:
        return math.nan
    low = below.iloc[-1]
    high = above.iloc[0]
    if int(low.dose) >= int(high.dose):
        return float(high.dose)
    fraction = (0.5 - float(low[field])) / (float(high[field]) - float(low[field]))
    return float(low.dose + fraction * (high.dose - low.dose))


def constraint_percolation() -> None:
    """Bond percolation of the constraint graph as inputs are added to the base set (Fig. 7, Sec. 6).

    The training set of a distinct branch is the 64 base inputs plus the first
    n entries of one random ordering of the inputs outside the base set; each
    distinct input is an edge between its left substring (32 vertices) and its
    right substring (64 vertices).  Along the grid of n from 0 to 1,216:

    Data selection of the study: components of the whole graph and of its
    left vertices, whether each is connected, f_L, and K_perc, the K_f of the
    state that is fully coherent inside each component and incoherent between
    them, the prediction that uses the graph and nothing about the network.

    Random orderings: 10,000 further orderings show what is typical; per n the
    mean, median and 5 and 95 per cent quantiles of f_L, the probabilities
    that the left vertices and that the whole graph are connected, and K_perc
    at the mean f_L.  Half points: the n at which each connection
    probability crosses one half, interpolated linearly.

    Reads no file: the base set and the ordering follow from the protocol
    constants in ``common.py``.
    Writes  results/constraint_percolation/
              selection.csv                 one row per n, data selection
              selection_components.csv      left substrings per component, largest first
              random_orderings.csv          one row per ordering and n (140,000 rows)
              random_orderings_summary.csv  one row per n
              summary.json                  the two half points
    About half a minute.
    """
    GRAPH_OUT.mkdir(parents=True, exist_ok=True)
    primary, components = selection_tables()
    ensemble_draws = graph_ensemble(N_GRAPH_DRAWS)
    ensemble = summarize_graph_ensemble(ensemble_draws)
    summary = {
        "half_points": {
            "full_graph": half_point(ensemble, "p_connected"),
            "left_graph": half_point(ensemble, "p_left_connected"),
        }
    }
    primary.to_csv(GRAPH_OUT / "selection.csv", index=False)
    components.to_csv(GRAPH_OUT / "selection_components.csv", index=False)
    ensemble_draws.to_csv(GRAPH_OUT / "random_orderings.csv", index=False)
    ensemble.to_csv(GRAPH_OUT / "random_orderings_summary.csv", index=False)
    (GRAPH_OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary["half_points"], indent=2), flush=True)
    print(f"wrote {GRAPH_OUT.relative_to(ROOT)}", flush=True)


# ==========================================================================
# branch_endpoint_state: pair decomposition at update 800 (Figs. 6, 7)
# ==========================================================================

ENDPOINT_OUT = RESULTS / "branch_endpoint_state"
CK_BOOTSTRAP_SEED = 20261000        # bootstrap of C_K, plus the addition index
DELTA_BOOTSTRAP_SEED = 20261100     # bootstrap of the other contrasts, see endpoint_contrasts


def endpoint_pair_table(signs: np.ndarray, seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``pair_averages`` of every starting model, addition, branch and long-range position, and the mean over positions.

    The duplicate branch is split with the graph of the distinct branch of
    the same addition, so both are compared on identical sets of pairs.
    """
    target = long_range_target_signs()
    rows = []
    for addition_index, addition in enumerate(ADDITION_SIZES):
        _state, labels = constraint_graph(distinct_branch_codes(addition))
        same = labels[:, None] == labels[None, :]
        for seed_index, seed in enumerate(seeds):
            for branch_index, branch in enumerate(BRANCHES):
                for position_index, position in enumerate(LONG_RANGE_POSITIONS):
                    rows.append({
                        "seed": seed,
                        "dose": addition,
                        "arm": branch,
                        "position": position,
                        **pair_averages(signs[seed_index, addition_index, branch_index, position_index],
                                       target[position_index], same),
                    })
    position = pd.DataFrame(rows)
    numeric = [column for column in position.columns if column not in {"seed", "dose", "arm", "position"}]
    start = position.groupby(["seed", "dose", "arm"], as_index=False)[numeric].mean()
    return position, start


def endpoint_accuracy() -> pd.DataFrame:
    """Block and bit accuracy of the distinct and duplicate branches at update 800."""
    # pandas' default float parser: the accuracy contrasts of the contrast
    # table are differences of the values it returns.
    trajectory = pd.read_csv(BRANCHING_STUDY / "trajectory_metrics.csv")
    endpoint = trajectory[
        (trajectory.step == ENDPOINT_UPDATE) & trajectory.family.isin(["random_unique", "duplicate"])
    ].copy()
    endpoint["arm"] = np.where(endpoint.family == "random_unique", "distinct", "duplicate")
    return endpoint[["seed", "nominal_new", "arm", "A_block", "bit_accuracy"]].rename(columns={"nominal_new": "dose"})


def endpoint_contrasts(start: pd.DataFrame, accuracy: pd.DataFrame, primary: pd.DataFrame,
                       ensemble: pd.DataFrame) -> pd.DataFrame:
    """Distinct-minus-duplicate contrasts per addition with bootstrap intervals and sign counts, and the graph values of that addition.

    C_K = K_f(duplicate) - K_f(distinct) draws its bootstrap from seed
    CK_BOOTSTRAP_SEED + addition index; the contrast of the m-th quantity of
    (Q_in, Q_out, R_in, R_out, w_star, A_block, bit_accuracy) from
    DELTA_BOOTSTRAP_SEED + 17 addition index + m, over its finite values (Q_out
    and R_out are undefined once all left substrings are connected).
    """
    merged = start.merge(accuracy, on=["seed", "dose", "arm"], how="left", validate="one_to_one")
    rows = []
    metrics = ("Q_in", "Q_out", "R_in", "R_out", "w_star", "A_block", "bit_accuracy")
    primary_index = primary.set_index("dose")
    ensemble_index = ensemble.set_index("dose")
    for addition_index, addition in enumerate(ADDITION_SIZES):
        subset = merged[merged.dose == addition]
        pivot = subset.pivot(index="seed", columns="arm", values=["K_f", *metrics])
        row: dict[str, float | int] = {
            "dose": addition,
            "n_starts": len(pivot),
            "f_left_primary": float(primary_index.loc[addition, "f_left"]),
            "f_left_ensemble_mean": float(ensemble_index.loc[addition, "f_left_mean"]),
            "p_left_connected": float(ensemble_index.loc[addition, "p_left_connected"]),
            "p_connected": float(ensemble_index.loc[addition, "p_connected"]),
            "K_perc_primary": float(primary_index.loc[addition, "K_perc_primary"]),
            "K_perc_ensemble_mean_f": float(ensemble_index.loc[addition, "K_perc_ensemble_mean_f"]),
            "K_f_dist_mean": float(pivot[("K_f", "distinct")].mean()),
            "K_f_dup_mean": float(pivot[("K_f", "duplicate")].mean()),
        }
        compression = pivot[("K_f", "duplicate")].to_numpy() - pivot[("K_f", "distinct")].to_numpy()
        low, high = bootstrap(compression, CK_BOOTSTRAP_SEED + addition_index)
        row.update({"C_K_mean": float(compression.mean()), "C_K_ci_low": low, "C_K_ci_high": high,
                    "C_K_positive": int((compression > 0).sum())})
        for metric_index, metric in enumerate(metrics):
            values = pivot[(metric, "distinct")].to_numpy() - pivot[(metric, "duplicate")].to_numpy()
            finite = values[np.isfinite(values)]
            low, high = bootstrap(finite, DELTA_BOOTSTRAP_SEED + 17 * addition_index + metric_index)
            row.update({
                f"delta_{metric}_mean": float(finite.mean()) if len(finite) else math.nan,
                f"delta_{metric}_ci_low": low,
                f"delta_{metric}_ci_high": high,
                f"delta_{metric}_positive": int((finite > 0).sum()),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def branch_endpoint_state() -> None:
    """Pair correlations and function state of the branches at update 800, per added-data size (Figs. 6, 7, Sec. 5.2).

    For every starting model, addition and branch the sign table at update 800 is
    resolved into Q_in, Q_out, R_in, R_out, K_f and w_star (``pair_averages``),
    with the pairs of left substrings split by the constraint graph of the
    distinct branch's training set.  Through

        1 / K_f = 1/32 + (31/32) [f_L Q_in + (1 - f_L) Q_out],
        w_star  = 1/32 + (31/32) [f_L R_in + (1 - f_L) R_out],

    the graph fixes which pairs the data can tie, and the trained network how
    strongly they are correlated (Q) and whether with the sign of the target (R).
    Repeating base inputs adds no edge.

    The contrast table has one row per addition: f_L and K_perc of the data
    selection and of the random orderings, the means of K_f over the starting
    models in the two branches, C_K = K_f(duplicate) - K_f(distinct), and the
    distinct-minus-duplicate differences of Q_in, Q_out, R_in, R_out, w_star, A_block and bit
    accuracy, each with the central 95 per cent of 20,000 bootstrap means
    over starting models and the number of starting models in which it is
    positive.  -C_K and the w_star difference with their intervals are the
    points and bars of Fig. 6; at +1,216 they and their sign counts are the
    numbers of Sec. 5.2.

    Reads   data/branching_study/protocol.json,
            endpoint_sign_tables.npz, trajectory_metrics.csv;
            results/constraint_percolation/selection.csv
            and random_orderings_summary.csv (``constraint_percolation``)
    Writes  results/branch_endpoint_state/
              by_position.csv        starting model x addition x branch x position
              by_starting_model.csv  mean over the four positions
              contrasts.csv          one row per addition
    """
    ENDPOINT_OUT.mkdir(parents=True, exist_ok=True)
    signs, seeds = load_endpoint_sign_tables()
    # The two graph tables are read with the exact float parser, so that the
    # values copied into the contrast table are the ones the graph analysis
    # computed.
    primary = pd.read_csv(GRAPH_OUT / "selection.csv", float_precision="round_trip")
    ensemble = pd.read_csv(GRAPH_OUT / "random_orderings_summary.csv", float_precision="round_trip")
    position, start = endpoint_pair_table(signs, seeds)
    contrasts = endpoint_contrasts(start, endpoint_accuracy(), primary, ensemble)
    position.to_csv(ENDPOINT_OUT / "by_position.csv", index=False)
    start.to_csv(ENDPOINT_OUT / "by_starting_model.csv", index=False)
    contrasts.to_csv(ENDPOINT_OUT / "contrasts.csv", index=False)
    largest = contrasts[contrasts.dose == max(ADDITION_SIZES)].iloc[0]
    print(f"+{max(ADDITION_SIZES)}, update {ENDPOINT_UPDATE}, distinct - duplicate over "
          f"{int(largest.n_starts)} starting models: K_f {-largest.C_K_mean:+.3f} (negative in "
          f"{int(largest.C_K_positive)}), w_star {largest.delta_w_star_mean:+.3f} (positive in "
          f"{int(largest.delta_w_star_positive)})", flush=True)
    print(f"wrote {ENDPOINT_OUT.relative_to(ROOT)}", flush=True)


# ==========================================================================
# fixed_support_timing: the order of the response at fixed support (Fig. 8, Sec. 7)
# ==========================================================================

TIMING_OUT = RESULTS / "fixed_support_timing"
TIMING_BOOTSTRAP_SEED = 20261200
# The bootstrap seed of a quantity is set by its place in this tuple.
TIMING_METRICS = ("Q_in", "R_in", "A_block", "bit_accuracy", "K_f", "Gamma_1", "w_star")


def time_pair_tables(signs: np.ndarray, metrics: pd.DataFrame, seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``pair_averages`` with every pair inside, per starting model, branch, update and position; mean over positions.

    The mean over positions is merged with Gamma_1, A_block and bit accuracy
    as stored with the sign tables.
    """
    target = long_range_target_signs()
    same = np.ones((N_LEFT, N_LEFT), dtype=bool)
    rows = []
    for seed_index, seed in enumerate(seeds):
        for branch_index, branch in enumerate(BRANCHES):
            for update_index, update in enumerate(SAVED_UPDATES):
                for position_index, position in enumerate(LONG_RANGE_POSITIONS):
                    rows.append({
                        "seed": seed,
                        "arm": branch,
                        "update": update,
                        "position": position,
                        **pair_averages(signs[seed_index, branch_index, update_index, position_index],
                                       target[position_index], same),
                    })
    position = pd.DataFrame(rows)
    numeric = [column for column in position.columns if column not in {"seed", "arm", "update", "position"}]
    start = position.groupby(["seed", "arm", "update"], as_index=False)[numeric].mean()
    start = start.merge(
        metrics[["seed", "arm", "update", "Gamma_1", "A_block", "bit_accuracy"]],
        on=["seed", "arm", "update"],
        validate="one_to_one",
    )
    return position, start


def time_contrasts(start: pd.DataFrame) -> pd.DataFrame:
    """Distinct-minus-duplicate contrasts per update; bootstrap seed TIMING_BOOTSTRAP_SEED + 13 update index + quantity index."""
    rows = []
    for update_index, update in enumerate(SAVED_UPDATES):
        pivot = start[start["update"] == update].pivot(index="seed", columns="arm", values=list(TIMING_METRICS))
        row: dict[str, float | int] = {"update": update, "n_starts": len(pivot)}
        for metric_index, metric in enumerate(TIMING_METRICS):
            values = pivot[(metric, "distinct")].to_numpy() - pivot[(metric, "duplicate")].to_numpy()
            low, high = bootstrap(values, TIMING_BOOTSTRAP_SEED + 13 * update_index + metric_index)
            row.update({
                f"delta_{metric}_mean": float(values.mean()),
                f"delta_{metric}_ci_low": low,
                f"delta_{metric}_ci_high": high,
            })
        rows.append(row)
    return pd.DataFrame(rows)


def fixed_support_timing() -> None:
    """Order in which the function state responds to added data at fixed support (Fig. 8, Sec. 7).

    The two branches with 1,216 added inputs keep their training sets
    for all 800 updates.  The constraint graph of the distinct set is
    connected, f_L = 1, so every pair of left substrings is inside one
    component and the identifiable support is constant along the branch;
    what changes is the learned state.  At each of the 11 saved updates the sign
    table is resolved into Q_in (mean C_h^2), R_in (mean target-gauged
    correlation), K_f and w_star; Gamma_1 and the accuracies are those stored
    with the sign tables.  The contrast table gives the distinct-minus-
    duplicate difference of Q_in, R_in, A_block, bit accuracy, K_f, Gamma_1
    and w_star per update, with the central 95 per cent of 20,000 bootstrap
    means over starting models.  Both branches start from the same weights, so
    every contrast is zero at update 0.

    Reads   data/branching_study/protocol.json,
            checkpoint_sign_tables.npz
    Writes  results/fixed_support_timing/
              by_position.csv        starting model x branch x update x position
              by_starting_model.csv  mean over positions, with Gamma_1 and accuracies
              contrasts.csv          one row per update
    """
    TIMING_OUT.mkdir(parents=True, exist_ok=True)
    time_signs, time_metrics, time_seeds = load_checkpoint_sign_tables()
    time_position, time_start = time_pair_tables(time_signs, time_metrics, time_seeds)
    time_summary = time_contrasts(time_start)
    time_position.to_csv(TIMING_OUT / "by_position.csv", index=False)
    time_start.to_csv(TIMING_OUT / "by_starting_model.csv", index=False)
    time_summary.to_csv(TIMING_OUT / "contrasts.csv", index=False)
    print(f"wrote {TIMING_OUT.relative_to(ROOT)}", flush=True)


# ==========================================================================
# conditional_function_splitting: splitting and merging at retreat and recovery (Figs. 9, 11, Sec. 8.2, App. B.2)
# ==========================================================================

SPLITTING_OUT = RESULTS / "conditional_function_splitting"
COHERENCE_THRESHOLDS = (0.85, 0.90, 0.95)


def stage_state(seed: int, epoch: int) -> tuple[float, float]:
    """K_f and Gamma_1 of one stored sign table, averaged over the four long-range outputs."""
    values = [rank_and_coherence(load_sign_table(seed, epoch, j)) for j in range(4)]
    return float(np.mean([v[0] for v in values])), float(np.mean([v[1] for v in values]))


def splitting_and_overlap(seed: int, epoch_a: int, epoch_b: int, seed_offset: int,
                          threshold: float = 0.90) -> tuple[float, float, float, np.ndarray, np.ndarray, int, int]:
    """B, R_2 and R_4 from checkpoint ``epoch_a`` to ``epoch_b`` of one run, averaged over the four outputs.

    The clustering of output j at ``epoch_a`` draws from seed_offset + 211 seed
    + j, at ``epoch_b`` from seed_offset + 50 + 211 seed + j, so the result does
    not depend on the order of the calculations.  Also returns the labels and
    numbers of clusters of the two checkpoints at the first output.
    """
    bs, r2, r4 = [], [], []
    saved = None
    for j in range(4):
        pa, pb = load_sign_table(seed, epoch_a, j), load_sign_table(seed, epoch_b, j)
        la, na, _ = cluster_conditional_functions(pa, seed=seed_offset + seed * 211 + j, threshold=threshold)
        lb, nb, _ = cluster_conditional_functions(pb, seed=seed_offset + 50 + seed * 211 + j, threshold=threshold)
        bs.append(splitting_entropy(la, lb))
        r2.append(subspace_overlap(pa, pb, 2))
        r4.append(subspace_overlap(pa, pb, 4))
        if j == 0:
            saved = (la, lb, na, nb)
    assert saved is not None
    return float(np.mean(bs)), float(np.mean(r2)), float(np.mean(r4)), saved[0], saved[1], saved[2], saved[3]


def function_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """F_h = q_h^2 with q_h = (1/2048) sum_x h(x) h'(x): one when two sign tables agree up to a global sign."""
    return float((np.dot(a.astype(float), b.astype(float)) / N_STRINGS) ** 2)


def splitting_by_event() -> list[dict]:
    """Splitting entropy B and subspace overlaps R_2, R_4 of every first event; writes ``splitting_by_event.csv``.

    One row per event: B, R_2 and R_4 from last-high to retreat, over the
    control interval and from retreat to recovery.

    For an event with a recovery checkpoint also H[T(retreat) | T(recovery)],
    the many-to-one merging, the function fidelity between last-high and
    recovery and K_f at recovery minus K_f at last-high.  Column suffixes
    ``_retreat``, ``_control`` and ``_recovery`` name the pair of checkpoints;
    ``B_forward_recovery`` is H[T(recovery) | T(retreat)], ``B_merge_recovery``
    is H[T(retreat) | T(recovery)].
    """
    by_event = []
    for row in first_events():
        seed = int(row["seed"])
        eps = stage_epochs(row)
        if eps["last_high"] is None or eps["retreat"] is None:
            continue
        b, r2, r4, *_ = splitting_and_overlap(seed, eps["last_high"], eps["retreat"], 20261000)
        cb, cr2, cr4, *_ = splitting_and_overlap(seed, int(row["local_control_start_epoch"]),
                                        int(row["local_control_end_epoch"]), 20262000)
        rec_b = merge_b = rec_r2 = rec_r4 = f_pre_rec = pr_rec_minus_pre = float("nan")
        if eps["recovery"] is not None:
            rec_b, rec_r2, rec_r4, *_ = splitting_and_overlap(seed, eps["retreat"], eps["recovery"], 20261100)
            # The reverse conditional entropy H(retreat | recovery) is the merging entropy.
            merge_values = []
            for j in range(4):
                retreat_labels, _, _ = cluster_conditional_functions(
                    load_sign_table(seed, eps["retreat"], j), seed=20261300 + seed * 211 + j)
                recovery_labels, _, _ = cluster_conditional_functions(
                    load_sign_table(seed, eps["recovery"], j), seed=20261350 + seed * 211 + j)
                merge_values.append(splitting_entropy(recovery_labels, retreat_labels))
            merge_b = float(np.mean(merge_values))
            f_pre_rec = float(np.mean([function_fidelity(load_sign_table(seed, eps["last_high"], j),
                                                         load_sign_table(seed, eps["recovery"], j)) for j in range(4)]))
            pr_rec_minus_pre = float(np.mean([rank_and_coherence(load_sign_table(seed, eps["recovery"], j))[0]
                                              - rank_and_coherence(load_sign_table(seed, eps["last_high"], j))[0]
                                              for j in range(4)]))
        by_event.append({
            "seed": seed, "event_id": row["event_id"],
            "B_retreat": b, "B_control": cb, "R2_retreat": r2, "R2_control": cr2,
            "R4_retreat": r4, "R4_control": cr4,
            "B_forward_recovery": rec_b, "B_merge_recovery": merge_b,
            "R2_recovery": rec_r2, "R4_recovery": rec_r4,
            "F_last_high_recovery": f_pre_rec, "PR_recovery_minus_last_high": pr_rec_minus_pre,
            "recovered": int(eps["recovery"] is not None),
        })
    write_rows(SPLITTING_OUT / "splitting_by_event.csv", by_event)
    return by_event


def example_clusters(by_event: list[dict]) -> list[dict]:
    """Cluster labels at last-high and at retreat of the event drawn in Fig. 9A; writes ``example_clusters.csv``.

    The event is the one whose B at retreat is nearest the median over
    events; the labels are those of output position 8, with the seeds of
    ``splitting_and_overlap`` for that output.
    """
    med_seed = int(min(by_event, key=lambda r: abs(r["B_retreat"] - np.median([x["B_retreat"] for x in by_event])))["seed"])
    ex = next(r for r in first_events() if int(r["seed"]) == med_seed)
    eps = stage_epochs(ex)
    p0, p1 = load_sign_table(med_seed, eps["last_high"], 0), load_sign_table(med_seed, eps["retreat"], 0)
    a0, n0, _ = cluster_conditional_functions(p0, seed=20261000 + med_seed * 211)
    a1, n1, _ = cluster_conditional_functions(p1, seed=20261050 + med_seed * 211)
    rows = [
        {
            "seed": med_seed,
            "left_substring": index,
            "last_high_cluster": int(a0[index]),
            "retreat_cluster": int(a1[index]),
            "n_last_high_clusters": int(n0),
            "n_retreat_clusters": int(n1),
        }
        for index in range(len(a0))
    ]
    write_rows(SPLITTING_OUT / "example_clusters.csv", rows)
    return rows


def approach_to_retreat() -> tuple[list[dict], dict]:
    """K_f, Gamma_1 and w_star at tau = -3..0 epochs from the retreat (Fig. 11A); writes ``approach_to_retreat.csv``.

    Only events with a sign table at all four epochs enter.  w_star is taken
    from the raw stored sign table times the target, so the target gauge is
    applied exactly once.  ``C_func_PR`` is K_f and ``C1`` is Gamma_1.
    Returns the rows and their medians and quartiles per tau.
    """
    window = tuple(range(-3, 1))
    targets = long_range_target_signs()
    eligible = []
    for row in first_events():
        seed = int(row["seed"])
        retreat = int(row["retreat_epoch"])
        if all(sign_table_path(seed, retreat + tau).exists() for tau in window):
            eligible.append(row)
    rows = []
    for row in eligible:
        seed = int(row["seed"])
        retreat = int(row["retreat_epoch"])
        for tau in window:
            pr, c1 = stage_state(seed, retreat + tau)
            ws = float(np.mean([target_population(load_sign_table(seed, retreat + tau, j) * targets[j]) for j in range(4)]))
            rows.append({"seed": seed, "event_id": row["event_id"], "tau": tau, "epoch": retreat + tau,
                         "C_func_PR": pr, "C1": c1, "w_star": ws})
    write_rows(SPLITTING_OUT / "approach_to_retreat.csv", rows)
    summary = {"window": list(window), "n_seeds": len(eligible)}
    for key in ("C_func_PR", "C1", "w_star"):
        summary[key] = {
            str(tau): {
                "median": float(np.median([r[key] for r in rows if r["tau"] == tau])),
                "q25": float(np.quantile([r[key] for r in rows if r["tau"] == tau], .25)),
                "q75": float(np.quantile([r[key] for r in rows if r["tau"] == tau], .75)),
            }
            for tau in window
        }
    return rows, summary


def threshold_robustness() -> tuple[list[dict], dict]:
    """B at retreat and over the control, and both retreat-recovery entropies, at each threshold on Gamma(n) (Fig. 11B, C).

    Writes ``threshold_robustness.csv``, one row per threshold and event, and
    returns the medians and fractions per threshold.
    """
    rows = []
    for threshold in COHERENCE_THRESHOLDS:
        for row in first_events():
            seed = int(row["seed"])
            eps = stage_epochs(row)
            b, _, _, *_ = splitting_and_overlap(seed, eps["last_high"], eps["retreat"], 20261000, threshold)
            cb, _, _, *_ = splitting_and_overlap(seed, int(row["local_control_start_epoch"]),
                                        int(row["local_control_end_epoch"]), 20262000, threshold)
            forward = reverse = float("nan")
            if eps["recovery"] is not None:
                fvals, rvals = [], []
                for j in range(4):
                    retreat_f, _, _ = cluster_conditional_functions(
                        load_sign_table(seed, eps["retreat"], j), 20261100 + seed * 211 + j, threshold)
                    recovery_f, _, _ = cluster_conditional_functions(
                        load_sign_table(seed, eps["recovery"], j), 20261150 + seed * 211 + j, threshold)
                    retreat_r, _, _ = cluster_conditional_functions(
                        load_sign_table(seed, eps["retreat"], j), 20261300 + seed * 211 + j, threshold)
                    recovery_r, _, _ = cluster_conditional_functions(
                        load_sign_table(seed, eps["recovery"], j), 20261350 + seed * 211 + j, threshold)
                    fvals.append(splitting_entropy(retreat_f, recovery_f))
                    rvals.append(splitting_entropy(recovery_r, retreat_r))
                forward, reverse = float(np.mean(fvals)), float(np.mean(rvals))
            rows.append({"threshold": threshold, "seed": seed, "B_retreat": b, "B_control": cb,
                         "H_recovery_given_retreat": forward, "H_retreat_given_recovery": reverse})
    write_rows(SPLITTING_OUT / "threshold_robustness.csv", rows)
    summary = {}
    for threshold in COHERENCE_THRESHOLDS:
        current = [r for r in rows if r["threshold"] == threshold]
        with_recovery = [r for r in current if np.isfinite(r["H_recovery_given_retreat"])]
        summary[str(threshold)] = {
            "B_retreat_median": float(np.median([r["B_retreat"] for r in current])),
            "B_control_median": float(np.median([r["B_control"] for r in current])),
            "B_retreat_gt_control_fraction": float(np.mean([r["B_retreat"] > r["B_control"] for r in current])),
            "forward_entropy_median": float(np.median([r["H_recovery_given_retreat"] for r in with_recovery])),
            "reverse_entropy_median": float(np.median([r["H_retreat_given_recovery"] for r in with_recovery])),
            "reverse_gt_forward_fraction": float(np.mean([r["H_retreat_given_recovery"] > r["H_recovery_given_retreat"]
                                                          for r in with_recovery])),
        }
    return rows, summary


def conditional_function_splitting() -> None:
    """Splitting and merging of the conditional functions at retreat and recovery (Figs. 9, 11, Sec. 8.2, App. B.2).

    The unit is the first retreat event of each of the 24 seeds, with its
    last-high, retreat and (in 20 events) recovery checkpoints and its
    ``local`` control interval, a stable interval of the same run and the
    same duration.  Each output j = 8..11 is analysed from its stored sign
    table and the results are averaged over the four.

    Each checkpoint is clustered on its own (``common.cluster_conditional_functions``:
    the smallest n <= 8 with Gamma(n) >= 0.90), so labels at two checkpoints
    are unrelated.  Between two checkpoints e0 -> e1:

        B      = H[T(e1) | T(e0)]               splitting entropy in bits
        R_chi  = Tr[P_chi(e0) P_chi(e1)] / chi   overlap of the leading subspaces, chi = 2, 4

    B is positive at retreat and near zero over the control: conditional
    functions that moved together split.  At recovery the reverse entropy
    exceeds the forward one: they merge many-to-one.  Every clustering draws
    from its own generator, seeded from a constant of the comparison, the
    seed of the run and the output index.

    Reads   results/retreat_events/events.csv, results/event_sign_tables/
    Writes  results/conditional_function_splitting/
              splitting_by_event.csv    one row per event
              example_clusters.csv      the 32 labels of the event of Fig. 9A
              approach_to_retreat.csv   one row per event and tau = -3..0
              threshold_robustness.csv  one row per threshold and event
              summary.json              medians over events
    About half a minute.
    """
    SPLITTING_OUT.mkdir(parents=True, exist_ok=True)
    by_event = splitting_by_event()
    example = example_clusters(by_event)
    _approach_rows, approach_summary = approach_to_retreat()
    _robustness_rows, robustness_summary = threshold_robustness()
    payload = {
        "n_primary_seeds": len(first_events()), "n_all_events": len(all_events()),
        "mechanism": {
            "B_retreat_median": float(np.median([r["B_retreat"] for r in by_event])),
            "B_control_median": float(np.median([r["B_control"] for r in by_event])),
            "B_retreat_positive_fraction": float(np.mean([r["B_retreat"] > 0 for r in by_event])),
            "B_merge_median": float(np.nanmedian([r["B_merge_recovery"] for r in by_event])),
            "R2_retreat_median": float(np.median([r["R2_retreat"] for r in by_event])),
            "R2_control_median": float(np.median([r["R2_control"] for r in by_event])),
            "R4_retreat_median": float(np.median([r["R4_retreat"] for r in by_event])),
            "R4_control_median": float(np.median([r["R4_control"] for r in by_event])),
        },
        "balanced_relative_window": approach_summary,
        "threshold_robustness": robustness_summary,
    }
    (SPLITTING_OUT / "summary.json").write_text(json.dumps(payload, indent=2))

    example_seed = example[0]["seed"]
    example_b = next(r["B_retreat"] for r in by_event if r["seed"] == example_seed)
    print(f"{len(by_event)} events, {sum(r['recovered'] for r in by_event)} with a recovery checkpoint")
    print(f"example event of Fig. 9A: seed {example_seed}, "
          f"{example[0]['n_last_high_clusters']} -> {example[0]['n_retreat_clusters']} clusters at output position "
          f"{LONG_RANGE_POSITIONS[0]}, B = {example_b:.3f} bits over the four outputs")
    print(json.dumps(payload["mechanism"], indent=2))
    print(f"wrote {SPLITTING_OUT.relative_to(ROOT)}")


# ==========================================================================
# retreat_pair_correlations: pair correlations at last-high, retreat and recovery (Secs. 3, 8.1)
# ==========================================================================

PAIR_OUT = RESULTS / "retreat_pair_correlations"
STAGES = ("last_high", "retreat", "recovery")
UPPER = np.triu_indices(N_LEFT, 1)
OFFDIAG = ~np.eye(N_LEFT, dtype=bool)
LARGE_N = 1280
PAIR_STAGE_BOOTSTRAP_SEED = 20261800


def training_set_graph(events: list[dict[str, str]]) -> tuple[pd.DataFrame, np.ndarray]:
    """Constraint graph of the N = 1,280 training set, one (identical) row per event and checkpoint.

    All N = 1,280 runs train on the same 1,280 inputs of ``TRAIN_SEED``, 963 of
    them distinct.  The data do not change during a run, so the graph is the
    same at every checkpoint.  Also returns the component root of each left
    substring.
    """
    codes = training_codes(LARGE_N).numpy()
    state, labels = constraint_graph(codes)
    rows = []
    for event in events:
        epochs = stage_epochs(event)
        for stage in STAGES:
            epoch = epochs[stage]
            if epoch is None:
                continue
            rows.append({
                "seed": int(event["seed"]),
                "event_id": event["event_id"],
                "stage": stage,
                "epoch": int(epoch),
                "training_records": 1280,
                "unique_constraints": int(len(np.unique(codes))),
                **state,
            })
    table = pd.DataFrame(rows)
    for field in ("components", "d_free", "c_left", "d_free_left", "f_left"):
        if table[field].nunique() != 1:
            raise RuntimeError(f"retreat graph unexpectedly changes in {field}")
    return table, labels


def retreat_pair_matrices(events: list[dict[str, str]], labels: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """Pair matrices and their scalars for every event, checkpoint and long-range output.

    Per checkpoint and output: Q_in, Q_out, R_in, R_out (``pair_averages``
    conventions), K_f = 1 / mean(C_h^2), its inverse, w_star = mean(C~_h),
    the mean and median of C_h^2 and of the Hamming fraction f over the 496
    pairs a < a', F_mid = fraction with 0.4 <= f <= 0.6, F_edge = fraction
    with f <= 0.2 or f >= 0.8, and the two reconstruction residuals.  Returns
    the table per output position, its mean over outputs per event and
    checkpoint, and the float32 pair matrices (NaN where an event has no
    recovery checkpoint).
    """
    n_events = len(events)
    shape = (n_events, len(STAGES), len(LONG_RANGE_POSITIONS), N_LEFT, N_LEFT)
    arrays = {name: np.full(shape, np.nan, dtype=np.float32) for name in ("C", "C2", "C_target", "hamming")}
    targets = long_range_target_signs()
    same = labels[:, None] == labels[None, :]
    inside = same & OFFDIAG
    outside = (~same) & OFFDIAG
    f_left = float(inside.sum() / (N_LEFT * (N_LEFT - 1)))
    position_rows: list[dict[str, Any]] = []

    for event_index, event in enumerate(events):
        seed = int(event["seed"])
        epochs = stage_epochs(event)
        for stage_index, stage in enumerate(STAGES):
            epoch = epochs[stage]
            if epoch is None:
                continue
            for position_index, position in enumerate(LONG_RANGE_POSITIONS):
                sign_table = load_sign_table(seed, int(epoch), position_index)
                C, C2, Ctarget, hamming = pair_matrices(sign_table, targets[position_index])
                arrays["C"][event_index, stage_index, position_index] = C
                arrays["C2"][event_index, stage_index, position_index] = C2
                arrays["C_target"][event_index, stage_index, position_index] = Ctarget
                arrays["hamming"][event_index, stage_index, position_index] = hamming
                q_in = float(C2[inside].mean()) if inside.any() else math.nan
                q_out = float(C2[outside].mean()) if outside.any() else math.nan
                r_in = float(Ctarget[inside].mean()) if inside.any() else math.nan
                r_out = float(Ctarget[outside].mean()) if outside.any() else math.nan
                inverse_k = float(C2.mean())
                w_star = float(Ctarget.mean())
                reconstructed_k = 1 / N_LEFT + (N_LEFT - 1) / N_LEFT * (
                    (f_left * q_in if f_left else 0.0)
                    + ((1 - f_left) * q_out if f_left < 1 else 0.0)
                )
                reconstructed_w = 1 / N_LEFT + (N_LEFT - 1) / N_LEFT * (
                    (f_left * r_in if f_left else 0.0)
                    + ((1 - f_left) * r_out if f_left < 1 else 0.0)
                )
                h = hamming[UPPER]
                c2 = C2[UPPER]
                position_rows.append({
                    "seed": seed,
                    "event_id": event["event_id"],
                    "stage": stage,
                    "epoch": int(epoch),
                    "position": position,
                    "f_left": f_left,
                    "Q_in": q_in,
                    "Q_out": q_out,
                    "R_in": r_in,
                    "R_out": r_out,
                    "K_f": 1.0 / inverse_k,
                    "inverse_K_f": inverse_k,
                    "w_star": w_star,
                    "mean_offdiag_C2": float(c2.mean()),
                    "median_offdiag_C2": float(np.median(c2)),
                    "mean_hamming_fraction": float(h.mean()),
                    "median_hamming_fraction": float(np.median(h)),
                    "F_mid": float(((h >= 0.4) & (h <= 0.6)).mean()),
                    "F_edge": float(((h <= 0.2) | (h >= 0.8)).mean()),
                    "K_inverse_reconstruction_residual": abs(inverse_k - reconstructed_k),
                    "w_star_reconstruction_residual": abs(w_star - reconstructed_w),
                })

    position = pd.DataFrame(position_rows)
    numeric = [column for column in position if column not in {"seed", "event_id", "stage", "epoch", "position"}]
    event_table = position.groupby(["seed", "event_id", "stage", "epoch"], as_index=False)[numeric].mean()
    arrays["event_ids"] = np.asarray([event["event_id"] for event in events])
    arrays["seeds"] = np.asarray([int(event["seed"]) for event in events])
    arrays["stages"] = np.asarray(STAGES)
    arrays["positions"] = np.asarray(LONG_RANGE_POSITIONS)
    return position, event_table, arrays


def stage_sign_tables(events: list[dict[str, str]]) -> dict[str, np.ndarray]:
    """Raw sign tables of every event at its three checkpoints and four outputs, with availability and epochs."""
    signs = np.zeros((len(events), len(STAGES), len(LONG_RANGE_POSITIONS), N_STRINGS), dtype=np.int8)
    available = np.zeros((len(events), len(STAGES)), dtype=np.int8)
    epochs = np.full((len(events), len(STAGES)), -1, dtype=int)
    for event_index, event in enumerate(events):
        stage_epoch = stage_epochs(event)
        for stage_index, stage in enumerate(STAGES):
            epoch = stage_epoch[stage]
            if epoch is None:
                continue
            available[event_index, stage_index] = 1
            epochs[event_index, stage_index] = int(epoch)
            for position_index in range(len(LONG_RANGE_POSITIONS)):
                signs[event_index, stage_index, position_index] = load_sign_table(int(event["seed"]), int(epoch), position_index)
    return {"signs": signs, "available": available, "epochs": epochs}


def retreat_stage_means(event_table: pd.DataFrame) -> pd.DataFrame:
    """Mean over events and bootstrap interval of each scalar at each checkpoint.

    The bootstrap seed is PAIR_STAGE_BOOTSTRAP_SEED + 17 stage index + quantity index.
    """
    rows = []
    for stage_index, stage in enumerate(STAGES):
        group = event_table[event_table.stage == stage]
        for metric_index, metric in enumerate(
            ("K_f", "inverse_K_f", "w_star", "mean_offdiag_C2", "median_offdiag_C2", "F_mid", "F_edge")
        ):
            values = group[metric].to_numpy(float)
            low, high = bootstrap(values, PAIR_STAGE_BOOTSTRAP_SEED + 17 * stage_index + metric_index)
            rows.append({
                "stage": stage,
                "metric": metric,
                "n_events": len(values),
                "mean": float(values.mean()),
                "ci_low": low,
                "ci_high": high,
            })
    return pd.DataFrame(rows)


def retreat_pair_correlations() -> None:
    """Pair correlations of the conditional functions at last-high, retreat and recovery (Secs. 3, 8.1).

    The unit is the first retreat event of each of the 24 seeds at its
    last-high, retreat and recovery checkpoints (20 events have the last).
    For each checkpoint and output j = 8..11 the raw sign table h(a, b) gives
    the 32 x 32 matrices C_h, C_h^2, the target-gauged C~_h and the Hamming
    fraction f = (1 - C_h) / 2 (``common.pair_matrices``).

    The training set of the N = 1,280 runs has 963 distinct inputs (Sec. 3)
    and its constraint graph is connected, f_L = 1 (Sec. 8.1): every pair counts as
    inside one component and Q_out, R_out are empty.  Retreat is therefore a
    re-differentiation on a fixed graph: the support is unchanged and the
    network loses its correlations.  Stage means are taken over events with
    the central 95 per cent of 20,000 bootstrap means over events; last-high
    and retreat use all 24 events, recovery the 20 that have one.

    Reads   results/retreat_events/events.csv, results/event_sign_tables/
    Writes  results/retreat_pair_correlations/
              graph.csv          one row per event and checkpoint
              by_position.csv    one row per event, checkpoint and output
              by_event.csv       the same averaged over the four outputs
              stage_means.csv    one row per checkpoint and quantity
              sign_tables.npz    signs [event, checkpoint, output, input] int8;
                                 available, epochs, event_ids, seeds, stages, positions
              pair_matrices.npz  C, C2, C_target, hamming [event, checkpoint,
                                 output, 32, 32] float32; event_ids, seeds,
                                 stages, positions
    A few seconds.
    """
    PAIR_OUT.mkdir(parents=True, exist_ok=True)
    events = first_events()
    graph, labels = training_set_graph(events)
    graph.to_csv(PAIR_OUT / "graph.csv", index=False)

    tables = stage_sign_tables(events)
    position, event_table, arrays = retreat_pair_matrices(events, labels)
    np.savez_compressed(
        PAIR_OUT / "sign_tables.npz",
        **tables,
        event_ids=arrays["event_ids"],
        seeds=arrays["seeds"],
        stages=arrays["stages"],
        positions=arrays["positions"],
    )
    np.savez_compressed(PAIR_OUT / "pair_matrices.npz", **arrays)
    position.to_csv(PAIR_OUT / "by_position.csv", index=False)
    event_table.to_csv(PAIR_OUT / "by_event.csv", index=False)
    stage = retreat_stage_means(event_table)
    stage.to_csv(PAIR_OUT / "stage_means.csv", index=False)

    print(f"training set: {int(graph.training_records.iloc[0])} inputs, "
          f"{int(graph.unique_constraints.iloc[0])} distinct; "
          f"components {int(graph.components.iloc[0])}, "
          f"left components {int(graph.c_left.iloc[0])}, f_L = {float(graph.f_left.iloc[0]):.6f}")
    print("largest residual of the pair relations: "
          f"1/K_f {float(position.K_inverse_reconstruction_residual.max()):.3e}, "
          f"w_star {float(position.w_star_reconstruction_residual.max()):.3e}")
    for stage_name in STAGES:
        means = {
            metric: stage[(stage.stage == stage_name) & (stage.metric == metric)].iloc[0]
            for metric in ("K_f", "w_star", "mean_offdiag_C2")
        }
        print(f"{stage_name:>9}: n = {int(means['K_f'].n_events)}, "
              + ", ".join(f"{metric} = {float(row['mean']):.3f}" for metric, row in means.items()))
    print(f"wrote {PAIR_OUT.relative_to(ROOT)}")


# ==========================================================================
# retreat_recurrence: how closely recovery returns to last-high (Fig. 10, Sec. 8.3)
# ==========================================================================

RECURRENCE_OUT = RESULTS / "retreat_recurrence"
STAGE_PAIRS = (
    ("last_high", "retreat", 0, 1),
    ("retreat", "recovery", 1, 2),
    ("last_high", "recovery", 0, 2),
)
RecurrenceData = dict  # the arrays of ``load_recurrence_inputs``
RECOVERY_STAGE_BOOTSTRAP_SEED = 20261900


def safe_corr(first: np.ndarray, second: np.ndarray, method: str = "pearson") -> float:
    """Pearson or Spearman correlation of the finite entries; NaN for fewer than three or a constant input."""
    first = np.asarray(first, dtype=float).ravel()
    second = np.asarray(second, dtype=float).ravel()
    keep = np.isfinite(first) & np.isfinite(second)
    first, second = first[keep], second[keep]
    if len(first) < 3 or np.std(first) == 0 or np.std(second) == 0:
        return math.nan
    if method == "spearman":
        return float(spearmanr(first, second).statistic)
    return float(pearsonr(first, second).statistic)


def load_recurrence_inputs() -> RecurrenceData:
    """Sign tables and pair matrices of the events at their three checkpoints, as ``retreat_pair_correlations`` wrote them."""
    tables_path = PAIR_OUT / "sign_tables.npz"
    pair_path = PAIR_OUT / "pair_matrices.npz"
    if not tables_path.exists() or not pair_path.exists():
        raise FileNotFoundError("run python3 analysis.py retreat_pair_correlations first")
    tables = np.load(tables_path, allow_pickle=False)
    pair = np.load(pair_path, allow_pickle=False)
    return {
        "signs": tables["signs"],
        "available": tables["available"].astype(bool),
        "event_ids": tables["event_ids"],
        "seeds": tables["seeds"],
        "epochs": tables["epochs"],
        "C2": pair["C2"].astype(float),
        "C_target": pair["C_target"].astype(float),
        "hamming": pair["hamming"].astype(float),
    }


def function_overlap_tables(data: RecurrenceData) -> tuple[pd.DataFrame, pd.DataFrame]:
    """q_h, |q_h|, F_h = q_h^2 and the Hamming distances between the sign tables of two checkpoints.

    q_h = mean over x of h(x) h'(x); d = mismatches / 2048 and the projective
    fraction min(d, 1 - d), which does not see a global sign.
    """
    overlap_rows: list[dict[str, Any]] = []
    hamming_rows: list[dict[str, Any]] = []
    for event_index, event_id in enumerate(data["event_ids"]):
        for first, second, i, j in STAGE_PAIRS:
            if not (data["available"][event_index, i] and data["available"][event_index, j]):
                continue
            for output_index, position in enumerate(LONG_RANGE_POSITIONS):
                first_signs = data["signs"][event_index, i, output_index].astype(float)
                second_signs = data["signs"][event_index, j, output_index].astype(float)
                q = float(np.mean(first_signs * second_signs))
                mismatches = int(np.count_nonzero(first_signs != second_signs))
                raw_hamming = mismatches / len(first_signs)
                projective_hamming = min(raw_hamming, 1.0 - raw_hamming)
                common = {
                    "seed": int(data["seeds"][event_index]),
                    "event_id": str(event_id),
                    "output_index": output_index,
                    "position": position,
                    "stage_first": first,
                    "stage_second": second,
                    "stage_pair": f"{first}_to_{second}",
                }
                overlap_rows.append({
                    **common,
                    "q_h": q,
                    "abs_q_h": abs(q),
                    "F_h": q * q,
                    "exact_return_up_to_global_sign": int(abs(q) == 1.0),
                })
                hamming_rows.append({
                    **common,
                    "truth_table_size": len(first_signs),
                    "mismatched_outputs": mismatches,
                    "full_truth_table_hamming_fraction": raw_hamming,
                    "projective_hamming_fraction": projective_hamming,
                })
    return pd.DataFrame(overlap_rows), pd.DataFrame(hamming_rows)


def pair_similarity_table(data: RecurrenceData) -> pd.DataFrame:
    """Pearson and Spearman correlation over the 496 pairs a < a' of C_h^2 and of C~_h, and the Frobenius distances.

    The Pearson correlation removes the mean of each pattern: it compares
    which pairs are more correlated than average, not how much correlation
    there is.  For last-high and recovery it is the pair-pattern correlation
    r of Sec. 8.3.
    """
    rows: list[dict[str, Any]] = []
    for event_index, event_id in enumerate(data["event_ids"]):
        for first, second, i, j in STAGE_PAIRS:
            if not (data["available"][event_index, i] and data["available"][event_index, j]):
                continue
            for output_index, position in enumerate(LONG_RANGE_POSITIONS):
                c2_first = data["C2"][event_index, i, output_index]
                c2_second = data["C2"][event_index, j, output_index]
                target_first = data["C_target"][event_index, i, output_index]
                target_second = data["C_target"][event_index, j, output_index]
                rows.append({
                    "seed": int(data["seeds"][event_index]),
                    "event_id": str(event_id),
                    "output_index": output_index,
                    "position": position,
                    "stage_first": first,
                    "stage_second": second,
                    "stage_pair": f"{first}_to_{second}",
                    "C2_pair_pattern_pearson": safe_corr(c2_first[UPPER], c2_second[UPPER]),
                    "C2_pair_pattern_spearman": safe_corr(c2_first[UPPER], c2_second[UPPER], "spearman"),
                    "target_pair_pattern_pearson": safe_corr(target_first[UPPER], target_second[UPPER]),
                    "target_pair_pattern_spearman": safe_corr(target_first[UPPER], target_second[UPPER], "spearman"),
                    "C2_frobenius_distance": float(np.linalg.norm(c2_first - c2_second)),
                    "target_frobenius_distance": float(np.linalg.norm(target_first - target_second)),
                })
    return pd.DataFrame(rows)


def return_ratio_table(overlap: pd.DataFrame, hamming: pd.DataFrame, pair: pd.DataFrame) -> pd.DataFrame:
    """Distance from last-high at recovery relative to the distance at retreat, per event and output.

        eta_pair_C2                      |C2(rec) - C2(high)| / |C2(ret) - C2(high)|
        eta_pair_target                  the same for C~_h
        eta_function_one_minus_F         [1 - F_h(high, rec)] / [1 - F_h(high, ret)]  (theta_h of Fig. 10C)
        eta_function_projective_hamming  the same ratio of projective Hamming fractions

    Zero is a complete return to the last-high state.
    """
    overlap_wide = overlap.pivot(index=["seed", "event_id", "output_index", "position"], columns="stage_pair", values="F_h")
    hamming_wide = hamming.pivot(
        index=["seed", "event_id", "output_index", "position"],
        columns="stage_pair",
        values="projective_hamming_fraction",
    )
    pair_distance = pair.pivot(
        index=["seed", "event_id", "output_index", "position"],
        columns="stage_pair",
        values=["C2_frobenius_distance", "target_frobenius_distance"],
    )
    rows = []
    for index in overlap_wide.index.intersection(hamming_wide.index).intersection(pair_distance.index):
        f_lh_ret = float(overlap_wide.loc[index, "last_high_to_retreat"])
        f_lh_rec = float(overlap_wide.loc[index, "last_high_to_recovery"])
        h_lh_ret = float(hamming_wide.loc[index, "last_high_to_retreat"])
        h_lh_rec = float(hamming_wide.loc[index, "last_high_to_recovery"])
        d_pair_ret = float(pair_distance.loc[index, ("C2_frobenius_distance", "last_high_to_retreat")])
        d_pair_rec = float(pair_distance.loc[index, ("C2_frobenius_distance", "last_high_to_recovery")])
        d_target_ret = float(pair_distance.loc[index, ("target_frobenius_distance", "last_high_to_retreat")])
        d_target_rec = float(pair_distance.loc[index, ("target_frobenius_distance", "last_high_to_recovery")])
        rows.append({
            "seed": index[0],
            "event_id": index[1],
            "output_index": index[2],
            "position": index[3],
            "eta_pair_C2": d_pair_rec / d_pair_ret if d_pair_ret else math.nan,
            "eta_pair_target": d_target_rec / d_target_ret if d_target_ret else math.nan,
            "eta_function_one_minus_F": (1.0 - f_lh_rec) / (1.0 - f_lh_ret) if f_lh_ret < 1 else math.nan,
            "eta_function_projective_hamming": h_lh_rec / h_lh_ret if h_lh_ret else math.nan,
            "C2_distance_last_high_retreat": d_pair_ret,
            "C2_distance_last_high_recovery": d_pair_rec,
            "function_distance_last_high_retreat": 1.0 - f_lh_ret,
            "function_distance_last_high_recovery": 1.0 - f_lh_rec,
        })
    return pd.DataFrame(rows)


def recovered_stage_means(data: RecurrenceData) -> pd.DataFrame:
    """Stage means over the 20 events with a recovery checkpoint.

    Mean, median and bootstrap interval, seed RECOVERY_STAGE_BOOTSTRAP_SEED
    + 19 stage index + quantity index.
    """
    with_recovery = np.flatnonzero(data["available"][:, 2])
    # Read with the default float parser of pandas, which keeps about 15
    # significant digits.  The means and intervals below are those of the
    # values it returns.
    source = pd.read_csv(PAIR_OUT / "by_event.csv")
    recovered_seeds = set(int(data["seeds"][index]) for index in with_recovery)
    source = source[source.seed.isin(recovered_seeds) & source.stage.isin(STAGES)].copy()
    rows = []
    metrics = ("K_f", "w_star", "mean_offdiag_C2", "F_mid", "F_edge")
    for stage_index, stage in enumerate(STAGES):
        group = source[source.stage == stage]
        if len(group) != len(with_recovery):
            raise RuntimeError(f"set of events with recovery has the wrong size at {stage}: {len(group)}")
        for metric_index, metric in enumerate(metrics):
            values = group[metric].to_numpy(float)
            low, high = bootstrap(values, RECOVERY_STAGE_BOOTSTRAP_SEED + 19 * stage_index + metric_index)
            rows.append({
                "stage": stage,
                "metric": metric,
                "n_events": len(values),
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "ci_low": low,
                "ci_high": high,
            })
    return pd.DataFrame(rows)


def recurrence_event_summary(overlap: pd.DataFrame, pair: pd.DataFrame, ratios: pd.DataFrame) -> pd.DataFrame:
    """Per event, the mean over the four outputs of the last-high to recovery comparison."""
    function = overlap[overlap.stage_pair == "last_high_to_recovery"].groupby(["seed", "event_id"], as_index=False).agg(
        F_h=("F_h", "mean"), q_h=("q_h", "mean"), abs_q_h=("abs_q_h", "mean")
    )
    pair_event = pair[pair.stage_pair == "last_high_to_recovery"].groupby(["seed", "event_id"], as_index=False).agg(
        C2_pair_pattern_pearson=("C2_pair_pattern_pearson", "mean"),
        target_pair_pattern_pearson=("target_pair_pattern_pearson", "mean"),
    )
    ratio_event = ratios.groupby(["seed", "event_id"], as_index=False).agg(
        eta_pair_C2=("eta_pair_C2", "mean"),
        eta_pair_target=("eta_pair_target", "mean"),
        eta_function_one_minus_F=("eta_function_one_minus_F", "mean"),
        eta_function_projective_hamming=("eta_function_projective_hamming", "mean"),
    )
    return function.merge(pair_event, on=["seed", "event_id"]).merge(ratio_event, on=["seed", "event_id"])


def summarize_recurrence(data: RecurrenceData, overlap: pd.DataFrame, hamming: pd.DataFrame, pair: pd.DataFrame,
                         ratios: pd.DataFrame, stage_means: pd.DataFrame, event: pd.DataFrame) -> dict[str, Any]:
    """Means over outputs and over events of the last-high to recovery comparison (``summary.json``)."""
    lh_rec = overlap[overlap.stage_pair == "last_high_to_recovery"]
    h_lh_rec = hamming[hamming.stage_pair == "last_high_to_recovery"]
    pair_lh_rec = pair[pair.stage_pair == "last_high_to_recovery"]
    summary: dict[str, Any] = {
        "counts": {
            "retreat_events": int(len(data["event_ids"])),
            "recovered_events": int(data["available"][:, 2].sum()),
            "outputs_per_event": len(LONG_RANGE_POSITIONS),
        },
        "exact_function_recovery": {
            "F_h_output_mean": float(lh_rec.F_h.mean()),
            "F_h_output_median": float(lh_rec.F_h.median()),
            "F_h_event_mean": float(event.F_h.mean()),
            "F_h_event_median": float(event.F_h.median()),
            "F_h_event_q25": float(event.F_h.quantile(0.25)),
            "F_h_event_q75": float(event.F_h.quantile(0.75)),
            "full_hamming_output_mean": float(h_lh_rec.full_truth_table_hamming_fraction.mean()),
            "exact_outputs_up_to_global_sign": int(lh_rec.exact_return_up_to_global_sign.sum()),
        },
        "pair_geometry_recovery": {
            "C2_output_correlation_mean": float(pair_lh_rec.C2_pair_pattern_pearson.mean()),
            "C2_event_correlation_mean": float(event.C2_pair_pattern_pearson.mean()),
            "target_output_correlation_mean": float(pair_lh_rec.target_pair_pattern_pearson.mean()),
            "target_event_correlation_mean": float(event.target_pair_pattern_pearson.mean()),
        },
        "return_efficiency": {
            "eta_pair_C2_output_mean": float(ratios.eta_pair_C2.mean()),
            "eta_pair_C2_event_mean": float(event.eta_pair_C2.mean()),
            "eta_function_output_mean": float(ratios.eta_function_one_minus_F.mean()),
            "eta_function_event_mean": float(event.eta_function_one_minus_F.mean()),
        },
        "fixed20_stage_means": {},
    }
    for stage in STAGES:
        summary["fixed20_stage_means"][stage] = {
            metric: float(stage_means[(stage_means.stage == stage) & (stage_means.metric == metric)]["mean"].iloc[0])
            for metric in ("K_f", "w_star", "mean_offdiag_C2", "F_mid", "F_edge")
        }
    return summary


def retreat_recurrence() -> None:
    """Recurrence of the learned function and of its pair pattern at recovery (Fig. 10, Sec. 8.3).

    Over the 24 first events (20 with a recovery checkpoint), three pairs of
    checkpoints are compared for each output j = 8..11: last-high with
    retreat, retreat with recovery, last-high with recovery.  Three levels are
    kept apart: the sign table itself (function overlap q_h, fidelity
    F_h = q_h^2, Hamming distance), the pattern of pair correlations (C_h^2
    and C~_h over the pairs a < a'), and the scalars K_f and w_star.  Recovery
    brings K_f, w_star and the pair pattern back while the sign table differs
    from the last-high one: recoherence without exact recurrence.

    Reads   results/retreat_pair_correlations/sign_tables.npz, pair_matrices.npz,
            by_event.csv (``retreat_pair_correlations``)
    Writes  results/retreat_recurrence/
              function_overlap.csv          one row per event, pair of checkpoints, output
              function_hamming.csv          the same rows, Hamming distances
              pair_pattern_similarity.csv   the same rows, pair-pattern comparison
              return_ratios.csv             one row per event and output
              recovered_stage_means.csv     one row per checkpoint and quantity
              event_summary.csv             one row per event with a recovery checkpoint
              summary.json                  means over outputs and over events
    A few seconds.
    """
    RECURRENCE_OUT.mkdir(parents=True, exist_ok=True)
    data = load_recurrence_inputs()
    overlap, hamming = function_overlap_tables(data)
    pair = pair_similarity_table(data)
    ratios = return_ratio_table(overlap, hamming, pair)
    stage_means = recovered_stage_means(data)
    event = recurrence_event_summary(overlap, pair, ratios)

    overlap.to_csv(RECURRENCE_OUT / "function_overlap.csv", index=False)
    hamming.to_csv(RECURRENCE_OUT / "function_hamming.csv", index=False)
    pair.to_csv(RECURRENCE_OUT / "pair_pattern_similarity.csv", index=False)
    ratios.to_csv(RECURRENCE_OUT / "return_ratios.csv", index=False)
    stage_means.to_csv(RECURRENCE_OUT / "recovered_stage_means.csv", index=False)
    event.to_csv(RECURRENCE_OUT / "event_summary.csv", index=False)

    summary = summarize_recurrence(data, overlap, hamming, pair, ratios, stage_means, event)
    with (RECURRENCE_OUT / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"wrote {RECURRENCE_OUT.relative_to(ROOT)}")


# ==========================================================================
# Command line
# ==========================================================================

# (name, function, what it writes, paper location), in dependency order: an
# item reads only data/ and the tables of items above it.
ITEMS: list[tuple[str, Callable[[], None], str, str]] = [
    ("retreat_events", retreat_events, "results/retreat_events/", "Table 1, Sec. 4.3"),
    ("event_sign_tables", event_sign_tables, "results/event_sign_tables/", "Sec. 8"),
    ("training_trajectories", training_trajectories, "results/training_trajectories/", "Fig. 3, Secs. 4.2, 4.3"),
    ("output_position_dependence", output_position_dependence, "results/output_position_dependence/", "Fig. 2, Sec. 4.1"),
    ("state_vs_accuracy", state_vs_accuracy, "results/state_vs_accuracy/", "Figs. 3B, 4, Secs. 4.3, 4.4"),
    ("largest_addition_branches", largest_addition_branches, "results/largest_addition_branches/", "Fig. 5, Sec. 5.1"),
    ("branch_contrasts", branch_contrasts, "results/branch_contrasts/", "Sec. 5.1, App. B.1"),
        ("constraint_percolation", constraint_percolation, "results/constraint_percolation/", "Fig. 7, Sec. 6"),
    ("branch_endpoint_state", branch_endpoint_state, "results/branch_endpoint_state/", "Figs. 6, 7, Sec. 5.2"),
    ("fixed_support_timing", fixed_support_timing, "results/fixed_support_timing/", "Fig. 8, Sec. 7"),
    ("conditional_function_splitting", conditional_function_splitting,
     "results/conditional_function_splitting/", "Figs. 9, 11, Sec. 8.2, App. B.2"),
    ("retreat_pair_correlations", retreat_pair_correlations, "results/retreat_pair_correlations/", "Sec. 3, Sec. 8.1"),
    ("retreat_recurrence", retreat_recurrence, "results/retreat_recurrence/", "Fig. 10, Secs. 8.1, 8.3"),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the analyses of the paper; each writes one directory under results/.",
        epilog="Run with --list for the items.",
    )
    parser.add_argument("names", nargs="*", metavar="NAME", help="items to run in this process")
    parser.add_argument("--list", action="store_true", help="list the items and stop")
    parser.add_argument("--all", action="store_true",
                        help="run every item in dependency order, each in its own process")
    args = parser.parse_args(argv)
    items = {name: (function, writes, paper) for name, function, writes, paper in ITEMS}

    if args.list:
        for name, function, writes, paper in ITEMS:
            summary = (function.__doc__ or "").strip().splitlines()[0]
            print(f"{name:31s} {paper:34s} -> {writes}\n{'':31s} {summary}")
        return 0
    if args.all:
        if args.names:
            parser.error("--all takes no item names")
        # One process per item keeps torch's thread settings and any module
        # state of one item from reaching the next.
        environment = dict(os.environ)
        for name in THREAD_VARIABLES:
            environment.setdefault(name, "1")
        for name, _function, writes, _paper in ITEMS:
            print(f"[analysis] {name}", flush=True)
            start = time.time()
            status = subprocess.run([sys.executable, str(Path(__file__).resolve()), name],
                                    cwd=ROOT, env=environment).returncode
            if status:
                print(f"{name} exited with status {status}", file=sys.stderr)
                return status
            print(f"    done in {time.time() - start:.0f} s -> {writes}", flush=True)
        return 0
    if not args.names:
        parser.error("name one or more items, or use --list or --all")
    unknown = [name for name in args.names if name not in items]
    if unknown:
        parser.error(f"unknown item {unknown[0]!r}; see --list")
    for name in args.names:
        items[name][0]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
