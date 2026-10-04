#!/usr/bin/env python3
"""Every training protocol of the paper: the commands that write data/.

This file trains the networks and writes the training records that the
analyses read; nothing in ``analysis.py`` or ``figures.py`` trains.  Each
protocol is one subcommand.  The task, the network and the function-state
observables are defined in ``common.py``, whose docstring gives the physics.

    python3 train.py --list                 every subcommand with the data/ directory it writes
    python3 train.py SUBCOMMAND --help      what it trains, its options and what it writes
    python3 train.py SUBCOMMAND [options]

The defaults of every subcommand are the protocol settings of data/.  A
subcommand first looks for its outputs and trains only what is absent;
``--overwrite`` trains again and writes over what is present.

TRAINING AS A STOCHASTIC RELAXATION
-----------------------------------
A network is a point theta in a weight space of 26,529 dimensions.  The
training energy of a set S of N data points (x, p(x)) is the binary cross-entropy
per output bit,

    E_S(theta) = (1 / (N L)) sum_(x in S) sum_j [ log(1 + e^(z_j)) - p_j(x) z_j ],
    z_j = logit_j(x; theta),

and training is a discrete-time stochastic relaxation in this landscape: at
each update the gradient of E on a minibatch of B = 128 data points drives AdamW
(learning rate 2e-3, weight decay 1e-4, moment estimates starting from zero).
The minibatch noise plays the part of a temperature.  The seed of a run fixes
the initial condition (torch's global generator draws the initial weights, in
the order in which ``ChainTransformer`` creates its parameters) and the
realization of the noise (a separate generator draws the minibatches).  The
training set is the quenched disorder; it is one realization, ``TRAIN_SEED``,
for every network.  Dropout is zero, so the forward pass is deterministic.

Two minibatch schemes occur.

  Epochs.  Every epoch draws a random permutation of the N data points from
  a generator seeded with seed + 1 and cuts it into ceil(N / B) batches, so
  every data point enters once per epoch (large-data trajectories,
  small-data models and small-data trajectories).  At N = 64 one batch holds
  the whole set: an epoch is one full-batch update, with the data points
  summed in permuted order.

  Uniform coordinates (branching study).  Starting model k draws one array u
  of shape (800, 128) of uniforms in [0, 1) from MINIBATCH_SEED + 7919 k.  At
  update t the branch with n training data points takes the data points
  floor(n u[t, i]), clipped at n - 1, so it samples with replacement.  Every
  branch of a starting model uses the same u: one realization of the noise,
  mapped onto the data points of each branch, so the branches differ only in
  their data.

Training in float32 is chaotic: the trajectory depends on the order of
floating-point accumulation, and so on the torch thread count and the machine.
Each protocol fixes its thread count (``--threads`` or
``--threads-per-worker``).  With the same thread count on the same machine a
protocol gives the same records bit for bit.

THE PROTOCOLS
-------------
Large-data trajectories.  30 networks (seeds 0-29) on the first 1,280 inputs
of TRAIN_SEED (963 distinct strings), 220 epochs of 10 updates.  After every
epoch each run is evaluated on the monitoring list, 4,096 inputs drawn from
seed 777: block accuracy (all 12 output bits right) and loss.  The weights are
stored at every epoch from 20 on, 201 files per run.  The monitoring values
of seeds 6-8 are evaluated from their stored weights by
``monitoring_from_checkpoints``.

Small-data models.  100 networks (seeds 0-99) on the first 64 inputs, 4,000
full-batch updates each.  Each fits its 64 inputs; they are the candidate
starting models of the branching study.

Small-data trajectories.  24 more networks of the same protocol (seeds
500-523), trained with five torch threads, with the function state and the
held-out block accuracy recorded at 52 updates and no weights kept.

Branching study.  The first 25 starting models, in ascending seed, with K_f >=
8.75 on all inputs are each copied into 15 branches that continue for 800
updates.  A branch differs from its siblings only in its training set: the 64
base inputs alone (control); the base inputs plus the first n of one random
ordering of the 1,984 inputs outside the base set (distinct, n = 16, 32, 64,
192, 576, 1,216); the base inputs plus n repeated base inputs (duplicate,
same n); the base inputs plus 64 inputs on which the starting model
contradicts the target at a long-range position, or agrees with it at all
four (counterexample, agreement).  A starting model stores no optimizer
state, so every branch starts AdamW from zero moments.  ``branching_study``
records every branch at the updates 0, 25, 50, 100, 200, 400 and 800.  The two
sign-table subcommands train the distinct and duplicate branches again and
store their output signs over all inputs: at update 800 for every addition,
and at the 11 updates of ``SAVED_UPDATES`` for the addition of 1,216.
``branch_subset_accuracy`` and ``branch_independent_orderings`` train single
branches again for the tables of Sec. 5.1 and App. B.1.

Data written                                     subcommand                      paper
------------                                     ----------                      -----
data/large_data_trajectories/checkpoints/        large_data_trajectories         Table 1, Figs. 2-4,
data/large_data_trajectories/                    (one call per seed group)       9-11, Secs. 4, 8
  monitoring_seeds{00_05,09_15,16_22,23_29}.csv
data/large_data_trajectories/                    monitoring_from_checkpoints     Table 1, Sec. 4.3
  monitoring_seeds06_08.csv
data/small_data_models/                          small_data_ensemble             starting models of
                                                 (small_data_model per network)  Secs. 5-7
data/small_data_trajectories/trajectories.json   small_data_trajectories         Fig. 3, Sec. 4.2
data/branching_study/                            branching_study                 Figs. 5-7, Secs. 5-7
  protocol.json, records/, trajectory_metrics.csv
data/branching_study/endpoint_sign_tables.npz    branch_endpoint_sign_tables     Figs. 6, 7, Secs. 5.2, 6
data/branching_study/checkpoint_sign_tables.npz  branch_checkpoint_sign_tables   Figs. 5, 8, Sec. 7
data/branching_study/checkpoints_added1216.csv   branch_subset_accuracy          Fig. 5, Sec. 5.1
data/branching_study/independent_orderings.csv   branch_independent_orderings    App. B.1
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import numpy as np
import torch
import torch.nn.functional as F

from common import (
    ADDITION_SIZES,
    ARCHITECTURE,
    BASE_N,
    BRANCHES,
    BRANCHING_STUDY,
    CHECKPOINT_SIGN_TABLES,
    CUT,
    D,
    ENDPOINT_SIGN_TABLES,
    ENDPOINT_UPDATE,
    INTERVENTION_DATA_SEED,
    L,
    LARGE_DATA_TRAJECTORIES,
    LONG_RANGE_POSITIONS,
    N_LEFT,
    N_STRINGS,
    RESULTS,
    ROOT,
    SAVED_UPDATES,
    SMALL_DATA_MODELS,
    SMALL_DATA_TRAJECTORIES,
    TRAIN_SEED,
    ChainTransformer,
    addition_order,
    base_training_codes,
    conditional_rows,
    full_binary_space,
    make_set,
    rank_and_coherence,
    selected_seeds,
    target_signs,
)

# ==========================================================================
# Settings shared by the protocols
# ==========================================================================

BATCH = 128                 # minibatch size B of every protocol
LEARNING_RATE = 2e-3        # AdamW learning rate of every network
WEIGHT_DECAY = 1e-4         # AdamW weight decay of every network
MONITORING_SEED = 777       # seed of the monitoring list, drawn like a training set
MONITORING_SIZE = 4096      # inputs of the monitoring list

# numpy 2.0 on Apple Accelerate raises divide-by-zero, overflow and invalid
# warnings from a float64 matmul whose operands and result are all finite;
# the Gram product of the spectrum below is that case.  Silencing the flags
# changes no value.
_GRAM_ERRSTATE = {"divide": "ignore", "over": "ignore", "invalid": "ignore"}


def parse_int_list(value: str) -> tuple[int, ...]:
    """A comma-separated list of integers, as an argparse type."""
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated integer list")
    return values


def write_csv(path: Path, rows: list[dict[str, Any]], lineterminator: str = "\r\n") -> None:
    """CSV of ``rows`` through a temporary file; the header is every key in order of first appearance.

    The table of ``monitoring_from_checkpoints`` is written with LF line ends
    and every other table with CRLF, the default of the csv module.  Nothing
    is written for no rows.
    """
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator=lineterminator)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def json_clean(value: Any) -> Any:
    """``value`` with tuples and arrays as lists, numpy scalars as Python numbers, paths as strings, NaN and inf as None."""
    if isinstance(value, dict):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_clean(value.tolist())
    if isinstance(value, np.generic):
        return json_clean(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def stable_hash(payload: Any) -> str:
    """SHA-256 of the compact JSON of ``json_clean(payload)`` with sorted keys.

    A record of the branching study carries the hash of its protocol, and a
    record whose hash differs from that of the requested protocol is trained
    again.  The thread count is not part of any protocol.
    """
    encoded = json.dumps(json_clean(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    """``json_clean(payload)`` with indent 2, sorted keys and a final newline, through a temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_clean(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def codes_digest(codes: torch.Tensor) -> str:
    """SHA-256 of a list of input codes as little-endian int64, in its order."""
    return hashlib.sha256(codes.numpy().astype("<i8", copy=False).tobytes()).hexdigest()


def pin_threads(threads: int) -> None:
    """Set torch to ``threads`` intra-op threads and one inter-op thread in this process.

    Called first in every worker of the branching study: the intra-op thread
    count fixes the order of accumulation of every matrix product, and with
    it the trajectory (see the module docstring).
    """
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # A reused worker may already have initialized the inter-op pool.
        pass


# ==========================================================================
# large_data_trajectories: the N = 1,280 runs with stored weights
# ==========================================================================

LARGE_N = 1280              # training-set size: the first 1,280 inputs of TRAIN_SEED, 963 distinct
LARGE_EPOCHS = 220          # epochs of every run, 10 updates each
CHECKPOINT_START = 20       # first epoch whose weights are stored; every later epoch is stored too


def large_data_complete(out: Path, seed: int) -> bool:
    """True when every stored epoch (``CHECKPOINT_START`` to ``LARGE_EPOCHS``) of run ``seed`` is in ``out``."""
    folder = out / "checkpoints" / f"seed{seed}"
    return all((folder / f"ep{epoch}.pt").exists()
               for epoch in range(CHECKPOINT_START, LARGE_EPOCHS + 1))


def monitoring_table_name(seeds: Iterable[int]) -> str:
    """File name of the monitoring log of one call: ``monitoring_seeds<first>_<last>.csv``, two digits each."""
    seeds = tuple(seeds)
    return f"monitoring_seeds{min(seeds):02d}_{max(seeds):02d}.csv"


def large_data_trajectories(args: argparse.Namespace) -> None:
    """Train N = 1,280 runs and store their weights at every epoch from 20 on.

    Every run trains a ChainTransformer of the sizes of ``ARCHITECTURE`` on
    the same training set, the first 1,280 inputs of TRAIN_SEED (963 distinct
    strings), with AdamW (learning rate 2e-3, weight decay 1e-4), batch 128,
    for 220 epochs of 10 updates.  ``torch.manual_seed(seed)`` fixes the
    initial weights and a generator seeded with seed + 1 the permutation of
    every epoch; the runs differ in nothing else.  After every epoch the run
    is evaluated on the monitoring list, 4,096 inputs drawn from seed 777: the
    block accuracy (all 12 output bits right) and the binary cross-entropy
    per output bit.  These are the values on which the retreat analyses
    define generalization onset, retreat and recovery (Table 1, Sec. 4.3).

    Writes, into ``--out-dir`` (a relative path is taken from the repository
    root; default data/large_data_trajectories):

        checkpoints/seed<k>/ep<e>.pt         state dictionary at epoch e = 20, ..., 220
        monitoring_seeds<first>_<last>.csv   seed, epoch, block_acc, loss on the
                                             monitoring list after every epoch, for
                                             the seeds this call trains; <first> and
                                             <last> are the smallest and the largest
                                             of ``--seeds``

    A run whose 201 weight files are present is not trained; ``--overwrite``
    trains every run of ``--seeds`` again.  The monitoring table is written
    when the last run of the call is finished and holds the runs of that
    call, so a call that trains nothing leaves it as it is.  The monitoring
    tables of data/ are those of the calls with seeds 0-5, 9-15, 16-22 and
    23-29, and monitoring_seeds06_08.csv, which ``monitoring_from_checkpoints``
    evaluates from the stored weights of seeds 6-8.
    """
    out = ROOT / args.out_dir
    pending = [seed for seed in args.seeds
               if args.overwrite or not large_data_complete(out, seed)]
    if not pending:
        print(f"{len(args.seeds)} runs present in {args.out_dir}; nothing trained", flush=True)
        return
    torch.set_num_threads(args.threads)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    x_tr, z_tr = make_set(LARGE_N, L, TRAIN_SEED)               # the shared training set
    x_te, z_te = make_set(MONITORING_SIZE, L, MONITORING_SEED)  # the monitoring list
    spe = int(np.ceil(LARGE_N / BATCH))                         # updates per epoch
    rows = []
    for s in pending:
        # The global generator draws the initial weights, in the order in
        # which the model creates its parameters; a separate generator draws
        # the permutations, so the two streams do not interfere.
        torch.manual_seed(s)
        model = ChainTransformer(L=L, **ARCHITECTURE, dropout=0.0)
        opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        gsh = torch.Generator().manual_seed(s + 1)
        (out / "checkpoints" / f"seed{s}").mkdir(parents=True, exist_ok=True)
        for ep in range(1, LARGE_EPOCHS + 1):
            model.train()
            perm = torch.randperm(LARGE_N, generator=gsh)
            for k in range(spe):
                idx = perm[k * BATCH:(k + 1) * BATCH]
                opt.zero_grad()
                F.binary_cross_entropy_with_logits(model(x_tr[idx]), z_tr[idx]).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                lt = model(x_te)
                ba = float(((lt > 0).float() == z_te).all(dim=1).float().mean())
                loss = float(F.binary_cross_entropy_with_logits(lt, z_te))
            rows.append({"seed": s, "epoch": ep, "block_acc": ba, "loss": loss})
            if ep >= CHECKPOINT_START:
                torch.save(model.state_dict(), out / "checkpoints" / f"seed{s}" / f"ep{ep}.pt")
            if ep % 40 == 0 or ep == 1:
                print(f"seed {s} ep {ep:03d}: block={ba:.3f}", flush=True)
    write_csv(out / monitoring_table_name(args.seeds), rows)
    print(f"trained {len(pending)} runs into {args.out_dir}", flush=True)


# ==========================================================================
# monitoring_from_checkpoints: the monitoring values of seeds 6-8
# ==========================================================================

MONITORING_CHECKPOINTS = LARGE_DATA_TRAJECTORIES / "checkpoints"
MONITORING_TABLE = LARGE_DATA_TRAJECTORIES / "monitoring_seeds06_08.csv"
MONITORING_SEEDS = (6, 7, 8)


def monitoring_from_checkpoints(args: argparse.Namespace) -> None:
    """Monitoring block accuracy and loss of seeds 6, 7 and 8, evaluated from their stored weights.

    The two monitoring values of seeds 6-8 are evaluated from their 201 weight
    files (epochs 20 to 220), on the same 4,096 inputs of seed 777 and with the
    same expressions as in ``large_data_trajectories``.  Inference only.  The
    loss is one float32 mean over all output bits of the list; its last digit
    depends on the thread count (``--threads``, default 4).

    Reads   data/large_data_trajectories/checkpoints/seed<k>/ep<e>.pt, k = 6, 7, 8
    Writes  data/large_data_trajectories/monitoring_seeds06_08.csv, columns seed,
            epoch, block_acc, loss (those of the other monitoring tables), 603
            rows, LF line ends

    A call that finds the table evaluates nothing; ``--overwrite`` evaluates
    again.  About four minutes.
    """
    if MONITORING_TABLE.exists() and not args.overwrite:
        print(f"{MONITORING_TABLE.relative_to(ROOT)} present; nothing evaluated", flush=True)
        return
    torch.set_num_threads(args.threads)
    inputs, targets = make_set(MONITORING_SIZE, L, MONITORING_SEED)
    # Every checkpoint is loaded into this one network; its own initial
    # weights are overwritten by each load.
    model = ChainTransformer(L=L, **ARCHITECTURE, dropout=0.0)
    expected = LARGE_EPOCHS - CHECKPOINT_START + 1
    rows = []
    for seed in MONITORING_SEEDS:
        paths = sorted((MONITORING_CHECKPOINTS / f"seed{seed}").glob("ep*.pt"),
                       key=lambda path: int(path.stem[2:]))
        if len(paths) != expected:
            raise ValueError(f"Expected {expected} checkpoints for seed {seed}, found {len(paths)}")
        for index, path in enumerate(paths, start=1):
            epoch = int(path.stem[2:])
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            model.eval()
            with torch.no_grad():
                lt = model(inputs)
                block_accuracy = float(((lt > 0).float() == targets).all(dim=1).float().mean())
                loss = float(F.binary_cross_entropy_with_logits(lt, targets))
            rows.append({"seed": seed, "epoch": epoch, "block_acc": block_accuracy, "loss": loss})
            if index == 1 or index % 25 == 0 or index == len(paths):
                print(f"seed {seed} {index:03d}/{len(paths)} ep{epoch}: block={block_accuracy:.3f}",
                      flush=True)
    write_csv(MONITORING_TABLE, rows, lineterminator="\n")
    print(f"Wrote {len(rows)} rows to {MONITORING_TABLE.relative_to(ROOT)}", flush=True)


# ==========================================================================
# small_data_model, small_data_ensemble: the 100 small-data models at N = 64
# ==========================================================================

TARGET_STEPS = 4000         # updates of every N = 64 network
SMALL_TEST_SIZE = 4000      # inputs of seed 777 behind ``final_test_acc``
SMALL_DATA_SEEDS = tuple(range(100))
SMALL_DATA_LOGS = RESULTS / "logs" / "small_data_models"
POLL_SECONDS = 2.0          # interval at which the ensemble looks at its jobs
# The ensemble starts every network as ``python3 <this file> small_data_model``.
TRAIN_SCRIPT = Path(__file__).resolve()


def small_data_dir(seed: int) -> Path:
    """Directory of the small-data model of ``seed``."""
    return SMALL_DATA_MODELS / f"L{L}_N{BASE_N}_seed{seed}"


def small_data_complete(seed: int) -> bool:
    """True when ``model.pt`` and ``metrics.json`` of ``seed`` are present and record this protocol.

    The record must hold L = 12, N = 64, the seed, training-set seed
    TRAIN_SEED and at least ``TARGET_STEPS`` updates.
    """
    out = small_data_dir(seed)
    metrics_path = out / "metrics.json"
    if not metrics_path.exists() or not (out / "model.pt").exists():
        return False
    try:
        with metrics_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        config = payload["config"]
        return (
            int(config["L"]) == L
            and int(config["N"]) == BASE_N
            and int(config["seed"]) == seed
            and int(payload["train_seed"]) == TRAIN_SEED
            and int(payload["steps"]) >= TARGET_STEPS
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def small_data_model(args: argparse.Namespace) -> None:
    """Train one network on the 64 base inputs for 4,000 full-batch updates and store its weights.

    The training set is the first 64 inputs of TRAIN_SEED, the same for every
    network; ``--seed`` sets the initial weights (global generator) and the
    order in which the data points of the one batch are summed (generator
    seeded with seed + 1).  With 64 data points and batch 128 an epoch is one
    update, so
    the run is 4,000 epochs.  AdamW at learning rate 2e-3 and weight decay
    1e-4.  Each of the 100 networks fits its 64 inputs exactly; its bit
    accuracy on 4,000 inputs of seed 777 lies between 0.66 and 0.82.

    Writes  data/small_data_models/L12_N64_seed<seed>/
              model.pt       state dictionary after the last update
              metrics.json   config (the settings, the thread count and the
                             output directory), train_seed, epochs, steps,
                             final bit accuracy on the training set and on
                             the 4,000 inputs of seed 777

    The branching study reads the architecture and the optimizer settings of
    a starting model from its ``config``.  A network whose record is complete
    (``small_data_complete``) is not trained; ``--overwrite`` trains it again.
    """
    out = small_data_dir(args.seed)
    if small_data_complete(args.seed) and not args.overwrite:
        print(f"{out.relative_to(ROOT)} present; nothing trained", flush=True)
        return
    torch.set_num_threads(args.threads)
    out.mkdir(parents=True, exist_ok=True)

    x_tr, z_tr = make_set(BASE_N, L, TRAIN_SEED)    # the shared training set
    x_te, z_te = make_set(SMALL_TEST_SIZE, L, MONITORING_SEED)

    torch.manual_seed(args.seed)
    model = ChainTransformer(L=L, **ARCHITECTURE, dropout=0.0)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    spe = math.ceil(BASE_N / BATCH)
    epochs = math.ceil(TARGET_STEPS / spe)
    gsh = torch.Generator().manual_seed(args.seed + 1)
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(BASE_N, generator=gsh)
        for k in range(spe):
            idx = perm[k * BATCH:(k + 1) * BATCH]
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(x_tr[idx]), z_tr[idx])
            loss.backward(); opt.step()
        # Six progress evaluations and the last one, whose accuracies are
        # stored; evaluation does not touch the weights or the generators.
        if ep % max(1, epochs // 6) == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                tr_acc = ((model(x_tr) > 0).float() == z_tr).float().mean().item()
                te_acc = ((model(x_te) > 0).float() == z_te).float().mean().item()
            print(f"[N={BASE_N} seed={args.seed}] ep{ep}/{epochs} loss={loss:.4f} "
                  f"train_acc={tr_acc:.3f} test_acc={te_acc:.3f}", flush=True)

    torch.save(model.state_dict(), out / "model.pt")
    config = {
        "L": L,
        "N": BASE_N,
        "seed": args.seed,
        "batch": BATCH,
        "channels": ARCHITECTURE["channels"],
        "depth": ARCHITECTURE["depth"],
        "heads": ARCHITECTURE["num_heads"],
        "ffn": ARCHITECTURE["ffn_dim"],
        "lr": LEARNING_RATE,
        "wd": WEIGHT_DECAY,
        "threads": args.threads,
        "out": out.relative_to(ROOT).as_posix(),
    }
    with (out / "metrics.json").open("w") as handle:
        json.dump({"config": config, "train_seed": TRAIN_SEED,
                   "epochs": epochs, "steps": epochs * spe,
                   "final_train_acc": tr_acc, "final_test_acc": te_acc},
                  handle, indent=2)
    print(f"saved {out.relative_to(ROOT)}/model.pt (train_acc={tr_acc:.3f})", flush=True)


def small_data_ensemble(args: argparse.Namespace) -> None:
    """Train the 100 small-data models (seeds 0-99), one process per network.

    Each network is ``small_data_model --seed s --threads <--threads>`` in its
    own process, ``--max-parallel`` of them at a time (default two jobs of
    three threads); the thread count of a job is written into its
    ``metrics.json``.  Networks whose record is complete are skipped, so the
    call can be resumed, and a call that finds all 100 complete trains
    nothing; ``--overwrite`` trains all 100 again.  Failed jobs are collected
    and reported at the end, and running jobs are terminated on interrupt.

    Writes  data/small_data_models/L12_N64_seed<s>/{model.pt, metrics.json}, s = 0-99
            results/logs/small_data_models/L12_N64_seed<s>.txt, the output of each job
    """
    if args.max_parallel < 1 or args.threads < 1:
        raise SystemExit("--max-parallel and --threads must be positive")
    pending = [seed for seed in SMALL_DATA_SEEDS
               if args.overwrite or not small_data_complete(seed)]
    print(f"requested={len(SMALL_DATA_SEEDS)}, reused={len(SMALL_DATA_SEEDS) - len(pending)}, "
          f"pending={len(pending)}", flush=True)

    def launch(seed: int) -> tuple[subprocess.Popen, Any]:
        SMALL_DATA_LOGS.mkdir(parents=True, exist_ok=True)
        log_handle = (SMALL_DATA_LOGS / f"L{L}_N{BASE_N}_seed{seed}.txt").open("w", encoding="utf-8")
        cmd = [sys.executable, str(TRAIN_SCRIPT), "small_data_model",
               "--seed", str(seed), "--threads", str(args.threads)]
        if args.overwrite:
            cmd.append("--overwrite")
        print(f"start L{L} N={BASE_N} seed={seed}", flush=True)
        return subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT), log_handle

    running: list[tuple[subprocess.Popen, Any, int]] = []
    failures: list[tuple[int, int]] = []
    try:
        while pending or running:
            while pending and len(running) < args.max_parallel:
                seed = pending.pop(0)
                proc, log_handle = launch(seed)
                running.append((proc, log_handle, seed))
            time.sleep(POLL_SECONDS)
            still_running = []
            for proc, log_handle, seed in running:
                returncode = proc.poll()
                if returncode is None:
                    still_running.append((proc, log_handle, seed))
                    continue
                log_handle.close()
                if returncode == 0 and small_data_complete(seed):
                    print(f"done  L{L} N={BASE_N} seed={seed}", flush=True)
                else:
                    failures.append((seed, returncode))
                    print(f"FAIL  L{L} N={BASE_N} seed={seed} rc={returncode}", flush=True)
            running = still_running
    finally:
        for proc, log_handle, _seed in running:
            if proc.poll() is None:
                proc.terminate()
            log_handle.close()
    if failures:
        details = ", ".join(f"seed={seed}/rc={rc}" for seed, rc in failures)
        raise SystemExit(f"failed runs: {details}")
    print("ALL DONE", flush=True)


# ==========================================================================
# small_data_trajectories: 24 runs at N = 64 with the state along training
# ==========================================================================

TRAJECTORY_FILE = SMALL_DATA_TRAJECTORIES / "trajectories.json"
TRAJECTORY_SEEDS = tuple(range(500, 524))
TRAJECTORY_STEPS = 4000
TRAJECTORY_THREADS = 5      # torch threads of the stored trajectories
# 52 saved updates: every 10 up to 200, every 50 up to 1,000, every 200 up to 4,000.
TRAJECTORY_CHECKPOINTS = sorted(set(
    list(range(0, 200, 10)) + list(range(200, 1000, 50))
    + list(range(1000, TRAJECTORY_STEPS + 1, 200))
))


def long_range_rank_and_coherence(hard: np.ndarray) -> tuple[float, float]:
    """K_f and Gamma_1 (``common.rank_and_coherence``) averaged over the four long-range positions."""
    values = [rank_and_coherence(hard[:, position]) for position in LONG_RANGE_POSITIONS]
    return (
        float(np.mean([value[0] for value in values])),
        float(np.mean([value[1] for value in values])),
    )


def train_trajectory(seed: int, x_tr: torch.Tensor, z_tr: torch.Tensor, inputs: torch.Tensor,
                     target: np.ndarray, training: np.ndarray) -> dict:
    """Train one N = 64 run for 4,000 updates; its record at every saved update.

    The protocol of ``small_data_model`` (batch min(128, 64) = 64, one update
    per epoch, data points in permuted order).  At a saved update the network is
    evaluated on all 2,048 inputs; ``training`` marks the 64 training inputs
    and its complement is the held-out set S.  See ``small_data_trajectories``
    for the keys.
    """
    torch.manual_seed(seed)
    model = ChainTransformer(L=L, **ARCHITECTURE, dropout=0.0)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    gsh = torch.Generator().manual_seed(seed + 1)
    rows = []
    t0 = time.time()
    step = 0
    batch = min(BATCH, BASE_N)
    spe = math.ceil(BASE_N / batch)
    while step <= TRAJECTORY_STEPS:
        if step in TRAJECTORY_CHECKPOINTS:
            model.eval()
            with torch.no_grad():
                soft = torch.tanh(model(inputs) / 2.0).numpy()
            # The sign is taken from tanh(logit / 2) >= 0, so a logit that
            # rounds to -0.0 there counts as +1.
            hard = np.where(soft >= 0, 1, -1).astype(np.int8)
            # Outputs j = 1, ..., 11; position 0 has the constant target 0.
            right = hard[:, 1:] == target[:, 1:]
            # The target-gauged table h t; K_f and Gamma_1 are blind to the
            # gauge, so this equals the raw table's values.
            error_hard = (hard * target.astype(np.int8)).astype(np.int8)
            k_f, gamma_1 = long_range_rank_and_coherence(error_hard)
            rows.append({
                "step": step,
                "A_block": float(right[~training].all(axis=1).mean()),
                "heldout_bit_accuracy": float(right[~training].mean()),
                "training_block_accuracy": float(right[training].all(axis=1).mean()),
                "training_bit_accuracy": float(right[training].mean()),
                "K_f": k_f,
                "K_f_excess": k_f - 1.0,
                "Gamma_1": gamma_1,
            })
        if step == TRAJECTORY_STEPS:
            break
        model.train()
        perm = torch.randperm(BASE_N, generator=gsh)
        for k in range(spe):
            idx = perm[k * batch:(k + 1) * batch]
            opt.zero_grad(set_to_none=True)
            F.binary_cross_entropy_with_logits(model(x_tr[idx]), z_tr[idx]).backward()
            opt.step()
            step += 1
    print(f"seed {seed} done ({time.time()-t0:.0f}s) final K_f={rows[-1]['K_f']:.3f} "
          f"training block accuracy={rows[-1]['training_block_accuracy']:.3f}", flush=True)
    return {"seed": seed, "history": rows}


def load_trajectories() -> dict:
    """The stored trajectory file, or the protocol with no runs when there is none."""
    if TRAJECTORY_FILE.exists():
        return json.loads(TRAJECTORY_FILE.read_text(encoding="utf-8"))
    return {"protocol": {"L": L, "N": BASE_N, "steps": TRAJECTORY_STEPS,
                         "train_seed": TRAIN_SEED, "seeds": [],
                         "held_out_inputs": int(N_STRINGS - np.unique(base_training_codes().numpy()).size),
                         "block_accuracy": "fraction of inputs whose outputs j = 1-11 are all right",
                         "cut": CUT,
                         "long_range_positions": list(LONG_RANGE_POSITIONS)},
            "runs": []}


def merge_trajectory(run: dict, threads: int) -> int:
    """Merge one finished run into the trajectory file (sorted by seed); the number of runs it holds."""
    payload = load_trajectories()
    runs = {int(value["seed"]): value for value in payload["runs"]}
    runs[int(run["seed"])] = run
    payload["runs"] = [runs[seed] for seed in sorted(runs)]
    payload["protocol"]["seeds"] = sorted(runs)
    payload["protocol"]["threads"] = threads
    TRAJECTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = TRAJECTORY_FILE.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    temporary.replace(TRAJECTORY_FILE)
    return len(runs)


def small_data_trajectories(args: argparse.Namespace) -> None:
    """Train 24 runs at N = 64 (seeds 500-523) and record the function state at 52 updates (Fig. 3, Sec. 4.2).

    The protocol of the starting models (``small_data_model``): the 64 base
    inputs, 4,000 full-batch updates, here with five torch threads
    (``--threads``, default 5).  No weights are kept.  At the updates
    0, 10, ..., 190, 200, 250, ..., 950, 1000, 1200, ..., 4000 the network is
    evaluated on all 2,048 inputs.

    Writes  data/small_data_trajectories/trajectories.json
              protocol   L, N, steps, the seeds present, the thread count, the
                         size of the held-out set, the cut and the long-range
                         positions
              runs       {"seed": s, "history": [...]}, one record per saved update

    Keys of a record (accuracies over the outputs j = 1, ..., 11; S is the
    held-out set of Eq. (17), the 1,984 inputs outside the training set):

        step                       number of updates
        A_block                    held-out block accuracy A_blk(S), Eq. (17)
        heldout_bit_accuracy       bit accuracy on S
        training_block_accuracy    block accuracy on the 64 training inputs
        training_bit_accuracy      bit accuracy on the 64 training inputs
        K_f, K_f_excess, Gamma_1   K_f, K_f - 1 and Gamma_1 of the target-gauged
                                   sign table at the cut after bit 5, mean over
                                   the long-range positions 8-11

    A call trains the seeds the file does not hold, one after the other, and
    merges each run into the file as soon as it is finished, so an
    interrupted call can be repeated.  A call that finds all 24 trains
    nothing; ``--overwrite`` trains all 24 again and writes over their records.
    """
    payload = load_trajectories()
    present = {int(run["seed"]) for run in payload["runs"]}
    missing = [seed for seed in TRAJECTORY_SEEDS if args.overwrite or seed not in present]
    if not missing:
        print(f"all {len(TRAJECTORY_SEEDS)} seeds present in {TRAJECTORY_FILE.relative_to(ROOT)}; "
              "nothing trained", flush=True)
        return

    torch.set_num_threads(args.threads)
    x_tr, z_tr = make_set(BASE_N, L, TRAIN_SEED)
    _codes, inputs, _targets = full_binary_space()
    target = target_signs()
    training = np.zeros(N_STRINGS, dtype=bool)
    training[base_training_codes().numpy()] = True
    for seed in missing:
        run = train_trajectory(seed, x_tr, z_tr, inputs, target, training)
        n_present = merge_trajectory(run, args.threads)
        print(f"merged seed {seed}: {n_present}/{len(TRAJECTORY_SEEDS)} trajectories", flush=True)

    final_seeds = tuple(sorted(int(run["seed"]) for run in load_trajectories()["runs"]))
    if final_seeds != TRAJECTORY_SEEDS:
        raise RuntimeError(f"incomplete set of trajectories: {final_seeds}")
    print(f"wrote all {len(TRAJECTORY_SEEDS)} trajectories to {TRAJECTORY_FILE.relative_to(ROOT)}",
          flush=True)


# ==========================================================================
# The branching study: starting models, observables, training sets, branches
# ==========================================================================

SELECTION_THRESHOLD = 8.75          # K_f a starting model needs to enter the study
STARTING_MODEL_CANDIDATES = tuple(range(100))
MAX_STARTING_MODELS = 25
BRANCH_STEPS = ENDPOINT_UPDATE      # updates of every branch
BRANCH_CHECKPOINTS = (0, 25, 50, 100, 200, 400, ENDPOINT_UPDATE)  # saved updates of the records
INJECTION_SIZE = 64                 # inputs added in a counterexample or agreement branch
# Starting model k draws its minibatch coordinates from MINIBATCH_SEED + 7919 k.
MINIBATCH_SEED = 20260823


def build_model(config: dict[str, Any]) -> ChainTransformer:
    """A network of the sizes recorded in a starting model's ``config``, to load its weights into."""
    return ChainTransformer(
        L=L,
        channels=int(config["channels"]),
        depth=int(config["depth"]),
        num_heads=int(config["heads"]),
        ffn_dim=int(config["ffn"]),
        dropout=0.0,
    )


def load_small_data_model(seed: int) -> tuple[dict[str, Any], dict[str, torch.Tensor], Path]:
    """Record, weights and directory of the small-data model of ``seed`` in data/small_data_models/.

    The record must hold L = 12, N = 64 and the training-set seed TRAIN_SEED.
    """
    run_dir = SMALL_DATA_MODELS / f"L{L}_N{BASE_N}_seed{seed}"
    with (run_dir / "metrics.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    config = metadata["config"]
    if int(config["L"]) != L or int(config["N"]) != BASE_N:
        raise ValueError(f"incompatible checkpoint config: {run_dir}")
    if int(metadata["train_seed"]) != TRAIN_SEED:
        raise ValueError(f"incompatible training-set seed: {run_dir}")
    state = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)
    return metadata, state, run_dir


def spectrum_observables(sign_table: np.ndarray) -> tuple[float, float, np.ndarray]:
    """K_f, Gamma_1 and the Schmidt spectrum of rho_A in descending order, for one sign table.

    rho_A is the crossing-gauged reduced state of ``common.conditional_rows``.
    Its eigenvalues are clipped at zero and normalized to unit sum before the
    purity is taken, so this K_f differs from ``common.effective_rank`` in the
    last bits; every record of the branching study uses this one.  Gamma_1 is
    the mean of the Gram matrix C^x_h.
    """
    rows = conditional_rows(sign_table)
    with np.errstate(**_GRAM_ERRSTATE):
        gram = rows @ rows.T
    kernel = gram / float(N_LEFT)
    spectrum = np.clip(np.linalg.eigvalsh(kernel), 0.0, None)
    spectrum /= spectrum.sum()
    purity = float(np.square(spectrum).sum())
    return 1.0 / purity, float(gram.mean()), spectrum[::-1]


@torch.no_grad()
def evaluate_model(
    model: ChainTransformer,
    exact_inputs: torch.Tensor,
    exact_targets: torch.Tensor,
) -> tuple[dict[str, float], np.ndarray]:
    """Observables of one network on all 2,048 inputs, and its raw output signs, int8 (2048, L).

    The sign is +1 for a logit >= 0.  Averages over the four long-range positions:

        K_f, K_f_exc = K_f - 1, Gamma_1   ``spectrum_observables`` of the raw table
        lambda_1, lambda_tail             largest Schmidt eigenvalue, the sum of the rest

    and the accuracies over all inputs: A_block (all 12 bits right),
    A_block_tail (the four long-range bits right), bit_accuracy, and
    disagreement_fraction = 1 - A_block.
    """
    model.eval()
    logits = model(exact_inputs)
    predicted_bits = logits >= 0.0
    target_bits = exact_targets.bool()
    signs = np.where(predicted_bits.numpy(), 1, -1).astype(np.int8)

    position_values = [spectrum_observables(signs[:, position]) for position in LONG_RANGE_POSITIONS]
    spectra = np.stack([value[2] for value in position_values])
    block_correct = (predicted_bits == target_bits).all(dim=1)
    long_range_correct = (predicted_bits[:, LONG_RANGE_POSITIONS] == target_bits[:, LONG_RANGE_POSITIONS]).all(dim=1)
    metrics = {
        "K_f": float(np.mean([value[0] for value in position_values])),
        "K_f_exc": float(np.mean([value[0] for value in position_values]) - 1.0),
        "Gamma_1": float(np.mean([value[1] for value in position_values])),
        "A_block": float(block_correct.float().mean()),
        "A_block_tail": float(long_range_correct.float().mean()),
        "bit_accuracy": float((predicted_bits == target_bits).float().mean()),
        "disagreement_fraction": float((~block_correct).float().mean()),
        "lambda_1": float(spectra[:, 0].mean()),
        "lambda_tail": float(spectra[:, 1:].sum(axis=1).mean()),
    }
    return metrics, signs


def branch_training_sets(
    start_signs: np.ndarray,
    base_codes: torch.Tensor,
    nested_codes: torch.Tensor,
    additions: tuple[int, ...],
    injection_size: int,
    seed: int,
    data_seed: int = INTERVENTION_DATA_SEED,
) -> list[dict[str, Any]]:
    """Training sets of all branches of the starting model ``seed``, in training order.

    ``start_signs`` are the raw output signs of the starting model on all inputs,
    ``base_codes`` its 64 training inputs, ``nested_codes`` one ordering of
    the inputs outside the base set (``common.addition_order``).  The
    branches, each a dict with ``arm``, ``family``, ``nominal_new`` (inputs
    added), ``new_unique`` (distinct inputs added), ``codes`` (training
    inputs), ``new_codes`` (added inputs) and ``q_contra`` (fraction of the
    added inputs on which the starting model contradicts the target at a
    long-range position):

        continuation             control: the base inputs alone
        random_plus<n>           distinct: base + nested_codes[:n], n in additions
        duplicate_plus<n>        duplicate: base + n repeated base inputs
        counterexample_plus64    base + 64 inputs outside the base set on which
                                 the starting model is wrong at a long-range position
        agreement_plus64         base + 64 on which it is right at all four

    The repeated inputs come from a stream of permutations of the 64 base
    inputs, drawn from data_seed + 37 and cut at n, so every complete cycle
    of 64 keeps the empirical distribution of the base set and the duplicate
    sets of different n are nested.  The counterexample and agreement inputs
    are drawn without replacement from data_seed + 1009 seed.  ``data_seed``
    is INTERVENTION_DATA_SEED in the study and the seed of the ordering in
    ``branch_independent_orderings``.
    """
    base_unique = set(int(value) for value in torch.unique(base_codes).tolist())
    candidate = np.asarray(
        [code for code in range(N_STRINGS) if code not in base_unique], dtype=np.int64
    )
    long_range_target = full_binary_space()[2].bool().numpy()[:, LONG_RANGE_POSITIONS]
    long_range_prediction = start_signs[:, LONG_RANGE_POSITIONS] > 0
    contradicts = np.any(long_range_prediction != long_range_target, axis=1)
    counter_candidates = candidate[contradicts[candidate]]
    agreement_candidates = candidate[~contradicts[candidate]]
    if len(counter_candidates) < injection_size or len(agreement_candidates) < injection_size:
        raise ValueError(
            f"seed {seed} lacks equal-size injection sets: "
            f"counter={len(counter_candidates)}, agreement={len(agreement_candidates)}"
        )

    # Both injection sets come from one generator, the counterexamples first.
    rng = np.random.default_rng(data_seed + 1009 * seed)
    counter_codes = torch.from_numpy(
        rng.choice(counter_candidates, injection_size, replace=False).copy()
    ).long()
    agreement_codes = torch.from_numpy(
        rng.choice(agreement_candidates, injection_size, replace=False).copy()
    ).long()
    duplicate_rng = np.random.default_rng(data_seed + 37)
    duplicate_indices = []
    while len(duplicate_indices) < max(additions):
        duplicate_indices.extend(duplicate_rng.permutation(BASE_N).tolist())
    duplicate_pool = base_codes[
        torch.as_tensor(duplicate_indices[: max(additions)], dtype=torch.long)
    ]

    specs: list[dict[str, Any]] = [
        {
            "arm": "continuation",
            "family": "control",
            "nominal_new": 0,
            "new_unique": 0,
            "codes": base_codes,
            "new_codes": torch.empty(0, dtype=torch.long),
            "q_contra": None,
        },
    ]
    for addition in additions:
        new_codes = nested_codes[:addition]
        specs.append(
            {
                "arm": f"random_plus{addition}",
                "family": "random_unique",
                "nominal_new": addition,
                "new_unique": addition,
                "codes": torch.cat((base_codes, new_codes)),
                "new_codes": new_codes,
                "q_contra": float(contradicts[new_codes.numpy()].mean()),
            }
        )
        specs.append(
            {
                "arm": f"duplicate_plus{addition}",
                "family": "duplicate",
                "nominal_new": addition,
                "new_unique": 0,
                "codes": torch.cat((base_codes, duplicate_pool[:addition])),
                "new_codes": torch.empty(0, dtype=torch.long),
                "q_contra": None,
            }
        )
    for family, new_codes in (
        ("counterexample", counter_codes),
        ("agreement", agreement_codes),
    ):
        specs.append(
            {
                "arm": f"{family}_plus{injection_size}",
                "family": family,
                "nominal_new": injection_size,
                "new_unique": injection_size,
                "codes": torch.cat((base_codes, new_codes)),
                "new_codes": new_codes,
                "q_contra": float(contradicts[new_codes.numpy()].mean()),
            }
        )
    return specs


@dataclass
class StartingModel:
    """One starting model with everything its branches need."""

    seed: int
    metadata: dict[str, Any]        # its metrics.json; ``config`` holds sizes and optimizer settings
    state: dict[str, torch.Tensor]  # its weights, copied into every branch
    run_dir: Path
    all_codes: torch.Tensor         # common.full_binary_space(): codes, inputs, 0/1 targets
    exact_inputs: torch.Tensor
    exact_targets: torch.Tensor
    base_codes: torch.Tensor        # the 64 base inputs
    nested_codes: torch.Tensor      # the ordering of the added distinct inputs
    start_metrics: dict[str, float]  # evaluate_model at the branch point
    specs: list[dict[str, Any]]     # branch_training_sets, in training order


def prepare_starting_model(seed: int, additions: tuple[int, ...],
                        ordering_seed: int = INTERVENTION_DATA_SEED) -> StartingModel:
    """Load the starting model ``seed``, evaluate it and build the training sets of its branches.

    ``ordering_seed`` seeds both the ordering of the added distinct inputs
    and the draws of ``branch_training_sets``.
    """
    metadata, state, run_dir = load_small_data_model(seed)
    all_codes, exact_inputs, exact_targets = full_binary_space()
    base_codes = base_training_codes()
    nested_codes = addition_order(base_codes, additions, seed=ordering_seed)
    starting_model = build_model(metadata["config"])
    starting_model.load_state_dict(state)
    start_metrics, start_signs = evaluate_model(starting_model, exact_inputs, exact_targets)
    specs = branch_training_sets(start_signs, base_codes, nested_codes, additions, INJECTION_SIZE, seed,
                      data_seed=ordering_seed)
    return StartingModel(seed, metadata, state, run_dir, all_codes, exact_inputs, exact_targets,
                      base_codes, nested_codes, start_metrics, specs)


def minibatch_uniforms(seed: int, steps: int) -> torch.Tensor:
    """The minibatch coordinates u of starting model ``seed``: uniforms of shape (steps, 128).

    One stream per starting model, shared by all its branches.
    """
    generator = torch.Generator().manual_seed(MINIBATCH_SEED + 7919 * seed)
    return torch.rand(steps, BATCH, generator=generator)


def train_branch(start: StartingModel, codes: torch.Tensor, uniforms: torch.Tensor,
                 saved: Iterable[int]) -> Iterator[tuple[int, ChainTransformer]]:
    """Train one branch from the start weights; yield (update, network) at every saved update.

    The network gets the start weights and a fresh AdamW with the learning
    rate and weight decay of the starting model's ``config``, so the moment
    estimates start from zero.  Update t trains on the inputs
    codes[floor(n u[t, i])], i = 0..127, with n = len(codes); there are
    len(uniforms) updates.  The network is yielded before the update of
    that index (update 0 is the branch point) and after the last one; the
    caller evaluates it, and training resumes in training mode.
    """
    model = build_model(start.metadata["config"])
    model.load_state_dict(start.state)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(start.metadata["config"]["lr"]),
        weight_decay=float(start.metadata["config"]["wd"]),
    )
    train_inputs = start.exact_inputs[codes]
    train_targets = start.exact_targets[codes]
    saved = set(saved)
    steps = len(uniforms)
    for step in range(steps + 1):
        if step in saved:
            yield step, model
        if step == steps:
            break
        model.train()
        indices = torch.clamp((uniforms[step] * len(codes)).long(), max=len(codes) - 1)
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(
            model(train_inputs[indices]), train_targets[indices]
        )
        loss.backward()
        optimizer.step()


def for_each_starting_model(task: Callable[..., Any], seeds: list[int], workers: int,
                            *arguments: Any,
                            on_result: Callable[[int, Any], None] | None = None) -> dict[int, Any]:
    """``task(seed, *arguments)`` for every seed, in a pool of ``workers`` processes or in this one.

    With one worker the seeds run here in order.  Otherwise each seed is one
    job of a process pool; when the machine denies the semaphores a pool
    needs, the seeds not yet finished run here.  The result of a seed depends
    only on the thread count the task sets.  ``on_result(seed, result)`` is
    called here as each seed finishes.  Returns the results by seed.
    """
    results: dict[int, Any] = {}

    def finished(seed: int, result: Any) -> None:
        results[seed] = result
        if on_result is not None:
            on_result(seed, result)
        print(f"seed {seed} done ({len(results)}/{len(seeds)})", flush=True)

    def serial(remaining: list[int]) -> None:
        for seed in remaining:
            finished(seed, task(seed, *arguments))

    if not seeds:
        return results
    if workers == 1:
        serial(seeds)
        return results
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(task, seed, *arguments): seed for seed in seeds}
            for future in as_completed(futures):
                finished(futures[future], future.result())
    except PermissionError:
        print("parallel workers unavailable; training serially", flush=True)
        serial([seed for seed in seeds if seed not in results])
    return results


def add_worker_arguments(parser: argparse.ArgumentParser) -> None:
    """``--workers`` and ``--threads-per-worker`` of the subcommands that train starting models in a pool."""
    parser.add_argument("--workers", type=int, default=3,
                        help="starting models trained at a time, each in its own process (default 3)")
    parser.add_argument("--threads-per-worker", type=int, default=2,
                        help="torch threads of each process (default 2)")


def check_workers(args: argparse.Namespace) -> None:
    """Refuse non-positive worker and thread counts."""
    if args.workers < 1 or args.threads_per_worker < 1:
        raise SystemExit("--workers and --threads-per-worker must be positive")


# ==========================================================================
# branching_study: the records of every branch (Figs. 5-7, Secs. 5-7)
# ==========================================================================

RECORDS = BRANCHING_STUDY / "records"
DELTA_METRICS = ("K_f", "K_f_exc", "Gamma_1", "A_block", "A_block_tail",
                 "bit_accuracy", "disagreement_fraction", "lambda_1", "lambda_tail")


def branch_protocol() -> dict[str, Any]:
    """The protocol of the study, whose hash every record carries.

    The keys and values are those of ``protocol`` in data/branching_study/protocol.json;
    a change of any of them changes the hash, and records of another hash are
    trained again.
    """
    return {
        "steps": BRANCH_STEPS,
        "checkpoints": tuple(sorted(set(BRANCH_CHECKPOINTS + (0, BRANCH_STEPS)))),
        "batch_size": BATCH,
        "doses": ADDITION_SIZES,
        "injection_size": INJECTION_SIZE,
        "complexity_threshold": SELECTION_THRESHOLD,
        "max_seeds": MAX_STARTING_MODELS,
        "intervention_data_seed": INTERVENTION_DATA_SEED,
        "train_data_seed": TRAIN_SEED,
    }


def select_starting_models() -> tuple[list[int], dict[int, dict[str, float]]]:
    """The first ``MAX_STARTING_MODELS`` seeds, ascending, of starting models with K_f >= 8.75; the observables of all.

    Every small-data model of data/small_data_models/ is evaluated on all inputs
    (``evaluate_model``); a missing one is passed over.  The selection depends
    only on sign tables, so not on the thread count.
    """
    _codes, inputs, targets = full_binary_space()
    screened: dict[int, dict[str, float]] = {}
    for seed in STARTING_MODEL_CANDIDATES:
        try:
            metadata, state, _run_dir = load_small_data_model(seed)
        except FileNotFoundError:
            continue
        model = build_model(metadata["config"])
        model.load_state_dict(state)
        screened[seed] = evaluate_model(model, inputs, targets)[0]
    selected = sorted(
        seed for seed, row in screened.items() if row["K_f"] >= SELECTION_THRESHOLD
    )[:MAX_STARTING_MODELS]
    if not selected:
        raise RuntimeError("no compatible high-complexity starting checkpoints found")
    return selected, screened


def run_branches(seed: int, protocol: dict[str, Any], threads: int) -> dict[str, Any]:
    """Train all 15 branches of one starting model, evaluate them at the saved updates, write its record.

    Runs in a worker process.  Every branch is evaluated with
    ``evaluate_model`` at each saved update.  Writes and returns the record
    records/seed_XXX.json (see ``branching_study``).
    """
    # Thread count first: it fixes the trajectory of every branch.
    pin_threads(threads)
    start = prepare_starting_model(seed, tuple(protocol["doses"]))
    batch_uniforms = minibatch_uniforms(seed, int(protocol["steps"]))

    trajectory: list[dict[str, Any]] = []
    branch_metadata = []
    for spec in start.specs:
        codes = spec["codes"]
        for step, model in train_branch(start, codes, batch_uniforms, protocol["checkpoints"]):
            metrics, _signs = evaluate_model(model, start.exact_inputs, start.exact_targets)
            trajectory.append(dict(sorted({
                "arm": spec["arm"],
                "family": spec["family"],
                "step": step,
                "nominal_new": spec["nominal_new"],
                "new_unique": spec["new_unique"],
                "q_contra": spec["q_contra"],
                **metrics,
            }.items())))
        branch_metadata.append(
            {key: value for key, value in spec.items() if key not in {"codes", "new_codes"}}
            | {
                "training_records": int(len(codes)),
                "training_unique_codes": int(torch.unique(codes).numel()),
                "codes_sha256": codes_digest(codes),
            }
        )

    result = {
        "seed": seed,
        "source_checkpoint": start.run_dir.relative_to(ROOT).as_posix(),
        "source_model_sha256": hashlib.sha256((start.run_dir / "model.pt").read_bytes()).hexdigest(),
        "source_steps": int(start.metadata["steps"]),
        "source_train_accuracy": float(start.metadata["final_train_acc"]),
        "protocol_hash": stable_hash(protocol),
        "start_metrics": start.start_metrics,
        "arm_metadata": branch_metadata,
        "trajectory": trajectory,
    }
    write_json(RECORDS / f"seed_{seed:03d}.json", result)
    return result


def flatten_trajectory(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows of trajectory_metrics.csv: seed, the record fields in alphabetical order, delta_<m> = m - m(branch point)."""
    rows = []
    for result in results:
        start = result["start_metrics"]
        for item in result["trajectory"]:
            row = {"seed": result["seed"], **dict(sorted(item.items()))}
            for metric in DELTA_METRICS:
                row[f"delta_{metric}"] = float(item[metric] - start[metric])
            rows.append(row)
    return rows


def branching_study(args: argparse.Namespace) -> None:
    """The branching study: 25 starting models x 15 branches x 800 updates, observables at 7 updates.

    The starting models are the first 25 seeds, ascending, of
    data/small_data_models/ with K_f >= 8.75 on all inputs.  Each is copied
    into the 15 branches of ``branch_training_sets``; every branch is trained by
    ``train_branch`` with the minibatch stream of its starting model and
    evaluated with ``evaluate_model`` at the updates 0, 25, 50, 100, 200, 400
    and 800 on all 2,048 inputs.

    Reads   data/small_data_models/L12_N64_seed<k>/{model.pt, metrics.json}
    Writes  data/branching_study/
              protocol.json            the protocol, its hash, the selected
                                       starting models and their observables
              records/seed_XXX.json    one record per starting model: every
                                       branch at every saved update, the
                                       training sets as counts and SHA-256
              trajectory_metrics.csv   the records as one table, one row per
                                       starting model, branch and saved update,
                                       with delta_<m> since the branch point

    protocol.json is written when absent and kept otherwise; a stored one with
    another protocol hash stops the call.  A starting model whose record carries
    the hash of the protocol is not trained.  ``trajectory_metrics.csv`` is
    written when this call trained a starting model.  ``--overwrite`` selects
    again and trains all 25.  The 375 branches take about an hour with three
    workers of two threads each.
    """
    check_workers(args)
    protocol = branch_protocol()
    protocol_hash = stable_hash(protocol)
    manifest_path = BRANCHING_STUDY / "protocol.json"
    if manifest_path.exists() and not args.overwrite:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("protocol_hash") != protocol_hash:
            raise SystemExit(f"{manifest_path.relative_to(ROOT)} holds another protocol; "
                             "pass --overwrite to train this one")
        selected = [int(seed) for seed in manifest["selected_seeds"]]
        print(f"{manifest_path.relative_to(ROOT)} present: {len(selected)} starting models", flush=True)
    else:
        selected, screened = select_starting_models()
        print(
            f"selected {len(selected)} seeds: {selected}; "
            f"K_f range {min(screened[s]['K_f'] for s in selected):.3f}-"
            f"{max(screened[s]['K_f'] for s in selected):.3f}",
            flush=True,
        )
        write_json(manifest_path, {
            "protocol": protocol,
            "protocol_hash": protocol_hash,
            "selected_seeds": selected,
            "selection_rule": (
                f"ascending seeds with K_f >= {protocol['complexity_threshold']}, "
                f"capped at {protocol['max_seeds']}"
            ),
            "screened_start_metrics": {str(seed): screened[seed] for seed in selected},
            "optimizer_state_policy": (
                "fresh identical AdamW state in every branch because the start models "
                "store weights only"
            ),
            "batch_matching": (
                "same update count, batch size, and uniform minibatch-coordinate stream "
                "within each seed"
            ),
            "duplicate_policy": (
                "each duplicate dose appends a nested stream of shuffled original records; "
                "complete 64-record cycles preserve the empirical distribution exactly"
            ),
        })

    results: list[dict[str, Any]] = []
    pending = []
    for seed in selected:
        path = RECORDS / f"seed_{seed:03d}.json"
        if path.exists() and not args.overwrite:
            with path.open(encoding="utf-8") as handle:
                cached = json.load(handle)
            if cached.get("protocol_hash") == protocol_hash:
                results.append(cached)
                continue
        pending.append(seed)

    trained = for_each_starting_model(run_branches, pending, args.workers,
                                      protocol, args.threads_per_worker)
    results.extend(trained.values())
    results.sort(key=lambda item: int(item["seed"]))
    if pending:
        write_csv(BRANCHING_STUDY / "trajectory_metrics.csv", flatten_trajectory(results))
    print(f"{len(results)} records in {RECORDS.relative_to(ROOT)}: "
          f"{len(results) - len(pending)} present, {len(pending)} trained", flush=True)


# ==========================================================================
# branch_endpoint_sign_tables: sign tables at update 800 (Figs. 6, 7, Secs. 5.2, 6)
# ==========================================================================

VALIDATED_METRICS = ("K_f", "Gamma_1", "A_block", "bit_accuracy")
VALIDATION_TOLERANCE = 1e-8


def recorded_endpoint(seed: int) -> dict[tuple[int, str], dict[str, float]]:
    """K_f, Gamma_1, A_block and bit_accuracy at update 800 in the record of one starting model, by (addition, branch)."""
    with (RECORDS / f"seed_{seed:03d}.json").open(encoding="utf-8") as handle:
        payload = json.load(handle)
    rows: dict[tuple[int, str], dict[str, float]] = {}
    for row in payload["trajectory"]:
        if int(row["step"]) != ENDPOINT_UPDATE:
            continue
        family = str(row["family"])
        if family not in {"random_unique", "duplicate"}:
            continue
        branch = "distinct" if family == "random_unique" else "duplicate"
        rows[(int(row["nominal_new"]), branch)] = {key: float(row[key]) for key in VALIDATED_METRICS}
    return rows


def endpoint_sign_tables_of_starting_model(seed: int, threads: int) -> np.ndarray:
    """Train the twelve distinct and duplicate branches of one starting model again; their signs at update 800.

    Runs in a worker process.  Every branch is trained for 800 updates by
    ``train_branch`` and evaluated once at the end; its K_f, Gamma_1, A_block
    and bit_accuracy are compared with its record.  A difference above 1e-8
    in any of them stops the call.  Returns int8 (6, 2, 4, 2048): addition,
    branch, long-range position, input.
    """
    # Thread count first: the branches follow the records only with the thread
    # count of branching_study.
    pin_threads(threads)
    start = prepare_starting_model(seed, ADDITION_SIZES)
    wanted: dict[tuple[int, str], dict[str, Any]] = {}
    for spec in start.specs:
        if spec["family"] == "random_unique":
            wanted[(int(spec["nominal_new"]), "distinct")] = spec
        elif spec["family"] == "duplicate":
            wanted[(int(spec["nominal_new"]), "duplicate")] = spec
    uniforms = minibatch_uniforms(seed, ENDPOINT_UPDATE)
    recorded = recorded_endpoint(seed)
    signs = np.empty((len(ADDITION_SIZES), len(BRANCHES), len(LONG_RANGE_POSITIONS), N_STRINGS), dtype=np.int8)
    maximum_errors = {metric: 0.0 for metric in VALIDATED_METRICS}

    for addition_index, addition in enumerate(ADDITION_SIZES):
        for branch_index, branch in enumerate(BRANCHES):
            spec = wanted[(addition, branch)]
            for _step, model in train_branch(start, spec["codes"], uniforms, (ENDPOINT_UPDATE,)):
                metrics, all_signs = evaluate_model(model, start.exact_inputs, start.exact_targets)
            signs[addition_index, branch_index] = all_signs[:, LONG_RANGE_POSITIONS].T
            reference = recorded[(addition, branch)]
            for metric in maximum_errors:
                maximum_errors[metric] = max(maximum_errors[metric],
                                             abs(float(metrics[metric]) - reference[metric]))

    if max(maximum_errors.values()) > VALIDATION_TOLERANCE:
        raise RuntimeError(f"seed {seed}: branches differ from the record: {maximum_errors}")
    return signs


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    """``np.savez_compressed`` through a temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp.{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def branch_endpoint_sign_tables(args: argparse.Namespace) -> None:
    """Sign tables at update 800 of the distinct and duplicate branches, checked against their records.

    A record of ``branching_study`` holds observables and not the outputs of
    the network; the constraint-graph analyses need the outputs.  For each of
    the 25 starting models of data/branching_study/protocol.json the six
    distinct and six duplicate branches (16 to 1,216 added inputs) are trained
    again with the start weights, optimizer and minibatch stream of the study,
    and the raw output signs at the four long-range positions are stored for
    all 2,048 inputs.  Every branch is first checked against its record
    (``endpoint_sign_tables_of_starting_model``), which holds with the thread
    count of ``branching_study``.

    Reads   data/branching_study/protocol.json, records/seed_XXX.json
            data/small_data_models/L12_N64_seed<k>/
    Writes  data/branching_study/endpoint_sign_tables.npz
              signs      int8 (25, 6, 2, 4, 2048): starting model, addition,
                         branch, long-range position, input
              seeds, doses, arms, positions   the axes

    The file is written again as each starting model finishes and holds the
    starting models done so far; a starting model it holds is not trained, and
    ``--overwrite`` trains all 25 again.  The 300 branches take about 55
    minutes with three workers of two threads each.
    """
    check_workers(args)
    seeds = selected_seeds()
    tables: dict[int, np.ndarray] = {}
    if ENDPOINT_SIGN_TABLES.exists() and not args.overwrite:
        with np.load(ENDPOINT_SIGN_TABLES, allow_pickle=False) as data:
            tables = {int(seed): table for seed, table in zip(data["seeds"], data["signs"])}

    def store(seed: int, signs: np.ndarray) -> None:
        tables[seed] = signs
        done = [value for value in seeds if value in tables]
        save_npz(ENDPOINT_SIGN_TABLES, signs=np.stack([tables[value] for value in done]),
                 seeds=np.asarray(done, dtype=np.int64), doses=np.asarray(ADDITION_SIZES, dtype=np.int64),
                 arms=np.asarray(BRANCHES), positions=np.asarray(LONG_RANGE_POSITIONS, dtype=np.int64))

    pending = [seed for seed in seeds if seed not in tables]
    for_each_starting_model(endpoint_sign_tables_of_starting_model, pending, args.workers,
                            args.threads_per_worker, on_result=store)
    print(f"{len(seeds)} starting models in {ENDPOINT_SIGN_TABLES.relative_to(ROOT)}: "
          f"{len(seeds) - len(pending)} present, {len(pending)} trained", flush=True)


# ==========================================================================
# branch_checkpoint_sign_tables: sign tables along +1,216 (Figs. 5, 8, Sec. 7)
# ==========================================================================

def checkpoint_sign_tables_of_starting_model(seed: int, threads: int) -> tuple[np.ndarray, list[dict]]:
    """Train the two branches with 1,216 added inputs of one starting model again; signs and observables at the 11 updates.

    Runs in a worker process.  The distinct and the duplicate branch are
    trained for 800 updates by ``train_branch`` and evaluated with
    ``evaluate_model`` at every update of ``SAVED_UPDATES``.  Returns the signs,
    int8 (2, 11, 4, 2048): branch, update, long-range position, input, and one
    row of observables per branch and update.
    """
    # Thread count first: it fixes the trajectory of both branches.
    pin_threads(threads)
    start = prepare_starting_model(seed, ADDITION_SIZES)
    full = max(ADDITION_SIZES)
    wanted = {
        "distinct": next(spec for spec in start.specs
                         if spec["family"] == "random_unique" and spec["nominal_new"] == full),
        "duplicate": next(spec for spec in start.specs
                          if spec["family"] == "duplicate" and spec["nominal_new"] == full),
    }
    uniforms = minibatch_uniforms(seed, max(SAVED_UPDATES))
    signs = np.empty((len(BRANCHES), len(SAVED_UPDATES), len(LONG_RANGE_POSITIONS), N_STRINGS), dtype=np.int8)
    metrics_rows: list[dict[str, float | int | str]] = []
    for branch_index, branch in enumerate(BRANCHES):
        for step, model in train_branch(start, wanted[branch]["codes"], uniforms, SAVED_UPDATES):
            metrics, all_signs = evaluate_model(model, start.exact_inputs, start.exact_targets)
            signs[branch_index, SAVED_UPDATES.index(step)] = all_signs[:, LONG_RANGE_POSITIONS].T
            metrics_rows.append({"seed": seed, "arm": branch, "update": step, **metrics})
    return signs, metrics_rows


def branch_checkpoint_sign_tables(args: argparse.Namespace) -> None:
    """Sign tables of the two branches with 1,216 added inputs at the 11 updates of ``SAVED_UPDATES``.

    For each of the 25 starting models of data/branching_study/protocol.json the
    distinct and the duplicate branch with 1,216 added inputs are trained
    again with the start weights, optimizer and minibatch stream of the study.
    At the updates 0, 5, 10, 15, 20, 25, 50, 100, 200, 400 and 800 the raw
    output signs at the four long-range positions are stored for all 2,048
    inputs, together with the observables of ``evaluate_model``.

    Reads   data/branching_study/protocol.json
            data/small_data_models/L12_N64_seed<k>/
    Writes  data/branching_study/checkpoint_sign_tables.npz
              signs          int8 (25, 2, 11, 4, 2048): starting model, branch,
                             update, long-range position, input
              metrics_json   one row of observables per starting model, branch
                             and update
              seeds, arms, updates, positions   the axes

    The file is written again as each starting model finishes and holds the
    starting models done so far; a starting model it holds is not trained, and
    ``--overwrite`` trains all 25 again.  The 50 branches take about 14
    minutes with three workers of two threads each.
    """
    check_workers(args)
    seeds = selected_seeds()
    tables: dict[int, tuple[np.ndarray, list[dict]]] = {}
    if CHECKPOINT_SIGN_TABLES.exists() and not args.overwrite:
        with np.load(CHECKPOINT_SIGN_TABLES, allow_pickle=False) as data:
            rows = json.loads(str(data["metrics_json"].item()))
            for seed, table in zip(data["seeds"], data["signs"]):
                tables[int(seed)] = (table, [row for row in rows if int(row["seed"]) == int(seed)])

    def store(seed: int, result: tuple[np.ndarray, list[dict]]) -> None:
        tables[seed] = result
        done = [value for value in seeds if value in tables]
        save_npz(CHECKPOINT_SIGN_TABLES, signs=np.stack([tables[value][0] for value in done]),
                 metrics_json=np.asarray(json.dumps([row for value in done for row in tables[value][1]])),
                 seeds=np.asarray(done, dtype=np.int64), arms=np.asarray(BRANCHES),
                 updates=np.asarray(SAVED_UPDATES, dtype=np.int64),
                 positions=np.asarray(LONG_RANGE_POSITIONS, dtype=np.int64))

    pending = [seed for seed in seeds if seed not in tables]
    for_each_starting_model(checkpoint_sign_tables_of_starting_model, pending, args.workers,
                            args.threads_per_worker, on_result=store)
    print(f"{len(seeds)} starting models in {CHECKPOINT_SIGN_TABLES.relative_to(ROOT)}: "
          f"{len(seeds) - len(pending)} present, {len(pending)} trained", flush=True)


# ==========================================================================
# branch_subset_accuracy: accuracy on input subsets along +1,216 (Fig. 5, Sec. 5.1)
# ==========================================================================

SUBSET_TABLE = BRANCHING_STUDY / "checkpoints_added1216.csv"


@torch.no_grad()
def evaluate_subsets(model: ChainTransformer, exact_inputs: torch.Tensor, exact_targets: torch.Tensor,
                     subsets: dict[str, torch.Tensor]) -> dict[str, Any]:
    """K_f, K_f_exc and Gamma_1 of one network (``spectrum_observables``, long-range mean), and its accuracies on each subset.

    For every subset: <set>_A_block, <set>_A_block_tail (the four long-range
    bits right) and <set>_bit_accuracy over its inputs.
    """
    model.eval()
    logits = model(exact_inputs)
    predicted = logits >= 0.0
    signs = np.where(predicted.numpy(), 1, -1).astype(np.int8)
    values = [spectrum_observables(signs[:, position]) for position in LONG_RANGE_POSITIONS]
    result: dict[str, Any] = {
        "K_f": float(np.mean([item[0] for item in values])),
        "K_f_exc": float(np.mean([item[0] for item in values]) - 1.0),
        "Gamma_1": float(np.mean([item[1] for item in values])),
    }
    target = exact_targets.bool()
    for name, indices in subsets.items():
        correct = predicted[indices] == target[indices]
        result[f"{name}_A_block"] = float(correct.all(dim=1).float().mean())
        result[f"{name}_A_block_tail"] = float(correct[:, LONG_RANGE_POSITIONS].all(dim=1).float().mean())
        result[f"{name}_bit_accuracy"] = float(correct.float().mean())
    return result


def subset_accuracy_of_starting_model(seed: int, threads: int) -> list[dict[str, Any]]:
    """Train the two branches with 1,216 added inputs of one starting model; one row per saved update.

    Runs in a worker process.  The four subsets are the same for both
    branches; their column prefixes are full_ (all 2,048 inputs), old_ (the
    64 base inputs), injected_ (the 1,216 inputs the distinct branch adds)
    and unseen_ (the 768 inputs in neither, on which neither branch trains).
    """
    # Thread count first: it fixes the trajectory of both branches.
    pin_threads(threads)
    start = prepare_starting_model(seed, ADDITION_SIZES)
    remainder = torch.as_tensor(
        sorted(set(start.all_codes.tolist()) - set(start.base_codes.tolist())
               - set(start.nested_codes.tolist())),
        dtype=torch.long,
    )
    subsets = {
        "full": start.all_codes,
        "old": start.base_codes,
        "injected": start.nested_codes,
        "unseen": remainder,
    }
    full = max(ADDITION_SIZES)
    wanted = {f"random_plus{full}", f"duplicate_plus{full}"}
    uniforms = minibatch_uniforms(seed, ENDPOINT_UPDATE)
    rows: list[dict[str, Any]] = []
    for spec in start.specs:
        if spec["arm"] not in wanted:
            continue
        for step, model in train_branch(start, spec["codes"], uniforms, SAVED_UPDATES):
            metrics = evaluate_subsets(model, start.exact_inputs, start.exact_targets, subsets)
            rows.append({"seed": seed, "arm": spec["arm"], "step": step, **metrics})
    return rows


def branch_subset_accuracy(args: argparse.Namespace) -> None:
    """K_f, Gamma_1 and accuracy on input subsets along the two branches with 1,216 added inputs.

    For each of the 25 starting models of data/branching_study/protocol.json the
    distinct branch (``arm`` random_plus1216) and the duplicate branch
    (duplicate_plus1216) are trained for 800 updates with the start weights,
    optimizer and minibatch stream of the study.  At the updates 0, 5, 10,
    15, 20, 25, 50, 100, 200, 400 and 800 each is evaluated on all 2,048
    inputs: K_f and Gamma_1 (long-range mean) and, on four subsets of inputs
    (``subset_accuracy_of_starting_model``), block accuracy, block accuracy
    at the four long-range positions and bit accuracy.  The paper quotes from
    this table the Gamma_1 contrast at update 25 and the block accuracy of the
    distinct branch on the added and on the never-trained inputs (Sec. 5.1).

    Reads   data/branching_study/protocol.json
            data/small_data_models/L12_N64_seed<k>/
    Writes  data/branching_study/checkpoints_added1216.csv, one row
            per starting model, branch and saved update (550 rows), sorted by
            seed, arm and step: seed, arm, step, K_f, K_f_exc, Gamma_1,
            <set>_A_block, <set>_A_block_tail, <set>_bit_accuracy

    A call that finds the table trains nothing; ``--overwrite`` trains again
    and writes over it.  The 50 branches take
    about a quarter of an hour on one thread.
    """
    check_workers(args)
    if SUBSET_TABLE.exists() and not args.overwrite:
        print(f"{SUBSET_TABLE.relative_to(ROOT)} present; nothing trained", flush=True)
        return
    BRANCHING_STUDY.mkdir(parents=True, exist_ok=True)
    seeds = selected_seeds()
    results = for_each_starting_model(subset_accuracy_of_starting_model, seeds, args.workers,
                                   args.threads_per_worker)
    rows = [row for seed_rows in results.values() for row in seed_rows]
    rows.sort(key=lambda row: (int(row["seed"]), str(row["arm"]), int(row["step"])))
    write_csv(SUBSET_TABLE, rows)
    print(f"wrote {len(rows)} rows to {SUBSET_TABLE.relative_to(ROOT)}", flush=True)


# ==========================================================================
# branch_independent_orderings: three orderings of the added inputs (App. B.1)
# ==========================================================================

ORDERING_RECORDS = BRANCHING_STUDY / "independent_orderings"
ORDERING_TABLE = BRANCHING_STUDY / "independent_orderings.csv"
# The ordering of the study (INTERVENTION_DATA_SEED) and two independent ones.
ORDERING_SEEDS = (20260822, 20260873, 20260924)
ORDERING_ADDITIONS = (192, 576, 1216)
ORDERING_STARTING_MODELS = 8


def ordering_starting_models() -> list[int]:
    """Eight of the 25 starting models, at equally spaced ranks of their K_f at the branch point.

    The ranks are round(linspace(0, 24, 8)) of the starting models of
    data/branching_study/protocol.json sorted by ``screened_start_metrics`` K_f.
    """
    protocol = json.loads((BRANCHING_STUDY / "protocol.json").read_text())
    selected = [int(seed) for seed in protocol["selected_seeds"]]
    screened = protocol["screened_start_metrics"]
    selected.sort(key=lambda seed: float(screened[str(seed)]["K_f"]))
    positions = np.linspace(0, len(selected) - 1, ORDERING_STARTING_MODELS).round().astype(int)
    return [selected[int(position)] for position in positions]


def ordering_record(seed: int, ordering_seed: int, threads: int) -> list[dict[str, Any]]:
    """The six branches (distinct, duplicate at 192, 576, 1,216) of one starting model under one ordering.

    The ordering of the added distinct inputs and the draws of the repeated
    base inputs both come from ``ordering_seed`` (``prepare_starting_model``).
    Each branch is evaluated with ``evaluate_model`` at the branch point and
    after 800 updates.  One row per branch, with the path and SHA-256 of the
    starting model and the hash of this protocol.
    """
    # Thread count first: it fixes the trajectory of every branch.
    pin_threads(threads)
    start = prepare_starting_model(seed, ORDERING_ADDITIONS, ordering_seed=ordering_seed)
    wanted = {f"{kind}_plus{addition}" for addition in ORDERING_ADDITIONS for kind in ("random", "duplicate")}
    uniforms = minibatch_uniforms(seed, ENDPOINT_UPDATE)
    source = {
        "source_checkpoint": start.run_dir.relative_to(ROOT).as_posix(),
        "source_model_sha256": hashlib.sha256((start.run_dir / "model.pt").read_bytes()).hexdigest(),
        "base_codes_sha256": codes_digest(start.base_codes),
        "nested_codes_sha256": codes_digest(start.nested_codes),
        "protocol_hash": stable_hash({
            "pool_seed": ordering_seed,
            "steps": ENDPOINT_UPDATE,
            "batch_size": BATCH,
            "doses": ORDERING_ADDITIONS,
            "optimizer": "AdamW",
            "uniform_seed": MINIBATCH_SEED + 7919 * seed,
        }),
    }
    rows = []
    for spec in start.specs:
        if spec["arm"] not in wanted:
            continue
        values = {}
        for step, model in train_branch(start, spec["codes"], uniforms, (0, ENDPOINT_UPDATE)):
            values[step] = evaluate_model(model, start.exact_inputs, start.exact_targets)[0]
        begin, end = values[0], values[ENDPOINT_UPDATE]
        rows.append({
            "seed": seed, "pool_seed": ordering_seed, "arm": spec["arm"], **source,
            "K_f_start": begin["K_f"], "K_f_end": end["K_f"],
            "Gamma_1_start": begin["Gamma_1"], "Gamma_1_end": end["Gamma_1"],
            "A_block_start": begin["A_block"], "A_block_end": end["A_block"],
            "bit_accuracy_start": begin["bit_accuracy"], "bit_accuracy_end": end["bit_accuracy"],
        })
    return rows


def branch_independent_orderings(args: argparse.Namespace) -> None:
    """Distinct and duplicate branches under three independent orderings of the added inputs (App. B.1).

    The distinct additions of the study take the first n inputs of one random
    ordering of the 1,984 inputs outside the base set.  Here the branches
    with 192, 576 and 1,216 added inputs are trained under that ordering and
    under two orderings drawn from independent seeds (``ORDERING_SEEDS``),
    for eight of the 25 starting models at equally spaced ranks of K_f
    (``ordering_starting_models``): 24 pairs of branches at each size.  The
    start weights, optimizer and minibatch stream are those of the study;
    the seed of an ordering also draws the repeated base inputs of the
    duplicate branches.

    Reads   data/branching_study/protocol.json (the starting models and
            their K_f at the branch point)
            data/small_data_models/L12_N64_seed<k>/
    Writes  data/branching_study/
              independent_orderings/pool_<ordering seed>_seed_<XXX>.json
                  one record per ordering and starting model: its six branches
              independent_orderings.csv
                  the records as one table, 144 rows sorted by ordering,
                  starting model and branch

    Columns: pool_seed (seed of the ordering), seed (starting model), arm,
    K_f, Gamma_1, A_block, bit_accuracy with suffix _start and _end, path
    and SHA-256 of the starting model, SHA-256 of the base inputs and of the
    ordering, and the hash of the protocol of the record.  The table is built
    from the records as stored (keys sorted), so its columns are in
    alphabetical order whichever records this call trained.

    A call that finds the table trains nothing.  Otherwise a record that is present is not trained again, so an
    interrupted call can be repeated; ``--overwrite`` trains all 24 records
    again.  The 144 branches take about half an hour on one thread.
    """
    if args.threads < 1:
        raise SystemExit("--threads must be positive")
    if ORDERING_TABLE.exists() and not args.overwrite:
        print(f"{ORDERING_TABLE.relative_to(ROOT)} present; nothing trained", flush=True)
        return
    seeds = ordering_starting_models()
    ORDERING_RECORDS.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for ordering_seed in ORDERING_SEEDS:
        for seed in seeds:
            path = ORDERING_RECORDS / f"pool_{ordering_seed}_seed_{seed:03d}.json"
            if args.overwrite or not path.exists():
                write_json(path, ordering_record(seed, ordering_seed, args.threads))
            # Read back as stored, so that the rows and their key order do not
            # depend on which records this call trained.
            rows.extend(json.loads(path.read_text()))
            print(f"completed ordering {ordering_seed}, seed {seed}", flush=True)
    rows.sort(key=lambda row: (row["pool_seed"], row["seed"], row["arm"]))
    write_csv(ORDERING_TABLE, rows)
    print(f"wrote {len(rows)} rows to {ORDERING_TABLE.relative_to(ROOT)}", flush=True)


# ==========================================================================
# Command line
# ==========================================================================

def add_overwrite(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--overwrite", action="store_true",
                        help="train again and write over outputs that are present")


def configure_large_data_trajectories(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seeds", type=parse_int_list, default=(0, 1, 2, 3, 4, 5),
                        help="comma-separated seeds of the runs (default 0,1,2,3,4,5)")
    parser.add_argument("--out-dir", default="data/large_data_trajectories",
                        help="directory of the runs, relative to the repository root "
                             "(default data/large_data_trajectories)")
    parser.add_argument("--threads", type=int, default=6, help="torch threads (default 6)")
    add_overwrite(parser)


def configure_threads(default: int) -> Callable[[argparse.ArgumentParser], None]:
    def configure(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--threads", type=int, default=default,
                            help=f"torch threads (default {default})")
        add_overwrite(parser)
    return configure


def configure_small_data_model(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, required=True,
                        help="initialization and minibatch-order seed")
    parser.add_argument("--threads", type=int, default=3, help="torch threads (default 3)")
    add_overwrite(parser)


def configure_small_data_ensemble(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-parallel", type=int, default=2,
                        help="networks trained at a time (default 2)")
    parser.add_argument("--threads", type=int, default=3,
                        help="torch threads of each network (default 3)")
    add_overwrite(parser)


def configure_workers(parser: argparse.ArgumentParser) -> None:
    add_worker_arguments(parser)
    add_overwrite(parser)


# (name, function, what it writes, paper location, options), in the order in
# which data/ is produced: a subcommand reads only what the ones above it write.
SUBCOMMANDS: list[tuple[str, Callable[[argparse.Namespace], None], str, str,
                        Callable[[argparse.ArgumentParser], None]]] = [
    ("large_data_trajectories", large_data_trajectories,
     "data/large_data_trajectories/{checkpoints/seed<k>/, monitoring_seeds<first>_<last>.csv}",
     "Table 1, Figs. 2-4, 9-11, Secs. 4, 8",
     configure_large_data_trajectories),
    ("monitoring_from_checkpoints", monitoring_from_checkpoints,
     "data/large_data_trajectories/monitoring_seeds06_08.csv", "Table 1, Sec. 4.3",
     configure_threads(4)),
    ("small_data_model", small_data_model,
     "data/small_data_models/L12_N64_seed<seed>/", "starting models of Secs. 5-7",
     configure_small_data_model),
    ("small_data_ensemble", small_data_ensemble,
     "data/small_data_models/", "starting models of Secs. 5-7",
     configure_small_data_ensemble),
    ("small_data_trajectories", small_data_trajectories,
     "data/small_data_trajectories/trajectories.json", "Fig. 3, Sec. 4.2",
     configure_threads(TRAJECTORY_THREADS)),
    ("branching_study", branching_study,
     "data/branching_study/{protocol.json, records/, trajectory_metrics.csv}",
     "Figs. 5-7, Secs. 5-7", configure_workers),
    ("branch_endpoint_sign_tables", branch_endpoint_sign_tables,
     "data/branching_study/endpoint_sign_tables.npz", "Figs. 6, 7, Secs. 5.2, 6", configure_workers),
    ("branch_checkpoint_sign_tables", branch_checkpoint_sign_tables,
     "data/branching_study/checkpoint_sign_tables.npz", "Figs. 5, 8, Sec. 7", configure_workers),
    ("branch_subset_accuracy", branch_subset_accuracy,
     "data/branching_study/checkpoints_added1216.csv", "Fig. 5, Sec. 5.1", configure_workers),
    ("branch_independent_orderings", branch_independent_orderings,
     "data/branching_study/independent_orderings{.csv, /}", "App. B.1",
     configure_threads(2)),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the networks of the paper; each subcommand writes one part of data/.",
        epilog="Run with --list for the subcommands, SUBCOMMAND --help for one of them.",
    )
    parser.add_argument("--list", action="store_true",
                        help="list the subcommands with the data/ they write and stop")
    subparsers = parser.add_subparsers(dest="subcommand", metavar="SUBCOMMAND")
    functions = {}
    for name, function, writes, paper, configure in SUBCOMMANDS:
        summary = (function.__doc__ or "").strip().splitlines()[0]
        sub = subparsers.add_parser(
            name, help=summary, description=inspect.cleandoc(function.__doc__ or ""),
            epilog=f"Writes {writes} ({paper}).",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        configure(sub)
        functions[name] = function
    args = parser.parse_args(argv)
    if args.list:
        for name, function, writes, paper, _configure in SUBCOMMANDS:
            summary = (function.__doc__ or "").strip().splitlines()[0]
            print(f"{name:30s} -> {writes}\n{'':30s} {paper}\n{'':30s} {summary}")
        return 0
    if args.subcommand is None:
        parser.error("name a subcommand, or use --list")
    functions[args.subcommand](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
