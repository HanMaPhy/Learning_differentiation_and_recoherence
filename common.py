"""Shared core of the analyses, the figures and the training protocols.

This file holds what more than one of ``analysis.py``, ``figures.py`` and
``train.py`` needs: the directories of the repository, the parity task on its
full binary space, the network, the function-state observables, the
constraint graph of a training set and the pair decomposition of the reduced
state, the protocol constants and data loaders of the branching study, and
the tables of the retreat analyses.  It has no command line; the three files
above import from it.  ``function_state.py`` is a separate, self-contained
account of the same quantities and imports nothing from here.

THE TASK
--------
An input is a string x = (x_1, ..., x_D) of D = L - 1 = 11 bits, labelled by
the integer code x = sum_i x_i 2^(i-1), so that bit i of the code is x_i.  The
target at output position j = 0, ..., D is the running parity

    p_j(x) = x_1 XOR ... XOR x_j,        p_0 = 0.

Read the L = 12 output positions as the sites of a chain and the bits as its
bonds: x_j = p_(j-1) XOR p_j is a domain wall, and the task is to reconstruct
the spin configuration p from its domain walls with the first spin fixed.  The
network sees two channels per site, channel 0 of site j carries the bond x_j
to its left and channel 1 the bond x_(j+1) to its right, so every site sees
only its two bonds and the running sum has to be built by the network.

For L = 12 the input space has 2^11 = 2,048 strings, so a network can be
evaluated on every input there is (``full_binary_space``).  A training set of N
inputs is drawn with replacement from one seed, ``TRAIN_SEED``; the set of N
inputs is the first N rows of that one stream, so the 64 inputs of the
small-data runs are the first 64 of the 1,280 inputs of the large-data runs.

THE FUNCTION STATE
------------------
At one output position the network's sign h(x) = sign(logit(x)) in {-1, +1}
over all inputs is its sign table.  Cutting the string after d_L = CUT = 5
bits, x = (a, b) with the left substring a = x mod 32 and the right substring
b = x div 32, turns the table into a 32 x 64 matrix M(a, b) = h(a + 32 b).
Row a is the conditional function of the left substring a.  With the
normalized rows phi_a(b) = M(a, b) / sqrt(64),

    C_h(a, a') = sum_b phi_a(b) phi_a'(b),        rho_A = C_h / 32,

and rho_A is the reduced state of the function state across the cut, with
unit trace.  Its eigenvalues are the Schmidt spectrum of the sign table.

    K_f       = 1 / Tr rho_A^2, the effective Schmidt rank: 1 for a single
                conditional mode, up to 32 when the rows are orthogonal.
                Its rise is differentiation, its fall recoherence.

Read M(a, b) as a Z2 link variable on the complete bipartite graph between
left and right substrings.  Row and column sign flips M -> e_a M f_b are gauge
transformations on its vertices: they leave the spectrum unchanged but move
the weight on any fixed mode.  The crossing gauge

    M_x(a, b) = M(a, b) M(a, 0) M(0, b) M(0, 0)

is the Wilson loop around the plaquette a -> b -> 0_A -> 0_B -> a and is
invariant under every flip.  With C^x_h and rho_A^x built from M_x,

    Gamma_1   = mean over (a, a') of C^x_h(a, a') = <u_uni| rho_A^x |u_uni>,
                the weight on the uniform left mode: how strongly the gauged
                conditional functions share one direction.

The parity target is pure gauge across the cut, t(a, b) = u(a) v(b) at every
output position (for j <= CUT it depends on a alone).  All its Wilson loops
are +1, so the crossing gauge removes it: K_f and Gamma_1 are the same for h
and for h t, the sign table multiplied entry by entry by the target signs
t(x).  We call h -> h t the target gauge.  The population of the target's
own left mode u_star(a) = u(a) / sqrt(32) is not invariant under it:

    w_star    = <u_star| rho_A |u_star> = mean over (a, a') of the Gram matrix
                of h t, with no crossing gauge.  It is 1 exactly when
                h(a, b) = u(a) g(b) for one function g, that is, when the left
                dependence of h is that of the target.

Sign tables are therefore stored raw, sign(logit), and every analysis applies
the target gauge at the point of use.  Gauging twice returns the raw table,
since t t = 1, and turns w_star into the ungauged Gram mean, an error that no
spectral quantity can detect.

The retreat analyses also cluster the 32 gauged rows (k-means with k-means++
seeding, 12 starts, at most 80 iterations).  For a partition into clusters c
of sizes n_c the cluster coherence is

    Gamma(n) = sum_c (n_c / 32) | (1/n_c) sum_(a in c) phi^x_a |^2,

with Gamma(1) = Gamma_1, and the number of clusters is the smallest n with
Gamma(n) >= 0.90 (at most 8).  Between two checkpoints with cluster labels
T(e0), T(e1) the splitting entropy is B = H[T(e1) | T(e0)] in bits, and the
overlap of the leading subspaces is R_chi = Tr[P_chi(e0) P_chi(e1)] / chi.

THE CONSTRAINT GRAPH
--------------------
A training set is a bipartite graph on 32 left and 64 right vertices with one
edge (a, b) per distinct training input.  A repeated input repeats an edge,
so adding distinct inputs is bond percolation on this graph.  A path of edges
fixes the relative target sign of two left substrings, so the data identify
the relation between a and a' exactly when they share a component.  With
J(a, a') = 1 for a pair in a common component,

    f_L    = fraction of ordered pairs a != a' with J = 1,
    K_perc = 1 / (1/32 + (31/32) f_L),

the inverse purity of rho_perc = J / 32, the state that occupies every
identifiable relation with full coherence and nothing else.

Splitting the off-diagonal pairs by J gives the exact identities

    1 / K_f = 1/32 + (31/32) [f_L Q_in + (1 - f_L) Q_out],
    w_star  = 1/32 + (31/32) [f_L R_in + (1 - f_L) R_out],

with Q the mean of C_h^2 and R the mean of the target-gauged correlation over
the pairs inside (in) and between (out) components.  The graph fixes which
pairs the data can tie; the trained network fixes how strongly they are
correlated (Q) and whether the correlation has the sign of the target (R).

THE BRANCHING STUDY
-------------------
A starting model is a network trained on the N = 64 base inputs until it
fits them.  Its weights are copied into branches that continue for 800 updates
with the same optimizer, zero optimizer state at the branch point and one
stream of minibatch coordinates; the branches differ only in the data added
to the base set: nothing (control), the first n inputs of one random ordering
of the 1,984 inputs outside the base set (distinct), or n repeated base inputs
(duplicate).  The records also hold branches that add 64 inputs on which
the starting model contradicts the target or agrees with it.  No analysis
reads them.  ``BRANCHING_STUDY`` holds the records of every branch at the
updates 0, 25, 50, 100, 200, 400 and 800, the sign tables of the distinct and
duplicate branches at update 800, and the sign tables of the two branches
with 1,216 added inputs at the 11 updates of ``SAVED_UPDATES``.

Paper location                         where it is computed here
--------------                         -------------------------
Sec. 2.1, task, full binary space      make_set, training_codes, full_binary_space, target_signs
Sec. 2.2, rho_A, crossing gauge, K_f   conditional_rows, reduced_state, effective_rank
Sec. 2.3, Gamma_1, w_star              rank_and_coherence, target_population
App. B, network (26,529 parameters)    ChainTransformer, ARCHITECTURE
Secs. 5, 6, 7, branching study         addition_order, selected_seeds, load_*_sign_tables
Sec. 6.1, f_L, K_perc                  constraint_graph
Sec. 7.1, Q, R                         pair_averages, pair_matrices
Sec. 8.2, App. B.2, B and R_chi        cluster_conditional_functions, splitting_entropy,
                                       subspace_overlap
App. B, bootstrap intervals            bootstrap
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ==========================================================================
# Directories
# ==========================================================================

# Every path is taken from the location of this file, so the code gives the
# same result from any working directory.  ``data/`` may be a symbolic link;
# paths under it are never resolved, so the paths written into the tables are
# relative to the repository root.
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"            # trained networks and training records (read only)
RESULTS = ROOT / "results"      # tables written by analysis.py
FIGURE = ROOT / "figure"        # figures written by figures.py

LARGE_DATA_TRAJECTORIES = DATA / "large_data_trajectories"  # runs at N = 1,280: monitoring tables, weights from epoch 20
SMALL_DATA_MODELS = DATA / "small_data_models"              # 100 networks at N = 64, the candidate starting models
SMALL_DATA_TRAJECTORIES = DATA / "small_data_trajectories"  # 24 runs at N = 64 with the state at 52 steps
BRANCHING_STUDY = DATA / "branching_study"                  # records, tables and sign tables of the branching study
ENDPOINT_SIGN_TABLES = BRANCHING_STUDY / "endpoint_sign_tables.npz"      # sign tables at update 800, every addition
CHECKPOINT_SIGN_TABLES = BRANCHING_STUDY / "checkpoint_sign_tables.npz"  # sign tables at the 11 saved updates, +1,216

EVENTS = RESULTS / "retreat_events" / "events.csv"
TRAJECTORIES = RESULTS / "retreat_events" / "trajectories.csv"
EVENT_SIGN_TABLES = RESULTS / "event_sign_tables"

# ==========================================================================
# The task
# ==========================================================================

L = 12                          # output positions, sites of the chain
D = L - 1                       # input bits, bonds of the chain
N_STRINGS = 2**D                # 2,048 strings, the full binary space
CUT = 5                         # d_L, bits in the left substring
N_LEFT = 2**CUT                 # 32 left substrings
N_RIGHT = 2 ** (D - CUT)        # 64 right substrings
N_NODES = N_LEFT + N_RIGHT      # 96 vertices of the constraint graph
OFFDIAG_COUNT = N_LEFT * (N_LEFT - 1)   # 992 ordered pairs a != a'
LONG_RANGE_POSITIONS = (8, 9, 10, 11)   # the long-range outputs every average runs over

TRAIN_SEED = 12345              # seed of the training set, the same for every run
BASE_N = 64                     # training-set size of the starting models


def make_set(num, L, seed):
    """Return ``num`` inputs drawn from ``seed``: inputs [num, 2, L], 0/1 targets [num, L].

    The bits are uniform and independent, drawn with replacement, so a set can
    repeat a string.  Channel 0 of position j carries bit j (the bond to its
    left), channel 1 bit j + 1 (the bond to its right); the target at position
    j is the parity of the first j bits.  A set of N inputs from ``TRAIN_SEED``
    is the first N rows of the one stream of that seed.
    """
    g = torch.Generator().manual_seed(seed)
    s = torch.randint(0, 2, (num, L - 1), generator=g).float()
    z = torch.zeros(num, L); z[:, 1:] = torch.cumsum(s, 1).remainder(2.0)
    x = torch.zeros(num, 2, L); x[:, 0, 1:] = s; x[:, 1, :-1] = s
    return x, z


def training_codes(num: int, seed: int = TRAIN_SEED) -> torch.Tensor:
    """Codes of the ``num`` inputs of ``make_set(num, L, seed)``, as a LongTensor.

    The generator is consumed exactly as in ``make_set``, so the code of row k
    is the input of row k there.  Integer arithmetic throughout.
    """
    generator = torch.Generator().manual_seed(seed)
    bits = torch.randint(0, 2, (num, D), generator=generator)
    return bits @ (2 ** torch.arange(D, dtype=torch.long))


def base_training_codes() -> torch.Tensor:
    """Codes of the 64 base inputs the starting models are trained on (with repeats)."""
    return training_codes(BASE_N)


def full_binary_space() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Codes, network inputs and 0/1 targets of all 2**D inputs.

    Row x is the input whose bits spell the integer x, so the cut of the
    function state is a reshape.  Returns ``codes`` (2048,), ``inputs`` float32
    (2048, 2, L) and ``targets`` float32 (2048, L), targets[:, 0] = 0.
    """
    codes = torch.arange(N_STRINGS, dtype=torch.long)
    bits = ((codes[:, None] >> torch.arange(D)[None, :]) & 1).float()
    targets = torch.zeros(N_STRINGS, L)
    targets[:, 1:] = torch.cumsum(bits, dim=1).remainder(2.0)
    inputs = torch.zeros(N_STRINGS, 2, L)
    inputs[:, 0, 1:] = bits
    inputs[:, 1, :-1] = bits
    return codes, inputs, targets


def target_signs() -> np.ndarray:
    """Target signs t(x) at every output position, int8 of shape (2**D, L).

    +1 for a target bit of one, -1 for zero, in the convention of the sign
    tables (+1 for a logit >= 0).  Column 0 is identically -1.
    """
    _codes, _inputs, targets = full_binary_space()
    return np.where(targets.bool().numpy(), 1, -1).astype(np.int8)


def long_range_target_signs() -> np.ndarray:
    """Target signs at the four long-range positions, int8 of shape (4, 2**D)."""
    return target_signs()[:, LONG_RANGE_POSITIONS].T


# ==========================================================================
# The network
# ==========================================================================

# Width, depth and heads of every network of the paper: 26,529 parameters (App. B).
ARCHITECTURE = dict(channels=16, depth=8, num_heads=4, ffn_dim=64)


class TransformerBlock(nn.Module):
    """Pre-norm encoder block: LayerNorm -> attention -> residual, LayerNorm -> FFN -> residual."""

    def __init__(self, channels: int, num_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        # The order in which the submodules are created fixes the order in
        # which they draw their initial weights from torch's global generator,
        # and their names fix the keys of the stored state dictionaries.
        self.norm_attn = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            channels, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_ffn = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, channels),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, L, C]; returns [B, L, C]."""
        h_norm = self.norm_attn(h)
        attn_out, _ = self.attn(h_norm, h_norm, h_norm, need_weights=False)
        h = h + self.dropout(attn_out)
        h = h + self.ffn(self.norm_ffn(h))
        return h


class ChainTransformer(nn.Module):
    """Encoder-only transformer over the L sites of the chain, one logit per site.

    The two input channels of each site are projected to ``channels`` and
    summed with a learned positional embedding of shape [1, L, channels]; the
    sequence passes through ``depth`` pre-norm blocks with bidirectional
    attention (no mask) and a final LayerNorm, and a linear head gives one
    logit per site.  Input [B, 2, L], output [B, L].  Every network of the
    paper has the sizes of ``ARCHITECTURE`` and dropout 0.

    The order of creation below (input projection, positional embedding and
    its normal initialization, blocks, output norm, head) is the order of the
    draws from torch's global generator, so it fixes the initial weights a
    seed gives.
    """

    def __init__(
        self,
        L: int,
        channels: int,
        depth: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_heads ({num_heads})"
            )
        self.in_proj = nn.Linear(2, channels)
        self.pos_embed = nn.Parameter(torch.zeros(1, L, channels))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(channels, num_heads, ffn_dim, dropout) for _ in range(depth)]
        )
        self.norm_out = nn.LayerNorm(channels)
        self.out_proj = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 2, L]; returns logits [B, L]."""
        # [B, 2, L] -> [B, L, 2] -> [B, L, C]
        h = self.in_proj(x.transpose(1, 2)) + self.pos_embed
        for block in self.blocks:
            h = block(h)
        h = self.norm_out(h)
        return self.out_proj(h).squeeze(-1)


# ==========================================================================
# Function-state observables of one sign table
# ==========================================================================

# numpy 2.0 on Apple Accelerate raises divide-by-zero, overflow and invalid
# warnings from a float64 matmul whose operands and result are entirely
# finite.  The Gram products below are that case; the flags are silenced
# where they are raised.
_GRAM_ERRSTATE = {"divide": "ignore", "over": "ignore", "invalid": "ignore"}


def conditional_rows(sign_table: np.ndarray) -> np.ndarray:
    """Crossing-gauged conditional functions phi^x_a, unit rows of shape (32, 64).

    ``sign_table`` is one output's sign table over the 2**D inputs, indexed by the
    input code.  After the reshape and transpose the low CUT bits of the code
    (the left substring) label the rows and the high bits the columns.  The
    gauge multiplies by the first column, the first row and the corner, in
    that order, and the rows are then divided by sqrt(64).
    """
    matrix = np.asarray(sign_table, dtype=np.float64).reshape(N_RIGHT, N_LEFT).T
    matrix = matrix * matrix[:, :1] * matrix[:1, :] * matrix[0, 0]
    return matrix / math.sqrt(N_RIGHT)


def reduced_state(sign_table: np.ndarray) -> np.ndarray:
    """rho_A^x = C^x_h / 32 of one sign table, with unit trace."""
    rows = conditional_rows(sign_table)
    with np.errstate(**_GRAM_ERRSTATE):
        return rows @ rows.T / float(N_LEFT)


def effective_rank(sign_table: np.ndarray) -> float:
    """K_f = 1 / Tr rho_A^2 of one sign table (Sec. 2.2).

    The purity is the sum of the squared eigenvalues of rho_A, clipped at
    zero, summed in the ascending order eigvalsh returns.  K_f does not
    depend on row or column flips, hence not on the target gauge either.
    """
    eigenvalues = np.clip(np.linalg.eigvalsh(reduced_state(sign_table)), 0.0, None)
    return 1.0 / float(np.square(eigenvalues).sum())


def rank_and_coherence(sign_table: np.ndarray) -> tuple[float, float]:
    """K_f and Gamma_1 of one sign table (Secs. 2.2, 2.3).

    Gamma_1 is the mean of the crossing-gauged Gram matrix C^x_h, the weight
    of rho_A^x on the uniform left mode and the cluster coherence of a single
    cluster.  Both are invariant under the target gauge, so the raw and the
    target-gauged table give the same values.
    """
    rows = conditional_rows(sign_table)
    with np.errstate(**_GRAM_ERRSTATE):
        gram = rows @ rows.T
    return effective_rank(sign_table), float(gram.mean())


def target_population(sign_table: np.ndarray) -> float:
    """w_star of one target-gauged sign table h t: the mean of its Gram matrix (Sec. 2.3).

    No crossing gauge is applied: that gauge removes the target, which is the
    one direction w_star measures.  The caller multiplies the raw table by
    the target first; passing the raw table gives the ungauged Gram mean.
    """
    matrix = np.asarray(sign_table, dtype=np.float64).reshape(N_RIGHT, N_LEFT).T
    rows = matrix / math.sqrt(N_RIGHT)
    with np.errstate(**_GRAM_ERRSTATE):
        return float((rows @ rows.T).mean())


def projector(sign_table: np.ndarray, chi: int) -> np.ndarray:
    """Projector P_chi on the ``chi`` leading left singular vectors of the gauged rows."""
    u, _s, _vh = np.linalg.svd(conditional_rows(sign_table), full_matrices=False)
    q = min(chi, u.shape[1])
    return u[:, :q] @ u[:, :q].T


def subspace_overlap(a: np.ndarray, b: np.ndarray, chi: int) -> float:
    """R_chi = Tr(P_a P_b) / chi, the overlap of the leading subspaces of two sign tables (Sec. 8.2).

    One when the two reduced states share their chi leading modes, chi / 32
    on average for unrelated subspaces.
    """
    pa = projector(a, chi)
    pb = projector(b, chi)
    q = min(chi, pa.shape[0])
    with np.errstate(**_GRAM_ERRSTATE):
        return float(np.trace(pa @ pb) / q)


def kmeans(points: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Cluster labels of the best of 12 k-means++ starts, each of at most 80 iterations (App. B.2).

    ``points`` are the gauged rows in their singular basis (rows of U S),
    which leaves all distances unchanged.  The best start is the one with the
    smallest summed squared distance to the centers, which equals
    32 [1 - Gamma(n)].  All draws come from one generator seeded with ``seed``.
    """
    rng = np.random.default_rng(seed)
    if n == 1:
        return np.zeros(len(points), dtype=int)
    best = None
    best_loss = float("inf")
    sq = np.square(points).sum(axis=1)
    for _ in range(12):
        # k-means++ seeding: each further center drawn with probability
        # proportional to the squared distance to the nearest chosen one.
        centers = np.empty((n, points.shape[1]))
        centers[0] = points[rng.integers(len(points))]
        dist = np.square(points - centers[0]).sum(axis=1)
        for j in range(1, n):
            total = float(dist.sum())
            if total <= 1e-14:
                centers[j] = points[rng.integers(len(points))]
            else:
                centers[j] = points[min(np.searchsorted(np.cumsum(dist), rng.random() * total), len(points) - 1)]
            dist = np.minimum(dist, np.square(points - centers[j]).sum(axis=1))
        for _it in range(80):
            with np.errstate(**_GRAM_ERRSTATE):
                d = sq[:, None] - 2 * points @ centers.T + np.square(centers).sum(axis=1)[None, :]
            labels = d.argmin(axis=1)
            new = centers.copy()
            for c in range(n):
                if np.any(labels == c):
                    new[c] = points[labels == c].mean(axis=0)
            if np.allclose(new, centers, atol=1e-12):
                centers = new
                break
            centers = new
        with np.errstate(**_GRAM_ERRSTATE):
            d = sq[:, None] - 2 * points @ centers.T + np.square(centers).sum(axis=1)[None, :]
        labels = d.argmin(axis=1)
        loss = float(np.maximum(d[np.arange(len(points)), labels], 0).sum())
        if loss < best_loss:
            best_loss, best = loss, labels.copy()
    return best


def cluster_coherence(gram: np.ndarray, labels: np.ndarray) -> float:
    """Cluster coherence Gamma(n): the Gram mass inside each cluster over its size, averaged over rows.

    For a single cluster this is the mean of the whole Gram matrix, Gamma_1.
    With unit rows it reaches one when every cluster holds identical rows.
    """
    total = 0.0
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        total += float(gram[np.ix_(members, members)].sum()) / len(members)
    return total / len(labels)


def cluster_conditional_functions(sign_table: np.ndarray, seed: int, threshold: float = 0.90) -> tuple[np.ndarray, int, float]:
    """Clustering with the smallest n <= 8 whose cluster coherence reaches ``threshold`` (App. B.2).

    n runs from 1 to 8; the clustering at n draws from seed + 17 n.  With no
    n reaching the threshold the clustering at n = 8 is kept.  Returns the
    labels of the 32 left substrings, n and Gamma(n).  The paper uses
    threshold 0.90 and checks 0.85 and 0.95 (Fig. 11).
    """
    rows = conditional_rows(sign_table)
    u, s, _vh = np.linalg.svd(rows, full_matrices=False)
    points = u * s[None, :]
    with np.errstate(**_GRAM_ERRSTATE):
        gram = rows @ rows.T
    chosen = None
    for n in range(1, 9):
        labels = kmeans(points, n, seed + 17 * n)
        coherence = cluster_coherence(gram, labels)
        if coherence >= threshold:
            chosen = (labels, n, coherence)
            break
    if chosen is None:
        labels = kmeans(points, 8, seed + 17 * 8)
        chosen = (labels, 8, cluster_coherence(gram, labels))
    return chosen


def splitting_entropy(parent: np.ndarray, child: np.ndarray) -> float:
    """B = H[T(child) | T(parent)] in bits between two clusterings of the left substrings (Sec. 8.2).

    The entropy of the child labels inside each parent cluster, weighted by
    the size of the parent cluster.  Zero when no parent cluster splits.
    Labels are local to each checkpoint, so the labels of the two clusterings
    need no correspondence.
    """
    total = 0.0
    for p in np.unique(parent):
        mask = parent == p
        probs = np.bincount(child[mask], minlength=max(child) + 1).astype(float)
        probs /= probs.sum()
        entropy = -sum(float(x) * math.log2(float(x)) for x in probs if x > 0)
        total += mask.mean() * entropy
    return float(total)


# ==========================================================================
# The branching study: protocol constants and stored sign tables
# ==========================================================================

INTERVENTION_DATA_SEED = 20260822              # seed of the ordering of the added distinct inputs
ADDITION_SIZES = (16, 32, 64, 192, 576, 1216)  # numbers of added inputs
BRANCHES = ("distinct", "duplicate")           # branch axis of the stored sign tables
ENDPOINT_UPDATE = 800                          # last update of every branch
SAVED_UPDATES = (0, 5, 10, 15, 20, 25, 50, 100, 200, 400, 800)   # saved updates of the +1,216 branches


def addition_order(base_codes: torch.Tensor, additions: Iterable[int],
                   seed: int = INTERVENTION_DATA_SEED) -> torch.Tensor:
    """One random ordering of the inputs outside the base set, up to the largest addition.

    The 1,984 codes absent from ``base_codes`` are shuffled by a generator
    seeded with ``seed``; an addition of n inputs takes the first n entries,
    so the added sets of different sizes are nested.  The study uses
    ``INTERVENTION_DATA_SEED``; App. B.1 repeats it with two further seeds.
    """
    unseen = np.setdiff1d(
        np.arange(N_STRINGS, dtype=np.int64),
        torch.unique(base_codes).numpy(),
        assume_unique=True,
    )
    rng = np.random.default_rng(seed)
    rng.shuffle(unseen)
    return torch.from_numpy(unseen[: max(additions)].copy()).long()


def selected_seeds() -> list[int]:
    """Seeds of the 25 starting models of the study, in protocol order (``BRANCHING_STUDY/protocol.json``)."""
    with (BRANCHING_STUDY / "protocol.json").open(encoding="utf-8") as handle:
        protocol = json.load(handle)
    return [int(seed) for seed in protocol["selected_seeds"]]


def check_axis(data, key: str, expected) -> None:
    """Stop unless the axis ``key`` stored in a sign-table file lists ``expected``."""
    stored = data[key].tolist()
    if stored != list(expected):
        raise SystemExit(f"axis {key} of the sign tables is {stored}, expected {list(expected)}")


def load_endpoint_sign_tables() -> tuple[np.ndarray, list[int]]:
    """Raw output signs of every distinct and duplicate branch at update ``ENDPOINT_UPDATE``.

    Reads ``ENDPOINT_SIGN_TABLES``.  Returns ``signs`` int8 of
    shape (starting model, addition, branch, position, input), with the axes in
    ``selected_seeds()``, ``ADDITION_SIZES``, ``BRANCHES`` and
    ``LONG_RANGE_POSITIONS`` order, and the seeds.
    """
    seeds = selected_seeds()
    with np.load(ENDPOINT_SIGN_TABLES, allow_pickle=False) as data:
        for key, expected in (("seeds", seeds), ("doses", ADDITION_SIZES), ("arms", BRANCHES),
                              ("positions", LONG_RANGE_POSITIONS)):
            check_axis(data, key, expected)
        signs = data["signs"]
    return signs, seeds


def load_checkpoint_sign_tables() -> tuple[np.ndarray, pd.DataFrame, list[int]]:
    """Raw output signs of the two branches with 1,216 added inputs at the updates ``SAVED_UPDATES``.

    Reads ``CHECKPOINT_SIGN_TABLES``.  Returns ``signs`` int8 of
    shape (starting model, branch, update, position, input), with the axes in
    ``selected_seeds()``, ``BRANCHES``, ``SAVED_UPDATES`` and ``LONG_RANGE_POSITIONS``
    order; the observables stored with them (K_f, Gamma_1, accuracies), one
    row per starting model, branch and update; and the seeds.
    """
    seeds = selected_seeds()
    with np.load(CHECKPOINT_SIGN_TABLES, allow_pickle=False) as data:
        for key, expected in (("seeds", seeds), ("arms", BRANCHES), ("updates", SAVED_UPDATES),
                              ("positions", LONG_RANGE_POSITIONS)):
            check_axis(data, key, expected)
        signs = data["signs"]
        rows = json.loads(str(data["metrics_json"].item()))
    return signs, pd.DataFrame(rows), seeds


# ==========================================================================
# The constraint graph of a training set (Sec. 6.1)
# ==========================================================================

class UnionFind:
    """Disjoint-set forest with path halving and union by size."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, node: int) -> int:
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, first: int, second: int) -> None:
        first = self.find(first)
        second = self.find(second)
        if first == second:
            return
        if self.size[first] < self.size[second]:
            first, second = second, first
        self.parent[second] = first
        self.size[first] += self.size[second]


def distinct_branch_codes(addition: int) -> np.ndarray:
    """Distinct codes of the training set of the distinct branch with ``addition`` added inputs.

    The base set plus the first ``addition`` entries of ``addition_order``; for
    an addition of 0 the base set alone.
    """
    base = base_training_codes().numpy().astype(np.int64)
    if addition == 0:
        return np.unique(base)
    nested = addition_order(base_training_codes(), ADDITION_SIZES).numpy().astype(np.int64)
    return np.unique(np.concatenate((base, nested[:addition])))


def constraint_graph(codes: np.ndarray) -> tuple[dict[str, float | int], np.ndarray]:
    """Components and f_L of the constraint graph of a set of inputs, and the component of each left substring.

    Vertex a (0..31) is a left substring, vertex 32 + b a right substring;
    each distinct code adds the edge (a, 32 + b).  Returns the row

        components, d_free = components - 1, connected     the whole graph,
                                                           isolated vertices included
        c_left, d_free_left, left_connected                the components holding
                                                           left vertices
        f_left                                             f_L
        largest_left_component, largest_component          sizes

    and the component root of each of the 32 left substrings, from which
    J(a, a') = [root(a) == root(a')].
    """
    unique = np.unique(np.asarray(codes, dtype=np.int64))
    union = UnionFind(N_NODES)
    for code in unique:
        left = int(code) & (N_LEFT - 1)
        right = int(code) >> CUT
        union.union(left, N_LEFT + right)
    roots = np.asarray([union.find(node) for node in range(N_NODES)], dtype=int)
    left_roots = roots[:N_LEFT]
    left_counts = np.unique(left_roots, return_counts=True)[1]
    # sum over components of n_c (n_c - 1), divided by the 992 ordered pairs
    f_left = float(np.sum(left_counts * (left_counts - 1)) / OFFDIAG_COUNT)
    full_counts = np.unique(roots, return_counts=True)[1]
    row: dict[str, float | int] = {
        "components": int(len(full_counts)),
        "d_free": int(len(full_counts) - 1),
        "connected": int(len(full_counts) == 1),
        "c_left": int(len(left_counts)),
        "d_free_left": int(len(left_counts) - 1),
        "left_connected": int(len(left_counts) == 1),
        "f_left": f_left,
        "largest_left_component": int(left_counts.max()),
        "largest_component": int(full_counts.max()),
    }
    return row, left_roots


# ==========================================================================
# Pair decomposition of the reduced state (Secs. 7, 8.1)
# ==========================================================================

BOOTSTRAP_DRAWS = 20_000        # resamples of every bootstrap interval (App. B)


def pair_matrices(sign_table: np.ndarray, target_table: np.ndarray) -> tuple[np.ndarray, ...]:
    """C_h, C_h^2, the target-gauged C~_h and the Hamming fraction of one raw sign table.

    All four are 32 x 32 and taken without the crossing gauge:

        C_h(a, a')  = (1/64) sum_b h(a, b) h(a', b)
        C~_h(a, a') = the same for h t
        f(a, a')    = (1 - C_h) / 2, the fraction of right substrings on which
                      rows a and a' differ.

    C_h^2 is invariant under row flips, so its mean is 1 / K_f.
    """
    matrix = sign_table.reshape(N_RIGHT, N_LEFT).T.astype(float)
    target = target_table.reshape(N_RIGHT, N_LEFT).T.astype(float)
    with np.errstate(**_GRAM_ERRSTATE):
        correlation = matrix @ matrix.T / N_RIGHT
        correlation2 = np.square(correlation)
        gauged = (matrix * target) @ (matrix * target).T / N_RIGHT
    hamming = 0.5 * (1.0 - correlation)
    return correlation, correlation2, gauged, hamming


def _weighted(value: float, weight: float) -> float:
    """weight * value, and zero for zero weight (where value may be NaN)."""
    if weight == 0:
        return 0.0
    return weight * value


def pair_averages(sign_table: np.ndarray, target_table: np.ndarray, same: np.ndarray) -> dict[str, float]:
    """Q_in, Q_out, R_in, R_out, K_f and w_star of one raw sign table, split by ``same``.

    ``same`` is the 32 x 32 boolean J(a, a') of a constraint graph.  Q is the
    mean of C_h^2 and R the mean of C~_h over the ordered off-diagonal pairs
    inside (J = 1) and between (J = 0) components; a mean over no pair is NaN.
    K_f = 1 / mean(C_h^2) and w_star = mean(C~_h) over all entries.  The two
    reconstruction residuals measure the rounding error of the identities

        1 / K_f = 1/32 + (31/32) [f_L Q_in + (1 - f_L) Q_out]
        w_star  = 1/32 + (31/32) [f_L R_in + (1 - f_L) R_out];

    ``connected_w_star_residual`` is the second with f_L = 1, defined only for
    a connected graph.  P_within and P_between split the purity by J,
    diagonal included.
    """
    correlation, correlation2, gauged, _hamming = pair_matrices(sign_table, target_table)
    offdiag = ~np.eye(N_LEFT, dtype=bool)
    inside = same & offdiag
    outside = (~same) & offdiag
    f_left = float(inside.sum() / OFFDIAG_COUNT)
    q_in = float(correlation2[inside].mean()) if inside.any() else math.nan
    q_out = float(correlation2[outside].mean()) if outside.any() else math.nan
    r_in = float(gauged[inside].mean()) if inside.any() else math.nan
    r_out = float(gauged[outside].mean()) if outside.any() else math.nan
    purity = float(correlation2.mean())
    w_star = float(gauged.mean())
    reconstructed_purity = 1.0 / N_LEFT + (N_LEFT - 1) / N_LEFT * (
        _weighted(q_in, f_left) + _weighted(q_out, 1.0 - f_left)
    )
    reconstructed_w = 1.0 / N_LEFT + (N_LEFT - 1) / N_LEFT * (
        _weighted(r_in, f_left) + _weighted(r_out, 1.0 - f_left)
    )
    return {
        "f_left": f_left,
        "Q_in": q_in,
        "Q_out": q_out,
        "R_in": r_in,
        "R_out": r_out,
        "K_f": 1.0 / purity,
        "w_star": w_star,
        "P_within": float(correlation2[same].sum() / N_LEFT**2),
        "P_between": float(correlation2[~same].sum() / N_LEFT**2),
        "K_inverse_reconstruction_residual": abs(purity - reconstructed_purity),
        "w_star_reconstruction_residual": abs(w_star - reconstructed_w),
        "connected_w_star_residual": abs(w_star - (1.0 / N_LEFT + (N_LEFT - 1) / N_LEFT * r_in))
        if f_left == 1.0
        else math.nan,
    }


def bootstrap(values: np.ndarray, seed: int) -> tuple[float, float]:
    """2.5 and 97.5 per cent quantiles of the resampled mean of the finite entries of ``values`` (App. B).

    ``BOOTSTRAP_DRAWS`` resamples with replacement, all indices drawn at once
    from one generator seeded with ``seed``.  NaN, NaN for no finite entry.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    means = values[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


# ==========================================================================
# Tables of the retreat analyses (Sec. 8)
# ==========================================================================
#
# ``analysis.py retreat_events`` writes ``EVENTS`` and ``TRAJECTORIES``,
# ``analysis.py event_sign_tables`` one file per checkpoint under
# ``EVENT_SIGN_TABLES``.  A sign-table file holds ``signs``, the raw output sign of
# the network on all 2,048 inputs with the identically-zero output position
# dropped: column j - 1 is output position j.  The unit of the retreat
# analyses is the first event of each seed.

def read_csv(path: Path) -> list[dict[str, str]]:
    """Rows of a CSV table as dictionaries of strings."""
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def all_events() -> list[dict[str, str]]:
    """Every row of the event table, in its order (by seed, then event number)."""
    return read_csv(EVENTS)


def first_events() -> list[dict[str, str]]:
    """The event with the lowest ``event_number_within_seed`` of each seed, sorted by seed."""
    first: dict[int, dict[str, str]] = {}
    for row in all_events():
        seed = int(row["seed"])
        if seed not in first or int(row["event_number_within_seed"]) < int(
            first[seed]["event_number_within_seed"]
        ):
            first[seed] = row
    return [first[seed] for seed in sorted(first)]


def stage_epochs(row: dict[str, str]) -> dict[str, int | None]:
    """Epochs of generalization onset, last-high, retreat and recovery of one event; None where empty."""
    def maybe(key: str) -> int | None:
        value = row.get(key, "")
        return int(float(value)) if value else None

    return {
        "onset": maybe("acquisition_epoch"),
        "last_high": maybe("last_high_epoch"),
        "retreat": maybe("retreat_epoch"),
        "recovery": maybe("recovery_epoch"),
    }


def sign_table_path(seed: int, epoch: int) -> Path:
    """File of the stored sign table of one N = 1,280 checkpoint."""
    return EVENT_SIGN_TABLES / f"seed{seed:02d}_ep{epoch:03d}.npz"


def load_sign_table(seed: int, epoch: int, position_index: int) -> np.ndarray:
    """Raw output signs, int8 (2048,), of one checkpoint at output position ``LONG_RANGE_POSITIONS[position_index]``."""
    path = sign_table_path(seed, epoch)
    if not path.exists():
        raise FileNotFoundError(path)
    signs = np.load(path, allow_pickle=False)["signs"]
    return signs[:, LONG_RANGE_POSITIONS[position_index] - 1].astype(np.int8)
