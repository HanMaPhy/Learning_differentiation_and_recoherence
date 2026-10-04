"""Training and observables of the function state in parity learning.

One self-contained file for the paper "Percolation of Data Constraints and
Reorganization of Boolean Function States in Parity Learning".
`python function_state.py` runs the self-checks and then the branching of one
starting model, in a few minutes, with no data files.

    python function_state.py                # self-checks, then demo at +64
    python function_state.py --added 1216   # the largest addition in the paper
    python function_state.py --check        # self-checks only

===========================================================================
THE TASK
===========================================================================
Cumulative parity on a finite input space.  An input is D = L-1 bits; the
target at output position j is the parity of the first j bits.  For L = 12 the
whole function lives on 2^11 = 2048 inputs, so at any point in training the
model can be evaluated on *every* input there is.  The object of study is that
complete input-output table -- the function the model represents at that
point -- never a subset of it.

===========================================================================
THE FUNCTION STATE
===========================================================================
Take one output position and write the model's answer as a sign,
M(s) = +1 if logit >= 0 else -1.  Cut the D input bits into a left block of
d_L = 5 and a right block of D - d_L = 6.  Because inputs are enumerated by the
integer their bits spell out, that cut is a reshape: M becomes a matrix
M(a, b) with n_L = 32 rows and n_R = 64 columns.  Then

    C_h(a,a') = (1/n_R) sum_b M(a,b) M(a',b),      rho_A = C_h / n_L

says how the function's answer to the right half depends on the left half.
rho_A is rank one if the left half is irrelevant and spreads over many modes if
the response is strongly left-dependent.

TWO GAUGES

  crossing gauge   M -> M(a,b) M(a,1) M(1,b) M(1,1)
                   removes whatever factorizes across the cut.

  target gauge     M -> M(a,b) T(a,b)
                   a model that has learned the function becomes all +1.

The target factorizes, T(a,b) = u(a) v(b), so the target gauge acts on the
correlation as C_h -> u(a)u(a') C_h.  The crossing gauge of a factorized matrix
is identically all +1, so K_f, Gamma_1 and the Schmidt spectrum are *exactly
blind* to the target gauge: they cannot tell a target-aligned function from one
that is merely simple.  Only w_star can, and only when taken from the
target-gauged matrix.

T*T = 1, so gauging twice returns the raw matrix and collapses w_star to the
ungauged Gram mean.  Sign patterns are cached RAW and gauged at the point of use.

===========================================================================
THE OBSERVABLES
===========================================================================
State, averaged over the long-range positions j = 8, 9, 10, 11:

  K_f       participation ratio of the spectrum of the crossing-gauged rho_A.
            1 when one conditional mode carries all the weight, rising to n_L
            as it spreads.  Target-gauge invariant.
  Gamma_1   <u_uni|rho_A^x|u_uni>, weight on the crossing gauge's uniform mode.
            Also target-gauge invariant.
  w_star    <u_star|rho_A|u_star>, weight on the TARGET's conditional mode.
            In the target-gauged frame u_star is the uniform vector, so this is
            the mean of the target-gauged Gram.  NOT invariant.
  lambda_a  the Schmidt spectrum itself.
  A_blk     fraction of inputs correct at every output position (also the
            long-range-only and per-bit variants).

Data, from the training set alone, before any training.  Each training input
contributes one edge between its left and right block, giving a bipartite graph
on n_L + n_R = 96 nodes.  Two inputs sharing a right block fix the relative
sign u(a)u(a'); chaining such steps, the relative target sign of two left
blocks is determined exactly when they share a component.

  f_L       fraction of ordered off-diagonal left pairs inside one component:
            the identifiable fraction.  Cannot decrease as data is added.
  d_free    components - 1, the sign choices the data leaves open.
  K_perc    [1/n_L + (n_L-1)/n_L * f_L]^-1, the effective rank of the state
            that occupies every determinable relation and nothing else -- the
            ceiling the graph alone implies.  Falls monotonically to 1.

The bridge.  Splitting the off-diagonal correlations by whether the data
relates the pair gives R_in, R_out (target-gauged) and Q_in, Q_out (squared
raw), and

    w_star = 1/n_L + (n_L-1)/n_L [ f_L R_in + (1 - f_L) R_out ]

is an identity: the diagonal contributes 1/n_L and the off-diagonal splits in
two.  `self_check` verifies it to machine precision.

===========================================================================
REPRODUCIBILITY
===========================================================================
Nothing here is unseeded; dropout is zero everywhere.  Fixing the CPU thread
count makes a run bit-for-bit reproducible.  The thread count sets the order in
which the LayerNorm backward pass accumulates parameter gradients, and training
amplifies the last-bit difference between two thread counts into an O(1)
difference in the per-model observables by update 800.

Requires torch and numpy.  Tested with torch 2.8.0, numpy 2.0.2.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

L = 12
D = L - 1                               # input bits
CUT = 5                                 # d_L
N_LEFT, N_RIGHT = 2**CUT, 2 ** (D - CUT)          # 32, 64
N_NODES = N_LEFT + N_RIGHT                        # 96 bipartite graph nodes
OFFDIAG_PAIRS = N_LEFT * (N_LEFT - 1)             # 992
LONG_RANGE = (8, 9, 10, 11)

TRAIN_SEED = 12345                      # the shared fixed training set
INTERVENTION_DATA_SEED = 20260822       # which inputs get added
MINIBATCH_SEED = 20260823               # continuation batch draws
MONITORING_SEED = 777                   # accuracy readout only
CONFIG = dict(channels=16, depth=8, heads=4, ffn=64, lr=2e-3, wd=1e-4)
BATCH, STARTING_MODEL_STEPS, BRANCH_STEPS = 128, 4000, 800


# ==========================================================================
# The task
# ==========================================================================

def _targets(bits: torch.Tensor) -> torch.Tensor:
    """Position 0 has no preceding bits; position j is the parity of bits 1..j."""
    z = torch.zeros(bits.shape[0], L)
    z[:, 1:] = torch.cumsum(bits, 1).remainder(2.0)
    return z


def _encode(bits: torch.Tensor) -> torch.Tensor:
    """Two channels: the bit entering at each position, and the same shifted.

    Every position gets a local view of the chain; nothing hands the model the
    running sum, which it has to build itself.
    """
    x = torch.zeros(bits.shape[0], 2, L)
    x[:, 0, 1:] = bits
    x[:, 1, :-1] = bits
    return x


def training_set(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """n inputs drawn uniformly, and their targets."""
    g = torch.Generator().manual_seed(seed)
    return _encode(b := torch.randint(0, 2, (n, D), generator=g).float()), _targets(b)


def full_binary_space() -> tuple[torch.Tensor, torch.Tensor]:
    """Every input in the space, indexed so that index == the bits it spells.

    That indexing is what makes the cut in `response_matrix` a plain reshape.
    """
    codes = torch.arange(2**D, dtype=torch.long)
    bits = ((codes[:, None] >> torch.arange(D)[None, :]) & 1).float()
    return _encode(bits), _targets(bits)


def training_codes(n: int = 64, seed: int = TRAIN_SEED) -> torch.Tensor:
    """The same training inputs as `training_set`, as input codes.

    Consumes the generator identically, so the two always agree.  The N = 64
    set is drawn once from seed 12345 and then quenched: every model in the
    ensemble trains on it, so models differ only in init and batch order.
    """
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (n, D), generator=g) @ (2 ** torch.arange(D, dtype=torch.long))


def branch_training_sets(base: torch.Tensor, count: int) -> dict[str, torch.Tensor]:
    """The three branches at one added-data size.

    A branch is specified entirely by which input codes it trains on, so the
    branches differ in exactly one respect: whether the added codes are absent
    from the base set or are copies of indices already present.  Starting
    weights, optimizer reset, batch draws and the number of added points are
    identical.

    The unseen pool is shuffled once and then sliced, so the distinct sets of
    different sizes are nested.  The duplicates are whole random permutations
    laid end to end, so copies spread evenly over the base set.
    """
    unseen = np.setdiff1d(np.arange(2**D, dtype=np.int64), torch.unique(base).numpy(),
                          assume_unique=True)
    np.random.default_rng(INTERVENTION_DATA_SEED).shuffle(unseen)
    distinct = torch.from_numpy(unseen[:count].copy()).long()

    rng, idx = np.random.default_rng(INTERVENTION_DATA_SEED + 37), []
    while len(idx) < count:
        idx.extend(rng.permutation(len(base)).tolist())
    duplicate = base[torch.as_tensor(idx[:count], dtype=torch.long)]

    return {"control": base,
            "duplicate": torch.cat((base, duplicate)),
            "distinct": torch.cat((base, distinct))}


# ==========================================================================
# The model -- the architecture and parameter names of the trained networks
# ==========================================================================

class TransformerBlock(nn.Module):
    """Pre-norm encoder block: LN -> attention -> residual, LN -> FFN -> residual."""

    def __init__(self, channels: int, num_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.norm_attn = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout,
                                          batch_first=True)
        self.norm_ffn = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(nn.Linear(channels, ffn_dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(ffn_dim, channels))
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        n = self.norm_attn(h)
        h = h + self.dropout(self.attn(n, n, n, need_weights=False)[0])
        return h + self.ffn(self.norm_ffn(h))


class ChainTransformer(nn.Module):
    """Encoder-only Transformer over chain positions.  x: [B, 2, L] -> [B, L] logits.

    Attention is bidirectional (no causal mask).
    """

    def __init__(self, L: int, channels: int, depth: int, num_heads: int,
                 ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        if channels % num_heads:
            raise ValueError(f"channels {channels} not divisible by heads {num_heads}")
        self.in_proj = nn.Linear(2, channels)
        self.pos_embed = nn.Parameter(torch.zeros(1, L, channels))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(channels, num_heads, ffn_dim, dropout) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(channels)
        self.out_proj = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x.transpose(1, 2)) + self.pos_embed
        for block in self.blocks:
            h = block(h)
        return self.out_proj(self.norm_out(h)).squeeze(-1)


# ==========================================================================
# State observables
# ==========================================================================

def response_matrix(sign_table) -> np.ndarray:
    """One position's sign vector as the [n_L, n_R] response matrix."""
    return np.asarray(sign_table, dtype=np.float64).reshape(N_RIGHT, N_LEFT).T


def crossing_gauge(m: np.ndarray) -> np.ndarray:
    """Flip rows and columns so the first row and column are all +1.

    Kills anything that factorizes across the cut; for the target itself the
    result is the all-ones matrix.
    """
    return m * m[:, :1] * m[:1, :] * m[0, 0]


def gram(m: np.ndarray) -> np.ndarray:
    """C_h = (1/n_R) M M^T, unit diagonal, with 1/sqrt(n_R) folded into the rows."""
    rows = m / math.sqrt(float(N_RIGHT))
    # NumPy 2.0 on macOS Accelerate raises divide/overflow/invalid flags from a
    # float64 matmul of finite operands.  The flags are silenced and the result
    # is checked for finiteness instead.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        product = rows @ rows.T
    if not np.isfinite(product).all():
        raise FloatingPointError("gram: non-finite entry in M M^T")
    return product


def schmidt_spectrum(m: np.ndarray) -> np.ndarray:
    """Normalized eigenvalues of rho_A, ASCENDING -- LAPACK's own order.

    Clipped at zero first: rho_A is positive semidefinite, so anything negative
    is round-off.  The order matters because the purity sum below is
    order-sensitive in its last bits; reverse for display, never before summing.
    """
    s = np.clip(np.linalg.eigvalsh(gram(m) / float(N_LEFT)), 0.0, None)
    return s / s.sum()


def uniform_mode_weight(m: np.ndarray) -> float:
    """mean(C_h) = <u_uni|rho_A|u_uni>.  WHICH quantity depends on the gauge:

        crossing-gauged -> Gamma_1 (target-gauge invariant)
        target-gauged   -> w_star  (the target-mode population)

    The raw matrix gives neither.  Always gauge first.
    """
    return float(gram(m).mean())


def position_observables(sign_table, target_signs) -> dict:
    """All state observables at one output position.

    Gauging happens here so a caller cannot read w_star off the wrong frame.
    """
    crossed = crossing_gauge(response_matrix(sign_table))
    gauged = response_matrix(np.asarray(sign_table) * np.asarray(target_signs))
    spectrum = schmidt_spectrum(crossed)                       # ascending
    return {"K_f": 1.0 / float(np.square(spectrum).sum()),
            "Gamma_1": uniform_mode_weight(crossed),
            "w_star": uniform_mode_weight(gauged),
            "spectrum": spectrum[::-1]}                        # descending to read


@torch.no_grad()
def measure(model, inputs: torch.Tensor, targets: torch.Tensor) -> dict:
    """Evaluate on the complete input space and return the observables.

    eval mode: these are properties of the function the model represents, not
    of its training-time behaviour.
    """
    model.eval()
    predicted = model(inputs) >= 0.0
    target = targets.bool()
    signs = np.where(predicted.numpy(), 1, -1).astype(np.int8)
    target_signs = np.where(target.numpy(), 1, -1).astype(np.int8)
    per = [position_observables(signs[:, j], target_signs[:, j]) for j in LONG_RANGE]
    return {"K_f": float(np.mean([p["K_f"] for p in per])),
            "Gamma_1": float(np.mean([p["Gamma_1"] for p in per])),
            "w_star": float(np.mean([p["w_star"] for p in per])),
            "A_block": float((predicted == target).all(dim=1).float().mean()),
            "A_block_tail": float((predicted[:, LONG_RANGE] == target[:, LONG_RANGE]).all(dim=1)
                                  .float().mean()),
            "A_bit": float((predicted == target).float().mean()),
            "spectrum": np.mean([p["spectrum"] for p in per], axis=0),
            "signs": signs}


# ==========================================================================
# The constraint graph, and what it determines
# ==========================================================================

class UnionFind:
    """Disjoint sets, path halving and union by size."""

    def __init__(self, n: int):
        self.parent, self.size = list(range(n)), [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b], self.size[a] = a, self.size[a] + self.size[b]


def build_graph(codes) -> dict:
    """Connectivity of the training set given by input codes.

    One edge per DISTINCT input, between its left and right block.  Repeated
    inputs add no edge, so the duplicate branch has the graph of the base set.
    """
    uf = UnionFind(N_NODES)
    for code in np.unique(np.asarray(codes, dtype=np.int64)):
        uf.union(int(code) & (N_LEFT - 1), N_LEFT + (int(code) >> CUT))
    roots = np.asarray([uf.find(n) for n in range(N_NODES)], dtype=int)
    left_labels = roots[:N_LEFT]
    left_sizes = np.unique(left_labels, return_counts=True)[1]
    all_sizes = np.unique(roots, return_counts=True)[1]
    f_L = float(np.sum(left_sizes * (left_sizes - 1)) / OFFDIAG_PAIRS)
    return {"f_L": f_L,
            "d_free": int(len(all_sizes) - 1),
            "components": int(len(all_sizes)),
            "K_perc": 1.0 / (1.0 / N_LEFT + (N_LEFT - 1) / N_LEFT * f_L),
            "left_labels": left_labels}


def block_decomposition(sign_table, target_signs, same: np.ndarray) -> dict:
    """Split the off-diagonal correlations by whether the data relates the pair.

    Inside a component the relative target sign is determined; outside it is
    not.  Q feeds K_f and R feeds w_star through the exact identities returned
    here: the diagonal contributes 1/n_L and the off-diagonal pairs a weighted
    average.
    """
    corr = gram(response_matrix(sign_table))
    gauged = gram(response_matrix(sign_table) * response_matrix(target_signs))
    offdiag = ~np.eye(N_LEFT, dtype=bool)
    inside, outside = same & offdiag, (~same) & offdiag
    f_L = float(inside.sum() / OFFDIAG_PAIRS)
    q_in = float(np.square(corr[inside]).mean()) if inside.any() else 0.0
    q_out = float(np.square(corr[outside]).mean()) if outside.any() else 0.0
    r_in = float(gauged[inside].mean()) if inside.any() else 0.0
    r_out = float(gauged[outside].mean()) if outside.any() else 0.0
    blend = lambda a, b: (f_L * a if inside.any() else 0.0) + \
                         ((1.0 - f_L) * b if outside.any() else 0.0)
    edge = (N_LEFT - 1) / N_LEFT
    return {"f_L": f_L, "Q_in": q_in, "Q_out": q_out, "R_in": r_in, "R_out": r_out,
            "K_f_reconstructed": 1.0 / (1.0 / N_LEFT + edge * blend(q_in, q_out)),
            "w_star_reconstructed": 1.0 / N_LEFT + edge * blend(r_in, r_out)}


# ==========================================================================
# Training
# ==========================================================================

def build_model() -> ChainTransformer:
    return ChainTransformer(L=L, channels=CONFIG["channels"], depth=CONFIG["depth"],
                            num_heads=CONFIG["heads"], ffn_dim=CONFIG["ffn"],
                            dropout=0.0)


def train_starting_model(seed: int, N: int = 64, verbose: bool = True) -> ChainTransformer:
    """One starting model on the shared fixed training set.

    `seed` selects initialization and batch order only.  The step budget is held
    near 4,000 whatever N is, so training sets of different sizes get comparable
    optimization pressure.
    """
    x_tr, z_tr = training_set(N, TRAIN_SEED)
    x_te, z_te = training_set(4000, MONITORING_SEED)     # readout only, never trained on
    torch.manual_seed(seed)                # after the data, so only weights depend on it
    model = build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"],
                            weight_decay=CONFIG["wd"])
    per_epoch = math.ceil(N / BATCH)
    epochs = math.ceil(STARTING_MODEL_STEPS / per_epoch)
    shuffle = torch.Generator().manual_seed(seed + 1)
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(N, generator=shuffle)
        for k in range(per_epoch):
            batch = order[k * BATCH:(k + 1) * BATCH]
            opt.zero_grad()
            F.binary_cross_entropy_with_logits(model(x_tr[batch]), z_tr[batch]).backward()
            opt.step()
        if verbose and (epoch % max(1, epochs // 4) == 0 or epoch == epochs):
            model.eval()
            with torch.no_grad():
                tr = ((model(x_tr) > 0).float() == z_tr).float().mean().item()
                te = ((model(x_te) > 0).float() == z_te).float().mean().item()
            print(f"   epoch {epoch}/{epochs}  train {tr:.3f}  held-out {te:.3f}",
                  flush=True)
    return model


def continue_branch(state: dict, codes: torch.Tensor, seed: int,
                    checkpoints: tuple[int, ...], steps: int = BRANCH_STEPS) -> list[dict]:
    """Continue a trained model on the training set given by `codes`.

    Every branch receives the same `state` and a freshly constructed optimizer, so
    they start from the same model AND the same empty optimizer state.  `seed`
    is the branch point's model seed, so the branches see the same batch
    draw sequence; only the set those draws index into differs.
    """
    inputs, targets = full_binary_space()
    model = build_model()
    model.load_state_dict(state)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"],
                            weight_decay=CONFIG["wd"])
    x_tr, z_tr = inputs[codes], targets[codes]
    # Drawn up front from one seeded generator, so the sequence depends on
    # nothing the model does.
    draws = torch.rand(steps, BATCH,
                       generator=torch.Generator().manual_seed(MINIBATCH_SEED + 7919 * seed))
    rows = []
    for step in range(steps + 1):
        if step in set(checkpoints):
            m = measure(model, inputs, targets)
            del m["signs"], m["spectrum"]
            rows.append({"step": step, **m})
        if step == steps:
            break
        model.train()
        # Sampling with replacement: the batch distribution is the same
        # whatever the training set size.
        idx = torch.clamp((draws[step] * len(codes)).long(), max=len(codes) - 1)
        opt.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(model(x_tr[idx]), z_tr[idx]).backward()
        opt.step()
    return rows


# ==========================================================================
# Self-checks -- regression tests on the gauge algebra and the graph
# ==========================================================================

def _target_signs(position: int) -> np.ndarray:
    return np.where(full_binary_space()[1].bool().numpy()[:, position], 1, -1).astype(np.int8)


def _flip(sign_table: np.ndarray, count: int, seed: int) -> np.ndarray:
    """Flip `count` signs at random: a deliberately imperfect model."""
    out = sign_table.copy()
    out[np.random.default_rng(seed).choice(len(out), count, replace=False)] *= -1
    return out


def self_check() -> None:
    inputs, targets = full_binary_space()
    target = _target_signs(9)

    # The reshape really is the left/right split: mark each input by its left
    # block and the rows of the response matrix must come out constant.
    left_block = (torch.arange(2**D).numpy() & (N_LEFT - 1)).astype(np.float64)
    m = response_matrix(left_block)
    assert m.shape == (N_LEFT, N_RIGHT) and (m == m[:, :1]).all()
    print(f"  cut geometry          rows are left blocks, shape {m.shape}")

    perfect = position_observables(target, target)
    assert abs(perfect["K_f"] - 1) < 1e-12 and abs(perfect["w_star"] - 1) < 1e-12
    print("  perfect model         K_f = 1, w_star = 1")

    s = schmidt_spectrum(crossing_gauge(response_matrix(_flip(target, 500, 3))))
    assert abs(s.sum() - 1) < 1e-12 and (s >= 0).all()
    print(f"  spectrum              sums to 1, K_f in [1, {N_LEFT}]")

    # Target gauge: K_f and Gamma_1 must NOT move, w_star must.
    sign_table = _flip(target, 300, 1)
    plain, twice = (position_observables(sign_table, target),
                    position_observables(sign_table * target, target))
    assert plain["K_f"] == twice["K_f"] and plain["Gamma_1"] == twice["Gamma_1"]
    assert abs(plain["w_star"] - twice["w_star"]) > 1e-3
    print(f"  target gauge          K_f, Gamma_1 invariant; "
          f"w_star {plain['w_star']:.4f} -> {twice['w_star']:.4f} under double gauge")

    print("  corrupting the state  w_star falls, K_f rises:")
    last_w, last_k = 1.01, 0.99
    for count in (0, 64, 256, 512, 1024):
        r = position_observables(_flip(target, count, 2), target)
        assert r["w_star"] < last_w and r["K_f"] > last_k
        last_w, last_k = r["w_star"], r["K_f"]
        print(f"      {count:5d} flipped       K_f {r['K_f']:7.4f}   w_star {r['w_star']:7.4f}")

    print("  constraint graph      f_L rises, K_perc falls, duplicates inert:")
    base = training_codes()
    last_f, last_k = -1.0, N_LEFT + 1.0
    for count in (0, 16, 32, 64, 192):
        g = build_graph(branch_training_sets(base, count)["distinct"].numpy())
        assert g["f_L"] > last_f and g["K_perc"] < last_k
        last_f, last_k = g["f_L"], g["K_perc"]
        print(f"      +{count:<5d} distinct      f_L {g['f_L']:7.4f}   "
              f"K_perc {g['K_perc']:7.4f}   d_free {g['d_free']:3d}")
    plain_g = build_graph(base.numpy())
    dup_g = build_graph(branch_training_sets(base, 1216)["duplicate"].numpy())
    assert dup_g["f_L"] == plain_g["f_L"] and dup_g["d_free"] == plain_g["d_free"]

    # The bridge: an identity, so any gauge error breaks it.
    g = build_graph(base.numpy())
    same = g["left_labels"][:, None] == g["left_labels"][None, :]
    direct = position_observables(_flip(target, 400, 5), target)["w_star"]
    recon = block_decomposition(_flip(target, 400, 5), target, same)["w_star_reconstructed"]
    assert abs(direct - recon) < 1e-12
    print(f"  w_star factorization  direct {direct:.10f} == decomposed {recon:.10f}")
    print("  all checks passed")


# ==========================================================================
# Demo
# ==========================================================================

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seed", type=int, default=0, help="init and batch order")
    p.add_argument("--added", type=int, default=64, help="inputs added per branch")
    p.add_argument("--steps", type=int, default=BRANCH_STEPS)
    p.add_argument("--threads", type=int, default=2,
                   help="fixing this makes the run deterministic")
    p.add_argument("--check", action="store_true", help="self-checks only")
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    print("self-checks")
    self_check()
    if args.check:
        return 0

    inputs, targets = full_binary_space()
    print(f"\n1. one starting model (seed {args.seed}, N = 64)")
    t0 = time.time()
    model = train_starting_model(args.seed)
    print(f"   {STARTING_MODEL_STEPS} updates in {time.time() - t0:.0f}s")

    start = measure(model, inputs, targets)
    print("\n2. the function it represents, on all 2^11 inputs")
    print(f"   K_f {start['K_f']:7.4f}   Gamma_1 {start['Gamma_1']:7.4f}   "
          f"w_star {start['w_star']:7.4f}   A_blk {start['A_block']:7.4f}")
    print("   it fits its 64 training inputs perfectly while representing a")
    print("   function far from the true one: K_f large, w_star near zero")

    print(f"\n3. three branches, +{args.added} added inputs, {args.steps} updates")
    print("   f_L, K_perc, d_free come from the training set alone, before any")
    print("   training.  Duplicates add no edge, so they cannot move them.\n")
    state = {k: v.clone() for k, v in model.state_dict().items()}
    print(f"   {'branch':10s} {'N':>6} {'f_L':>8} {'K_perc':>8} {'d_free':>7} | "
          f"{'K_f':>8} {'w_star':>8} {'A_blk':>8}")
    end = {}
    for name, codes in branch_training_sets(training_codes(), args.added).items():
        g = build_graph(codes.numpy())
        end[name] = continue_branch(state, codes, args.seed, (0, args.steps),
                                    steps=args.steps)[-1]
        print(f"   {name:10s} {len(codes):6d} {g['f_L']:8.4f} {g['K_perc']:8.4f} "
              f"{g['d_free']:7d} | {end[name]['K_f']:8.4f} "
              f"{end[name]['w_star']:8.4f} {end[name]['A_block']:8.4f}")

    dk = end["distinct"]["K_f"] - end["duplicate"]["K_f"]
    dw = end["distinct"]["w_star"] - end["duplicate"]["w_star"]
    print("\n4. the contrast that isolates the added information")
    print(f"   Delta K_f    {dk:+8.4f}   "
          f"the state {'differentiated' if dk > 0 else 'recohered'}")
    print(f"   Delta w_star {dw:+8.4f}   "
          f"{'towards' if dw > 0 else 'away from'} the target mode")
    print(f"   Delta A_blk  {end['distinct']['A_block'] - end['duplicate']['A_block']:+8.4f}")
    print("\n   One model only.  The paper's numbers are means over 25, and")
    print("   the sign of Delta K_f flips between small and large additions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
