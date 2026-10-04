# Percolation of Data Constraints and Reorganization of Boolean Function States in Parity Learning: source code

The training protocols, the analyses and the figures of the paper.

The repository holds code only. The analyses read trained networks and training
records from `data/`, which is distributed as `data.tar.gz` (517 MB), attached to
the release [`data`](https://github.com/HanMaPhy/Learning_differentiation_and_recoherence/releases/tag/data) of this repository. Unpack it at the root of the
repository:

```sh
shasum -a 256 data.tar.gz     # prints 648a32702a4e9bb46276c36765240afb62634a33c0cfbb47c3b2c7fcf2286ace
tar -xzf data.tar.gz          # creates data/
```

Python 3.9 or later, with `requirements.txt` installed.

## Files

```
common.py           the task, the model, the observables and the paths shared by the files below
analysis.py         the analyses: data/ -> results/
figures.py          the eleven figures: results/ -> figure/
train.py            the training protocols: -> data/
function_state.py   the training procedure and the observables in one self-contained file
```

Each file opens with a docstring that states the physics of what it computes and maps
paper locations to functions; `common.py` defines the observables (K_f, Γ₁, w⋆, the
constraint graph, the conditional functions) once for all of them. `data/`, `results/`
and `figure/` are not tracked. Every file finds the repository root from its own
location, so it can be started from any directory.

To see what is measured, start with `function_state.py`. It depends on nothing else
here and reads no data:

```sh
python3 function_state.py --check   # self-checks, a few seconds
python3 function_state.py           # self-checks, then a small branching experiment
```

## Running

```sh
python3 analysis.py --list                       # every analysis: name, paper location, what it writes
python3 analysis.py --all                        # all tables in results/, from data/   (about 30 minutes)
python3 analysis.py retreat_events               # one analysis
python3 figures.py --all                         # the eleven figures, from results/    (under a minute)
python3 figures.py fig2_output_position_dependence   # one figure
```

`--all` runs every item in dependency order, each in its own process. Nothing in
`analysis.py` or `figures.py` trains. Both files set the BLAS thread variables to 1
unless they are set already.

## Figures

| Figure | File in `figure/` and function in `figures.py` | Tables it reads |
|---|---|---|
| 1 | `fig1_support_state_performance` | none (schematic) |
| 2 | `fig2_output_position_dependence` | `results/output_position_dependence/` |
| 3 | `fig3_training_trajectories` | `results/training_trajectories/`, `results/state_vs_accuracy/`, `results/retreat_events/` |
| 4 | `fig4_state_vs_accuracy` | `results/state_vs_accuracy/` |
| 5 | `fig5_starting_model_branches` | `results/largest_addition_branches/` |
| 6 | `fig6_added_data_size` | `results/branch_endpoint_state/` |
| 7 | `fig7_constraint_percolation_vs_state` | `results/constraint_percolation/`, `results/branch_endpoint_state/` |
| 8 | `fig8_fixed_identifiable_support` | `results/fixed_support_timing/` |
| 9 | `fig9_conditional_function_change` | `results/conditional_function_splitting/` |
| 10 | `fig10_retreat_recovery_fixed_graph` | `results/retreat_recurrence/`, `results/retreat_pair_correlations/` |
| 11 | `fig11_retreat_recovery_robustness` | `results/conditional_function_splitting/` |

Each figure is written to `figure/<name>.pdf`.

## Analyses

In dependency order. Each function of `analysis.py` writes the directory of
`results/` that has its name.

| Function | Needs | Computes | Paper |
|---|---|---|---|
| `retreat_events` | `data/large_data_trajectories/` | retreat events of the N=1280 trajectories: last-high, retreat, recovery and control checkpoints | Table 1, Sec. 4.3 |
| `event_sign_tables` | retreat events | sign tables over all 2,048 inputs at the checkpoints the retreat analyses use | Sec. 8 |
| `training_trajectories` | retreat events, `data/small_data_trajectories/` | K_f, Γ₁ and accuracy along the N=64 and N=1280 trajectories; state at interpolation; K_f peak and state at generalization onset | Fig. 3, Secs. 4.2, 4.3 |
| `output_position_dependence` | retreat events | K_f, w⋆ and accuracy per output position before onset; the three frontiers | Fig. 2, Sec. 4.1 |
| `state_vs_accuracy` | retreat events | function state and held-out block accuracy at every checkpoint; ranges at fixed accuracy; onset-aligned means and the K_f peak of each trajectory | Figs. 3B, 4, Secs. 4.3, 4.4 |
| `largest_addition_branches` | `data/branching_study/` | traces, means, spectra and contrasts of the three branches (distinct, duplicate, control) at the largest addition | Fig. 5, Sec. 5.1 |
| `branch_contrasts` | `data/branching_study/` | branch-point means, contrasts at update 25, accuracy on the added and on the never-trained inputs, independent orderings | Sec. 5.1, App. B.1 |
| `constraint_percolation` | nothing | constraint graph of the added data and of 10,000 random orderings: components, f_L, half points | Fig. 7, Sec. 6 |
| `branch_endpoint_state` | `data/branching_study/`, constraint percolation | pair correlations, K_f and w⋆ at update 800 per added-data size; the contrasts with bootstrap intervals and sign counts | Figs. 6, 7, Sec. 5.2 |
| `fixed_support_timing` | `data/branching_study/` | the same quantities at the saved updates of the largest addition, with intervals | Fig. 8, Sec. 7 |
| `conditional_function_splitting` | retreat events, sign tables | splitting entropy B and subspace overlaps R_2, R_4 for retreat, control and recovery; approach to retreat; threshold sweep | Figs. 9, 11, Sec. 8.2, App. B.2 |
| `retreat_pair_correlations` | retreat events, sign tables | pair correlations, K_f and w⋆ at last-high, retreat and recovery; constraint graph of the N=1280 training set | Sec. 3, Sec. 8.1 |
| `retreat_recurrence` | retreat pair correlations | function fidelity, pair-pattern correlation and return ratios between last-high and recovery; the stage means of the 20 events with a recovery | Fig. 10, Secs. 8.1, 8.3 |

## Data and the training protocols that produce it

`data.tar.gz` holds the files the analyses read:

```
data/large_data_trajectories/checkpoints/seed<k>/ep<e>.pt   the 24 trajectories at N=1280 with a retreat event, weights at every epoch from 20 to 220
data/large_data_trajectories/monitoring_seeds<a>_<b>.csv    monitoring block accuracy and loss along all 30 trajectories
data/small_data_trajectories/trajectories.json              24 trajectories at N=64 with the state at 52 updates
data/branching_study/protocol.json                          the branching protocol, its 25 starting models and their state
data/branching_study/trajectory_metrics.csv                 every branch at updates 0, 25, 50, 100, 200, 400, 800
data/branching_study/endpoint_sign_tables.npz               sign tables at update 800 of the distinct and duplicate branches, all six additions
data/branching_study/checkpoint_sign_tables.npz             sign tables of the two branches with 1,216 added inputs at the 11 saved updates
data/branching_study/checkpoints_added1216.csv              the same two branches: accuracy on input subsets at the 11 saved updates
data/branching_study/independent_orderings.csv              8 starting models under three orderings of the added inputs, at the branch point and at update 800
```

`python3 train.py --list` shows every protocol with what it writes, and
`python3 train.py PROTOCOL --help` its options. The defaults are the protocol settings
of the data.

| Data | Command |
|---|---|
| `data/small_data_trajectories/trajectories.json` | `python3 train.py small_data_trajectories` |
| `data/large_data_trajectories/`: seeds 0 to 5 and `monitoring_seeds00_05.csv` | `python3 train.py large_data_trajectories` |
| seeds 6 to 8 | `python3 train.py large_data_trajectories --seeds 6,7,8` |
| seeds 9 to 15 and `monitoring_seeds09_15.csv` | `python3 train.py large_data_trajectories --seeds 9,10,11,12,13,14,15` |
| seeds 16 to 22 and `monitoring_seeds16_22.csv` | `python3 train.py large_data_trajectories --seeds 16,17,18,19,20,21,22` |
| seeds 23 to 29 and `monitoring_seeds23_29.csv` | `python3 train.py large_data_trajectories --seeds 23,24,25,26,27,28,29` |
| `monitoring_seeds06_08.csv` | `python3 train.py monitoring_from_checkpoints` |
| the starting models of the branching study, `data/small_data_models/` | `python3 train.py small_data_ensemble` (runs `python3 train.py small_data_model --seed k` once per network) |
| `data/branching_study/protocol.json`, `trajectory_metrics.csv` and the records under `records/` | `python3 train.py branching_study` |
| `data/branching_study/endpoint_sign_tables.npz` | `python3 train.py branch_endpoint_sign_tables` |
| `data/branching_study/checkpoint_sign_tables.npz` | `python3 train.py branch_checkpoint_sign_tables` |
| `data/branching_study/checkpoints_added1216.csv` | `python3 train.py branch_subset_accuracy` |
| `data/branching_study/independent_orderings.csv` | `python3 train.py branch_independent_orderings` |

The monitoring values of seeds 6 to 8 are evaluated from their stored weights, from
epoch 20 on, by `monitoring_from_checkpoints`. Every protocol skips an output that is
present and trains it again only with `--overwrite`. The package leaves out the outputs
no analysis reads: the weights of the six trajectories without an event, the starting
models and the records under `data/branching_study/records/`. Train into a checkout
without the unpacked package; over it, a protocol would train the left-out outputs
again and write over files the analyses read.

## What reproduces exactly

Every analysis and every figure. Run on the same data, `analysis.py --all` and
`figures.py --all` give the same tables byte for byte and the same figures.

The training values of an individual network depend on the machine and the torch
thread count. Every protocol fixes its thread count (`--threads` or
`--threads-per-worker`); with the same thread count on the same machine it gives the
same records bit for bit. `branch_endpoint_sign_tables` checks every branch it trains
against the record of `branching_study`.

## The sign convention

The sign tables in `results/event_sign_tables/` and under `data/branching_study/`
store the raw signs `sign(logit_j(x))`. In the event tables the always-zero output
column is dropped, so column `j-1` holds output position `j`. Each analysis applies
the target gauge itself and forms the target-gauged table `sign(logit) * t` before
taking a Gram matrix.

The parity target factorizes across the cut, `t(a, b) = u(a) v(b)`, so the target
gauge `h -> h t` acts on the conditional correlation as `C -> u(a) u(a') C`. `K_f`, `Γ₁`
and `C_h²` are exactly invariant under it. Only sign-sensitive quantities move: `w⋆`,
`R_in` and the pairwise Hamming distribution.
