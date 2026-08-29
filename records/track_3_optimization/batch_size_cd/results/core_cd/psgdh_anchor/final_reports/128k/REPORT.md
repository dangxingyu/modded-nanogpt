# Track-3 PSGD-H coordinate descent

Each round changes one optimizer coordinate at a time around the same
center recipe. All 32 cases in a round use one seed, the same token
horizon, and H20 GPUs. Independent coordinate winners are combined
only when their validation-loss improvement is at least 0.003.

| Batch | Rounds | Final center loss | Converged |
|---|---:|---:|---|
| 128k | 3 | 3.263530 | yes |

## Final recipes

Learning-rate entries multiply the corresponding PR #316 reference
learning rate. A first- or second-moment entry multiplies one minus
the corresponding reference beta before the beta is reconstructed.
Cooldown entries are absolute fractions of the training horizon.

| Batch | Matrix LR factor | Preconditioner LR factor | Matrix first-moment factor | Auxiliary LR factor | Auxiliary first-moment factor | Auxiliary second-moment factor | Auxiliary cooldown | Matrix cooldown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 128k | 0.5 | 1 | 2 | 2 | 2 | 0.5 | 0.5 | 1 |

The complete per-case observations are in `raw_results.csv`. The
round-level decisions are in `round_summary.csv`, every coordinate
decision is in `coordinate_improvements.csv`, and the exact final
environment multipliers are in `final_recipes.json`.
