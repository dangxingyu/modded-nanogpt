# Track-3 PSGD-H coordinate descent

Each round changes one optimizer coordinate at a time around the same
center recipe. All 32 cases in a round use one seed, the same token
horizon, and H20 GPUs. A coordinate is considered positive only when
its validation-loss improvement is at least 0.003. The center used by
the next round is an actually observed recipe, so an unevaluated
combination of coordinate winners is not treated as evidence.

| Batch | Rounds | Final center loss | Converged |
|---|---:|---:|---|
| 512k | 9 | 3.273700 | yes |

## Final recipes

Learning-rate entries multiply the corresponding PR #316 reference
learning rate. A first- or second-moment entry multiplies one minus
the corresponding reference beta before the beta is reconstructed.
Cooldown entries are absolute fractions of the training horizon.

| Batch | Matrix LR factor | Preconditioner LR factor | Matrix first-moment factor | Auxiliary LR factor | Auxiliary first-moment factor | Auxiliary second-moment factor | Auxiliary cooldown | Matrix cooldown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 512k | 1 | 1 | 2 | 1.41421356237 | 1 | 0.5 | 0.5 | 1 |

The complete per-case observations are in `raw_results.csv`. The
round-level decisions are in `round_summary.csv`, every coordinate
decision is in `coordinate_improvements.csv`, and the exact final
environment multipliers are in `final_recipes.json`.
