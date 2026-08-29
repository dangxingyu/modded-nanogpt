# Exact PSGD PR #316 baseline reproduction

All five runs use the unmodified merged PR #316 recipe for 3,400 steps.
The gate passes when the H20 mean is within 0.003 loss of the published H100 mean.

| Seed | H20 loss | Historical H100 loss |
|---:|---:|---:|
| 1 | 3.27823 | 3.27640 |
| 2 | 3.27660 | 3.27770 |
| 3 | 3.27480 | 3.27740 |
| 4 | 3.27426 | 3.27500 |
| 5 | 3.27651 | 3.27680 |

H20 mean: 3.276080

Historical H100 mean: 3.276660

Mean difference: -0.000580

Gate: PASS
