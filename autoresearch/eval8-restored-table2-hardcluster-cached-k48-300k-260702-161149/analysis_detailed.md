# Hardcluster Cached K48 Analysis

## Summary
| Method | Done | Macro ΔTC | Min ΔTC | Pos/Neg | Plan sec | Eval sec |
|---|---:|---:|---:|---:|---:|---:|
| A_heuristic_only | 8/8 | 3.013% | -1.960% | 7/1 | 1319.0 | 440.0 |
| B_world_rerank | 8/8 | 3.378% | 0.467% | 8/0 | 8562.9 | 432.2 |
| C_depth2_rollout | 8/8 | 3.419% | 0.222% | 8/0 | 12682.2 | 452.9 |

## Per-circuit ΔTC and Hardcone96 Gap
| Circuit | Method | ΔTC | Hardcone96 ΔTC | Gap | Plan sec |
|---|---|---:|---:|---:|---:|
| b15_C | A_heuristic_only | 9.782% | 0.794% | +8.988% | 27.0 |
| b15_C | B_world_rerank | 8.290% | 9.859% | -1.569% | 225.8 |
| b15_C | C_depth2_rollout | 9.538% | 9.941% | -0.403% | 404.6 |
| b20_C | A_heuristic_only | 2.274% | -0.768% | +3.042% | 50.0 |
| b20_C | B_world_rerank | 2.339% | 2.289% | +0.050% | 858.5 |
| b20_C | C_depth2_rollout | 2.411% | 2.133% | +0.278% | 1242.7 |
| b21_C | A_heuristic_only | 2.104% | -1.271% | +3.375% | 51.0 |
| b21_C | B_world_rerank | 2.051% | 2.114% | -0.063% | 826.9 |
| b21_C | C_depth2_rollout | 2.045% | 4.181% | -2.136% | 1006.5 |
| b22_C | A_heuristic_only | 0.393% | -1.463% | +1.856% | 116.0 |
| b22_C | B_world_rerank | 0.467% | 2.372% | -1.905% | 1480.3 |
| b22_C | C_depth2_rollout | 0.222% | 0.809% | -0.587% | 1943.9 |
| i2c_aig | A_heuristic_only | -1.960% | -4.698% | +2.738% | 5.0 |
| i2c_aig | B_world_rerank | 1.156% | 0.325% | +0.831% | 16.0 |
| i2c_aig | C_depth2_rollout | 1.161% | 0.205% | +0.956% | 24.5 |
| max_aig | A_heuristic_only | 7.823% | -1.417% | +9.240% | 15.0 |
| max_aig | B_world_rerank | 8.819% | 12.957% | -4.138% | 48.5 |
| max_aig | C_depth2_rollout | 8.873% | 12.898% | -4.025% | 75.3 |
| b17_C | A_heuristic_only | 1.348% | -2.202% | +3.550% | 307.0 |
| b17_C | B_world_rerank | 1.346% | 2.227% | -0.881% | 1982.9 |
| b17_C | C_depth2_rollout | 0.328% | 2.949% | -2.621% | 2709.6 |
| mem_ctrl_aig | A_heuristic_only | 2.344% | 0.847% | +1.497% | 748.0 |
| mem_ctrl_aig | B_world_rerank | 2.558% | 1.402% | +1.156% | 3124.2 |
| mem_ctrl_aig | C_depth2_rollout | 2.772% | 1.827% | +0.945% | 5275.0 |

## Baseline / Final TC
| Circuit | Method | Baseline TC | Final TC | ΔTC |
|---|---|---:|---:|---:|
| b15_C | A_heuristic_only | 81.764% | 91.546% | 9.782% |
| b15_C | B_world_rerank | 81.764% | 90.054% | 8.290% |
| b15_C | C_depth2_rollout | 81.764% | 91.302% | 9.538% |
| b20_C | A_heuristic_only | 90.757% | 93.031% | 2.274% |
| b20_C | B_world_rerank | 90.757% | 93.096% | 2.339% |
| b20_C | C_depth2_rollout | 90.757% | 93.168% | 2.411% |
| b21_C | A_heuristic_only | 89.682% | 91.786% | 2.104% |
| b21_C | B_world_rerank | 89.682% | 91.733% | 2.051% |
| b21_C | C_depth2_rollout | 89.682% | 91.727% | 2.045% |
| b22_C | A_heuristic_only | 92.130% | 92.523% | 0.393% |
| b22_C | B_world_rerank | 92.130% | 92.597% | 0.467% |
| b22_C | C_depth2_rollout | 92.130% | 92.352% | 0.222% |
| i2c_aig | A_heuristic_only | 94.900% | 92.940% | -1.960% |
| i2c_aig | B_world_rerank | 94.900% | 96.056% | 1.156% |
| i2c_aig | C_depth2_rollout | 94.900% | 96.061% | 1.161% |
| max_aig | A_heuristic_only | 59.796% | 67.619% | 7.823% |
| max_aig | B_world_rerank | 59.796% | 68.615% | 8.819% |
| max_aig | C_depth2_rollout | 59.796% | 68.669% | 8.873% |
| b17_C | A_heuristic_only | 85.623% | 86.971% | 1.348% |
| b17_C | B_world_rerank | 85.623% | 86.969% | 1.346% |
| b17_C | C_depth2_rollout | 85.623% | 85.951% | 0.328% |
| mem_ctrl_aig | A_heuristic_only | 65.823% | 68.167% | 2.344% |
| mem_ctrl_aig | B_world_rerank | 65.823% | 68.381% | 2.558% |
| mem_ctrl_aig | C_depth2_rollout | 65.823% | 68.595% | 2.772% |
