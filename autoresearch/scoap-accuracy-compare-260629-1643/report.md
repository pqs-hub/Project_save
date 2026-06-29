# SCOAP / Delta-SCOAP Accuracy Comparison

| metric | incumbent | control | version_A | version_B | best |
|---|---:|---:|---:|---:|---|
| `scoap_mae` | 0.068131 | 0.056134 | 0.059089 | 0.049371 | `version_B` |
| `scoap_acc_at_005` | 0.499832 | 0.549735 | 0.555089 | 0.603871 | `version_B` |
| `delta_scoap_mae` | 0.005661 | 0.003198 | 0.005066 | 0.005997 | `control` |
| `delta_scoap_acc_at_001` | 0.869946 | 0.977686 | 0.873118 | 0.927258 | `control` |

## Conclusion

- Version B has the best direct SCOAP prediction: lowest `scoap_mae` and highest `scoap_acc_at_005`.
- Control has the best delta-SCOAP prediction: lowest `delta_scoap_mae` and highest `delta_scoap_acc_at_001`.
- Version A improves SCOAP accuracy over incumbent, but its delta-SCOAP accuracy is only close to incumbent, not close to control.
- These SCOAP metrics do not explain planner ranking by themselves: B predicts SCOAP best but derived action value still ranks poorly.
