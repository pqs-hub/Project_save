# Summary

Research phase complete. The selected must-have feature is a prefix-counterfactual, support-constrained typed residual. It attacks the measured Round4 failure mode: selected-only supervision, early off-policy divergence, and CP0 over-selection.

The immediate experiment uses the existing Round4 checkpoint to isolate the planner-side trust mechanism. Round5 then retrains the typed heads from balanced same-prefix real ATPG labels at the final 300k evaluation budget. Encoder changes are deliberately deferred until the action-label and policy-distribution mismatch is resolved.

Primary success remains one exact-legal configuration, selected on b15 only, beating DeepTPI on all five circuits.

