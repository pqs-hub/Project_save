# Model-training research basis

- V-JEPA 2-AC: preserve a pretrained encoder and post-train an action-conditioned latent predictor on interaction trajectories.
- TD-MPC2: jointly supervise latent consistency, immediate reward, and value under one transferable hyperparameter set.
- DeepGate4: circuit models benefit from gate-aware local/global structural encoding and sparse graph attention.
- FF-JEPA: reduce long-horizon collapse through hierarchical short-horizon latent targets; our first implementation uses multi-step return and overshooting because current real sequences have length five.

Local data audit: the 100,131-row Atalanta-BIST dataset contains 20,000 real sequences of exactly five inserted points at 300,000 random patterns. The incumbent configurations set `lambda_return=0`, so the return head was not trained despite being used by several planner scores.
