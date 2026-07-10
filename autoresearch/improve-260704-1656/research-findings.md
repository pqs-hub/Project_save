# Research Findings

## Finding 1: Accuracy Is Not Planner Ranking

The new v3/v4 models improved static trained-head metrics but failed b21 insertion. The best new b21 result was v3 `reward_pred` at 1.150% Delta TC, far below the incumbent 5.838%.

Confidence: HIGH.

## Finding 2: v4 Is Numerically Unsafe In Closed Loop

v4 generated NaN scores in b21 planning for `reward_pred`, `hard_reduction_total_pred`, `derived_hard_reduction_hybrid_pred`, and `hybrid_pred`. This disqualifies it for planner use without stability changes.

Confidence: HIGH.

## Finding 3: The Incumbent Recipe Matches Deployment

The incumbent is `rollout_loss_A_reward_only/epoch_009.pt`. Its config uses horizon 5 and emphasizes reward/fc behavior with weak hard-head regularization. This is closer to deployed `world_rerank` than the newer hard-accuracy configs.

Confidence: HIGH.

## Finding 4: Existing Code Supports Oracle Candidate Ranking

`tpi_jepa.train` already supports pairwise/listwise oracle ranking through `lambda_q_rank`, `lambda_candidate`, `lambda_q_value`, and arbitrary `oracle_ranking_score_field`. No new model architecture is needed for the first improvement attempt.

Confidence: HIGH.

## Finding 5: Gate Needs To Be Real Insertion

Validation accuracy did not predict b21 insertion performance. Future model selection should use b21 `world_rerank` as a cheap gate before full 8-circuit evaluation.

Confidence: HIGH.
