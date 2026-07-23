#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
OUT_ROOT="autoresearch/loop-260718-0223/deterministic_replay_round17"
mkdir -p "$OUT_ROOT/logs"

export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TPI_PLAN_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TPI_TORCH_DETERMINISTIC=1
export TPI_HARD_CLUSTER_MAX_HARD_NODES=1024 TPI_LATENT_NORM_CLIP_RATIO=4
export TPI_Q_CONTEXT_SUPPORT_ALPHA=0.45 TPI_Q_CONTEXT_DISAGREEMENT_BETA=0.10
export TPI_SCORE_QUANTIZATION=0.001
export TPI_ADAPTIVE_BASE_CANDIDATES=48 TPI_ADAPTIVE_EXPANSION_MARGIN=0.01

run_one() {
    local replica="$1" gpu="$2"
    local out="$OUT_ROOT/$replica/b15_C"
    echo "[round17] start replica=$replica gpu=$gpu"
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_gmean_sweep.py \
        --eval-protocol configs/eval_protocol_coverage_only.json --protocol-keep-cli-benchmarks \
        --benchmarks iscas99__b15_1 \
        --checkpoint runs/planner_aligned_q_rank_v5_context_safe/best_final_horizon.pt \
        --planners greedy --score-fields q_pred_context --beam-objectives cumulative \
        --beam-widths 1 --lookahead-depths 1 --max-candidates 64 --discount-gammas 0.9 \
        --candidate-strategies hard_fault_cluster --candidate-diversity-penalties 0.0 \
        --candidate-diversity-depths 4 \
        --candidate-real-fault-priors autoresearch/eval8-real-priors-budget5-v1/real_fault_priors.csv \
        --candidate-allowlist autoresearch/original-netlist-recovery-260712/exact_itc99/b15_C/exact_candidate_nodes.txt \
        --plan-device cuda --time-limit-hours 72 --out-dir "$out" 2>&1 \
        | sed -u "s|^|[round17/$replica] |" | tee "$OUT_ROOT/logs/$replica.log"
    echo "[round17] done replica=$replica"
}

run_one replay_a 0 & p1=$!
run_one replay_b 1 & p2=$!
failed=0
wait "$p1" || failed=1
wait "$p2" || failed=1
if (( failed )); then exit 1; fi

python - <<'PY'
import csv
import glob
import hashlib
import json

plans = sorted(glob.glob("autoresearch/loop-260718-0223/deterministic_replay_round17/*/b15_C/plans/*.csv"))
rows = [list(csv.DictReader(open(path))) for path in plans]
actions = [[(row["node"], row["type"]) for row in replica] for replica in rows]
payload = {
    "plans": plans,
    "sequence_equal": actions[0] == actions[1],
    "position_equal": sum(left == right for left, right in zip(actions[0], actions[1])),
    "set_overlap": len(set(actions[0]) & set(actions[1])),
    "sha256": [hashlib.sha256(repr(action).encode()).hexdigest() for action in actions],
}
print(json.dumps(payload, indent=2))
open("autoresearch/loop-260718-0223/deterministic_replay_round17/replay_audit.json", "w").write(json.dumps(payload, indent=2) + "\n")
PY
