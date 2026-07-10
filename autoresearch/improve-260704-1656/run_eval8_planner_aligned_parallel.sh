#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/data4/pengqingsong/DFT/TPI-my.3}
cd "$REPO_ROOT"

echo "[launcher] run_eval8_planner_aligned_parallel.sh is now a strict Table-II protocol wrapper."
echo "[launcher] It uses configs/eval_protocol_coverage_only.json fixed #TP budgets."

export OUT_ROOT=${OUT_ROOT:-autoresearch/improve-260704-1656/eval8_planner_aligned_parallel_oldbudgets}
exec bash autoresearch/improve-260704-1656/run_eval8_planner_aligned_parallel_oldbudgets.sh
