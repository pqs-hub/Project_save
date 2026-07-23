#!/usr/bin/env bash
set -euo pipefail

bash autoresearch/loop-260720-0945/run_collect_onpolicy_plans_round5.sh
bash autoresearch/loop-260720-0945/run_collect_prefix_oracle_round5.sh
bash autoresearch/loop-260720-0945/run_model_training_round5.sh
bash autoresearch/loop-260720-0945/run_b15_model_selection_round5.sh
bash autoresearch/loop-260720-0945/run_five_typed_winner_round5.sh
