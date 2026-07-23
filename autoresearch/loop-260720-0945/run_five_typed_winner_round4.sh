#!/usr/bin/env bash
set -euo pipefail

export BASE=autoresearch/loop-260720-0945/model_training_round4
export OUT=autoresearch/loop-260720-0945/typed_winner_round4_five
exec bash autoresearch/loop-260720-0945/run_five_typed_winner_round3.sh
