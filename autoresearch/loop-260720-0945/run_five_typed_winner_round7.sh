#!/usr/bin/env bash
set -euo pipefail

BASE=autoresearch/loop-260720-0945/model_training_round7 \
OUT=autoresearch/loop-260720-0945/typed_winner_round7_five \
exec bash autoresearch/loop-260720-0945/run_five_typed_winner_round3.sh

