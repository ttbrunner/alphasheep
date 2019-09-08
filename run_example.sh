#!/bin/bash
# Starts a game with DQNAgent as Player 0.

python3 play_with_gui.py --p0-agent=alphasau --alphasau-checkpoint="experiments/dqn_solo_decl_inv_g9_lr0001/model-p0-0.4955500066280365.h5" --agent-config="experiments/dqn_solo_decl_inv_g9_lr0001.yaml"
