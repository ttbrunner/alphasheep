#!/bin/bash
# Starts a game with DQNAgent as Player 0.

python3 play_with_gui.py --p0-agent=alphasheep --alphasheep-checkpoint="experiments/dqn_solo_decl_inv_g99_lr0001/model-p0-0.5342000126838684.h5" --agent-config="experiments/dqn_solo_decl_inv_g99_lr0001.yaml"
