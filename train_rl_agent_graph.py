"""
Trains an agent, as specified in an experiment config file.
"""
import argparse
import logging
import os
import shutil
import time
from collections import deque

from agents.dummy.random_card_agent import RandomCardAgent
from agents.reinforcment_learning.dqn_agent import DQNAgent
from agents.rule_based.rule_based_agent import RuleBasedAgent
from simulator.controller.dealing_behavior import DealWinnableHand
from simulator.controller.game_controller import GameController
from simulator.card_defs import Suit
from simulator.game_mode import GameContract, GameMode
from simulator.game_state import Player
from utils.log_util import init_logging, get_class_logger, get_named_logger
from timeit import default_timer as timer

from utils.config_util import load_config

#################################################################################
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('dark_background')

def live_plotter(x_vec,y1_data,line1,identifier='training progress',pause_time=0.1):
    if line1==[]:
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        fig = plt.figure(figsize=(13,6))
        ax = fig.add_subplot(111)
        # create a variable for the line so we can later update it
        line1, = ax.plot(x_vec,y1_data,'-',alpha=0.8)        
        #update plot label/title
        plt.ylabel('win rate')
        plt.xlabel('episodes')
        plt.title('Title: {}'.format(identifier))
        plt.grid()
        plt.tight_layout()
        #plt.style.use('ggplot')
        #plt.style.use('classic')
        plt.show()
    
    # after the figure, axis, and line are created, we only need to update the y-data
    line1.set_ydata(y1_data)
    # adjust limits if new data goes beyond bounds
    if np.min(y1_data)<=line1.axes.get_ylim()[0] or np.max(y1_data)>=line1.axes.get_ylim()[1]:
        plt.ylim([np.max([0,0.9*np.min(y1_data)]),np.min([1,1.1*np.max(y1_data)])])
    # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
    plt.pause(pause_time)
    
    # return line so we can update it again in the next iteration
    return line1

#####################################################################################
def main():
    line1 = []
    # Game Setup:
    # - In every game, Player 0 will play a Herz-Solo
    # - The cards are rigged so that Player 0 always receives a pretty good hand, most of them should be winnable.

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="An experiment config file. Must always be specified.", required=True)
    args = parser.parse_args()
    

    # Init logging and adjust log levels for some classes.
    init_logging()
    logger = get_named_logger("{}.main".format(os.path.splitext(os.path.basename(__file__))[0]))
    get_class_logger(GameController).setLevel(logging.INFO)     # Don't log specifics of a single game

    # Load config.
    # Create experiment dir and prepend it to all paths.
    # If it already exists, then training will simply resume from existing checkpoints in that dir.
    logger.info(f'Loading config from "{args.config}"...')
    config = load_config(args.config)
    experiment_dir = config["experiment_dir"]
    os.makedirs(config["experiment_dir"], exist_ok=True)
    agent_checkpoint_paths = {i: os.path.join(experiment_dir, name) for i, name in config["training"]["agent_checkpoint_names"].items()}

    # Create agents.
    agents = []
    for i in range(4):
        x = config["training"]["player_agents"][i]
        if x == "DQNAgent":
            agent = DQNAgent(i, config=config, training=True)
        elif x == "RandomCardAgent":
            agent = RandomCardAgent(i)
        elif x == "RuleBasedAgent":
            agent = RuleBasedAgent(i)
        else:
            raise ValueError(f'Unknown agent type: "{x}"')
        agents.append(agent)

    # Load weights for agents.
    for i, weights_path in agent_checkpoint_paths.items():
        if not os.path.exists(weights_path):
            logger.info('Weights file "{}" does not exist. Will create new file.'.format(weights_path))
        else:
            agents[i].load_weights(weights_path)

    players = [Player(f"Player {i} ({a.__class__.__name__})", agent=a) for i, a in enumerate(agents)]

    # Rig the game so Player 0 has the cards to play a Herz-Solo. Force them to play it.
#    game_mode = GameMode(GameContract.suit_solo, trump_suit=Suit.eichel, declaring_player_id=0)
#    game_mode = GameMode(GameContract.suit_solo, trump_suit=Suit.gras, declaring_player_id=0)
    game_mode = GameMode(GameContract.suit_solo, trump_suit=Suit.herz, declaring_player_id=0)
#    game_mode = GameMode(GameContract.suit_solo, trump_suit=Suit.schellen, declaring_player_id=0)
     
    controller = GameController(players, dealing_behavior=DealWinnableHand(game_mode), forced_game_mode=game_mode)

    n_episodes = config["training"]["n_episodes"]
    logger.info(f"Will train for {n_episodes} episodes.")

    # Calculate win% as simple moving average (just for display in the logfile).
    # The real evaluation is done in eval_rl_agent.py, with training=False.
    win_rate = float('nan')
    n_won = 0
    sma_window_len = 1000
    won_deque = deque()

    save_every_s = config["training"]["save_checkpoints_every_s"]

#####################################################
    # autospwan evaluation window
    #os.system("start /B start cmd.exe @cmd /k eval_rl_agent.py --config experiments\dqn_solo_decl_6_6_6.yaml --loop")
#####################################################
    # definitions for realtime-graphic
    size = n_episodes
    x_vec = np.linspace(0,size,(size//100)+1)[0:-1]
    y_vec = (size//100)*[0]
    y_vec[0] = 0.25
    line1 = live_plotter(x_vec,y_vec,line1,args.config)
#####################################################

#####################################################
    
    time_start = timer()
    time_last_save = timer()
    for i_episode in range(n_episodes):
        if i_episode > 0:
            # Calculate avg win%
            if i_episode < sma_window_len:
                win_rate = n_won / i_episode
            else:
                if won_deque.popleft() is True:
                    n_won -= 1
                win_rate = n_won / sma_window_len

            # Log
            if i_episode % 100 == 0:
                s_elapsed = timer() - time_start
                logger.info("Ran {} Episodes. Win rate (last {} episodes) is {:.1%}. Speed is {:.0f} episodes/second.".format(
                    i_episode, sma_window_len, win_rate, i_episode/s_elapsed))
#####################################################
                y_vec[i_episode//100] = win_rate
                line1 = live_plotter(x_vec,y_vec,line1)
#####################################################

            # Save model checkpoint.
            # Also make a copy for evaluation - the eval jobs will sync on this file and later remove it.
            if timer() - time_last_save > save_every_s:
                for i, weights_path in agent_checkpoint_paths.items():
                    agents[i].save_weights(weights_path, overwrite=True)
                    shutil.copyfile(weights_path, f"{os.path.splitext(weights_path)[0]}.for_eval.h5")
                time_last_save = timer()
                
                #autospwan evaluation windos
                #os.system("start /B start cmd.exe @cmd /k eval_rl_agent.py --config experiments\dqn_solo_decl_6_6_6.yaml")

        winners = controller.run_game()
        won = winners[0]
        won_deque.append(won)
        if won:
            n_won += 1

    # save final results
    for i, weights_path in agent_checkpoint_paths.items():
        agents[i].save_weights(weights_path, overwrite=True)
        shutil.copyfile(weights_path, f"{os.path.splitext(weights_path)[0]}.for_eval.h5")

    logger.info("Finished training.")
    logger.info("Final win rate: {:.1%}".format(win_rate))
    
#####################################################
    plt.savefig(os.path.join(weights_path + 'training_') + time.strftime("%Y_%m_%d_%H_%M_%S") + '.png')
    plt.close()


if __name__ == '__main__':
    main()
