"""
Evaluates the winrate of a baseline agent.

This script doesn't need an agent config because all the baseline agents are self-contained.
"""

import argparse
import logging
import os

from agents.dummy.random_card_agent import RandomCardAgent
from agents.dummy.static_policy_agent import StaticPolicyAgent
from agents.rule_based.rule_based_agent import RuleBasedAgent
from simulator.controller.game_controller import GameController
from evaluation import eval_agent
from utils.log_util import init_logging, get_class_logger, get_named_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p0-agent", type=str, choices=['static', 'rule', 'random'], required=True)
    args = parser.parse_args()
    agent_choice = args.p0_agent

    # Init logging and adjust log levels for some classes.
    init_logging()
    logger = get_named_logger("{}.main".format(os.path.splitext(os.path.basename(__file__))[0]))
    get_class_logger(GameController).setLevel(logging.INFO)     # Don't log specifics of a single game

    # Create the agent for Player 0.
    if agent_choice == "rule":
        agent = RuleBasedAgent(0)
    elif agent_choice == "static":
        agent = StaticPolicyAgent(0)
    else:
        agent = RandomCardAgent(0)

    logger.info(f'Evaluating agent "{agent.__class__.__name__}"')
    perf = eval_agent(agent)


if __name__ == '__main__':
    main()
