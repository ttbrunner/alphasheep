import logging
import os

from agents.rule_based.rule_based_agent import RuleBasedAgent
from controller.game_controller import GameController
from evaluation import eval_agent
from utils.log_util import init_logging, get_class_logger, get_named_logger


def main():

    # Init logging and adjust log levels for some classes.
    init_logging()
    logger = get_named_logger("{}.main".format(os.path.splitext(os.path.basename(__file__))[0]))
    get_class_logger(GameController).setLevel(logging.INFO)     # Don't log specifics of a single game

    agent = RuleBasedAgent(0)
    perf = eval_agent(agent)


if __name__ == '__main__':
    main()
