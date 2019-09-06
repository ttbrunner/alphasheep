"""
Runs in an endless loop, constantly evaluating a checkpoint. Copies the checkpoint to save the best-performing model so far.
"""

import glob
import re
from time import sleep

import argparse
import logging
import os

from agents.reinforcment_learning.dqn_agent import DQNAgent
from controller.game_controller import GameController
from eval_util import eval_agent
from log_util import init_logging, get_class_logger, get_named_logger
from utils import load_config


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="A yaml config file. Must always be specified.", required=True)
    parser.add_argument("--loop", help="If set, then runs in an endless loop.", required=False, action="store_true")
    args = parser.parse_args()
    do_loop = args.loop is True

    # Init logging and adjust log levels for some classes.
    init_logging()
    logger = get_named_logger("{}.main".format(os.path.splitext(os.path.basename(__file__))[0]))
    get_class_logger(GameController).setLevel(logging.INFO)     # Don't log specifics of a single game

    # Load config and check experiment dir.
    logger.info(f'Loading config from "{args.config}"...')
    config = load_config(args.config)
    experiment_dir = config["experiment_dir"]
    while not os.path.exists(experiment_dir):
        logger.warn(f'The experiment dir specified in the config does not exist: "{experiment_dir}" - waiting...')
        sleep(10)

    agent_checkpoint_paths = {i: os.path.join(experiment_dir, name) for i, name in config["training"]["agent_checkpoint_names"].items()}

    while True:

        # Wait until a ".for_eval" checkpoint exists (for any of possibly multiple agents). Then rename it to ".in_eval".
        # After the end, it will be renamed to ".{score}".
        # In this way, both the training and multiple eval scripts can run in parallel.
        for i_agent, cp_path in agent_checkpoint_paths.items():
            checkpoint_path_in = f"{os.path.splitext(cp_path)[0]}.for_eval.h5"
            checkpoint_path_tmp = f"{os.path.splitext(cp_path)[0]}.in_eval.pid{os.getpid()}.h5"
            if os.path.exists(checkpoint_path_in):
                # Found a new checkpoint.
                # Rename the file to "in_eval" and mark with PID so we don't collide with other workers
                try:

                    # Load the latest checkpoint and evaluate it
                    os.rename(checkpoint_path_in, checkpoint_path_tmp)
                    logger.info('Found a new checkpoint, evaluating...')

                    # Create agent
                    agent_type = config["training"]["player_agents"][i_agent]
                    if agent_type == "DQNAgent":
                        alphasau_agent = DQNAgent(0, config=config, training=False)
                    else:
                        raise ValueError(f"Unknown agent type specified: {agent_type}")
                    alphasau_agent.load_weights(checkpoint_path_tmp)

                    # Eval agent
                    current_perf = eval_agent(alphasau_agent)

                    # Now we know the performance. Find best-performing previous checkpoint that exists on disk
                    splitext = os.path.splitext(cp_path)
                    checkpoints = glob.glob(os.path.join(os.path.dirname(cp_path), "{}-*{}".format(splitext[0], splitext[1])))
                    best_perf = 0.
                    for cp in checkpoints:
                        perf_str = re.findall(r"{}-(.*){}".format(os.path.basename(splitext[0]), splitext[1]), cp)
                        if len(perf_str) > 0:
                            p = float(perf_str[0])
                            if p > best_perf:
                                best_perf = p

                    logger.info("Comparing performance to previous checkpoints...")
                    if best_perf > 0:
                        logger.info("Previously best checkpoint has performance {}".format(best_perf))
                    else:
                        logger.info("Did not find any previous results.")

                    if current_perf > best_perf:
                        best_perf = current_perf
                        logger.info("Found new best-performing checkpoint!")
                        cp_best = "{}-{}{}".format(splitext[0], str(best_perf), splitext[1])

                        os.rename(checkpoint_path_tmp, cp_best)

                except OSError:
                    # Probably a concurrent rename by another worker; continue and try again.
                    logger.exception("Could not rename checkpoint!")

        logger.info("Waiting...")
        sleep(10)

        if not do_loop:
            # Run only once.
            return


if __name__ == '__main__':
    main()
