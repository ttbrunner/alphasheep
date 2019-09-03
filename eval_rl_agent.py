"""
Runs in an endless loop, constantly evaluating a checkpoint. Copies the checkpoint to save the best-performing model so far.
"""

import glob
import re
from time import sleep

import argparse
import logging
import os

from agents.dqn_agent import DQNAgent
from controller.game_controller import GameController
from eval_util import eval_agent
from log_util import init_logging, get_class_logger, get_named_logger


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", help="Weights are loaded from this file.")
    parser.add_argument("--loop", help="If set, then runs in an endless loop.", required=False, action="store_true")
    args = parser.parse_args()
    do_loop = args.loop is True
    checkpoint_path = args.checkpoint_path

    # Wait until a ".for_eval" checkpoint exists. Then rename it to ".in_eval".
    # After the end, it will be renamed to ".{score}".
    # In this way, both the training and multiple eval scripts can run in parallel, and
    checkpoint_path_in = f"{os.path.splitext(checkpoint_path)[0]}.for_eval.h5"
    checkpoint_path_tmp = f"{os.path.splitext(checkpoint_path)[0]}.in_eval.pid{os.getpid()}.h5"

    # Init logging and adjust log levels for some classes.
    init_logging()
    logger = get_named_logger("{}.main".format(os.path.splitext(os.path.basename(__file__))[0]))
    get_class_logger(GameController).setLevel(logging.INFO)     # Don't log specifics of a single game

    while True:

        # Wait until a new "for_eval" checkpoint exists (provided by the training script)
        while True:
            if os.path.exists(checkpoint_path_in):
                # Rename the file to "in_eval" and mark with PID so we don't collide with other workers
                try:
                    os.rename(checkpoint_path_in, checkpoint_path_tmp)
                    break
                except OSError:
                    # Probably a concurrent rename by another worker; continue and try again.
                    logger.exception("Could not rename checkpoint!")
            else:
                logger.info(f'No checkpoint found at "{checkpoint_path_in}"')

            logger.info("Waiting...")
            sleep(10)

        # Load the latest checkpoint and evaluate it
        logger.info('Found a new checkpoint, evaluating...')
        alphasau_agent = DQNAgent(0, training=False)
        alphasau_agent.load_weights(checkpoint_path_tmp)
        current_perf = eval_agent(alphasau_agent)

        # Now we know the performance. Find best-performing previous checkpoint that exists on disk
        splitext = os.path.splitext(checkpoint_path)
        checkpoints = glob.glob(os.path.join(os.path.dirname(checkpoint_path), "{}-*{}".format(splitext[0], splitext[1])))
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

            try:
                # The probability of a race condition between multiple eval processses is miniscule,
                # as the exact filename needs to collide at the same time as multiple saves. Not impossible though!
                os.rename(checkpoint_path_tmp, cp_best)
            except OSError as ex:
                # Log & continue.
                logger.exception(f"Could not rename checkpoint: {ex}")

        if not do_loop:
            # Run only once.
            return


if __name__ == '__main__':
    main()
