"""
Runs in an endless loop, constantly evaluating a checkpoint. Copies the checkpoint to save the best-performing model so far.
"""

import glob
import re
from time import sleep

import numpy as np
import argparse
import logging
import os

from agents.agents import RandomCardAgent
from agents.dqn_agent import DQNAgent
from controller.dealing_behavior import DealWinnableHand, DealExactly
from controller.game_controller import GameController
from game.card import Suit
from game.game_mode import GameContract, GameMode
from game.game_state import Player
from log_util import init_logging, get_class_logger, get_named_logger
from timeit import default_timer as timer


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", help="Weights are loaded from this file.")
    parser.add_argument("--loop", help="If set, then runs in an endless loop.", required=False, action="store_true")
    args = parser.parse_args()
    do_loop = args.loop is True
    checkpoint_path = args.checkpoint_path

    # During evaluation, the checkpoint is renamed so we know that this process is working on it.
    checkpoint_path_tmp = f"{checkpoint_path}.in_eval.pid{os.getpid()}"

    # Init logging and adjust log levels for some classes.
    init_logging()
    logger = get_named_logger("{}.main".format(os.path.splitext(os.path.basename(__file__))[0]))
    get_class_logger(GameController).setLevel(logging.INFO)     # Don't log specifics of a single game

    while True:

        # Wait for a new checkpoint to appear (written by training script)
        while True:
            if os.path.exists(checkpoint_path):
                # Rename the file so other workers don't pick it up
                try:
                    os.rename(checkpoint_path, checkpoint_path_tmp)
                    break
                except OSError:
                    # Probably a concurrent rename by another worker; continue and try again.
                    logger.exception("Could not rename checkpoint!")
            else:
                logger.info(f'No checkpoint found at "{checkpoint_path}"')

            logger.info("Waiting...")
            sleep(10)

        # Load the latest checkpoint and evaluate it
        logger.info('Found a new checkpoint, evaluating...')
        alphasau_agent = DQNAgent(training=False)
        alphasau_agent.load_weights(checkpoint_path_tmp)
        current_perf = eval_checkpoint(alphasau_agent)

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


def eval_checkpoint(alphasau_agent):
    logger = get_named_logger("{}.eval_checkpoint".format(os.path.splitext(os.path.basename(__file__))[0]))

    # Main set of players
    players = [
        Player("0-AlphaSau", agent=alphasau_agent),
        Player("1-Zenzi", agent=RandomCardAgent()),
        Player("2-Franz", agent=RandomCardAgent()),
        Player("3-Andal", agent=RandomCardAgent())
    ]

    # Baseline player: We compare the winrate of AlphaSau with a baseline agent (in this case, RandomCardAgent).
    # - For each game, we fix the cards dealt and sample the same game a number of times.
    # - From this we obtain the winrate of AlphaSau and the baseline for this specific game.
    # - Per game, we then calculate the "relative performance", comparing winrate with the baseline.
    # - Finally, we repeat this for a large number of games and measure the mean/median relative performance.
    baseline_agent = RandomCardAgent()
    baseline_players = [Player("0-Baseline", agent=baseline_agent), *players[1:]]

    # Rig the game so Player 0 has the cards to play a Herz-Solo.
    game_mode = GameMode(GameContract.suit_solo, trump_suit=Suit.herz, declaring_player_id=0)
    rng_dealer = DealWinnableHand(game_mode)

    # Run 1k different games. Each games is sampled 100 times.
    n_games = 1000
    n_baseline_samples = 100
    n_agent_samples = 100
    perf_record = np.empty(n_games, dtype=np.float32)

    time_start = timer()
    for i_game in range(n_games):
        if i_game > 0 and i_game % 10 == 0:
            s_elapsed = timer() - time_start
            mean_perf = np.mean(perf_record[:i_game])
            median_perf = np.median(perf_record[:i_game])
            logger.info("Ran {} games. Mean rel. performance={:.3f}. Median rel. performance={:.3f}. "
                        "Speed is {:.1f} games/second.".format(i_game, mean_perf, median_perf, i_game/s_elapsed))

        # Deal a single random hand and then create a dealer that will replicate this hand, so we can compare how different agents
        #  would fare with exactly this hand.
        # We might want to create a mechanism for replicating exact game states in the future.
        player_hands = rng_dealer.deal_hands()
        replicating_dealer = DealExactly(player_hands)
        i_player_dealer = i_game % 4

        def sample_games(sample_players, n_samples):
            n_samples_won = 0
            for i_sample in range(n_samples):
                controller = GameController(sample_players, i_player_dealer=i_player_dealer,
                                            dealing_behavior=replicating_dealer, forced_game_mode=game_mode)
                winners = controller.run_game()
                if winners[0] is True:
                    n_samples_won += 1
            return n_samples_won / n_samples

        def rel_performance(win_rate, win_rate_base):
            eps = 1e-2
            if win_rate_base < eps:
                if win_rate < eps:
                    return 1.0                         # special case: 0/0 := 1
                win_rate_base = eps                    # special case: x/0 := x/0.01
            return win_rate / win_rate_base

        baseline_win_rate = sample_games(baseline_players, n_baseline_samples)
        agent_win_rate = sample_games(players, n_agent_samples)
        perf = rel_performance(agent_win_rate, baseline_win_rate)
        logger.debug("Baseline win rate: {:.1%}. Agent win rate: {:.1%}. Relative agent performance={:.3f}".format(
            baseline_win_rate, agent_win_rate, perf))

        perf_record[i_game] = perf

    s_elapsed = timer() - time_start
    mean_perf = np.mean(perf_record)
    median_perf = np.median(perf_record)
    logger.info("Finished evaluation. Took {:.0f} seconds.".format(s_elapsed))
    logger.info("Mean rel. performance={:.3f}. Median rel. performance={:.3f}. ".format(mean_perf, median_perf))

    return mean_perf


if __name__ == '__main__':
    main()
