import numpy as np
import os
from timeit import default_timer as timer

from agents.agents import PlayerAgent
from agents.dummy.random_card_agent import RandomCardAgent
from controller.dealing_behavior import DealWinnableHand, DealExactly
from controller.game_controller import GameController
from game.card import Suit
from game.game_mode import GameMode, GameContract
from game.game_state import Player
from utils.log_util import get_named_logger


def eval_agent(agent: PlayerAgent) -> float:
    """
    Evaluates an agent by playing 1000 games. For each game,
    - The agent is dealt quasi-random cards, with the condition that they enable a Herz-solo.
    - The cards are frozen and the game is repeated 100 times
    - The agent is replaced by a baseline (RandomCardAgent) and the game is again repeated 100 times
    - The win rate of the agent is compared with the baseline, the result is the relative performance improvement over the baseline.

    The final output is the mean relative performance improvement over all 1000 games.

    :param agent: The agent to evaluate against the baseline.
    :return: The mean relative performance improvement over the baseline.
    """

    logger = get_named_logger("{}.eval_agent".format(os.path.splitext(os.path.basename(__file__))[0]))

    # Main set of players
    players = [
        Player("0-agent", agent=agent),
        Player("1-Zenzi", agent=RandomCardAgent(1)),
        Player("2-Franz", agent=RandomCardAgent(2)),
        Player("3-Andal", agent=RandomCardAgent(3))
    ]

    # Baseline player: We compare the winrate of AlphaSau with a baseline agent (in this case, RandomCardAgent).
    # - For each game, we fix the cards dealt and sample the same game a number of times.
    # - From this we obtain the winrate of AlphaSau and the baseline for this specific game.
    # - Per game, we then calculate the "relative performance", comparing winrate with the baseline.
    # - Finally, we repeat this for a large number of games and measure the mean/median relative performance.
    baseline_agent = RandomCardAgent(0)
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
    mean_perf = np.mean(perf_record).item()
    median_perf = np.median(perf_record)
    logger.info("Finished evaluation. Took {:.0f} seconds.".format(s_elapsed))
    logger.info("Mean rel. performance={:.3f}. Median rel. performance={:.3f}. ".format(mean_perf, median_perf))

    return mean_perf
