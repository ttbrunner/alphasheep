import numpy as np
import os
from timeit import default_timer as timer

from simulator.player_agent import PlayerAgent
from agents.rule_based.rule_based_agent import RuleBasedAgent
from simulator.controller.dealing_behavior import DealWinnableHand, DealExactly
from simulator.controller.game_controller import GameController
from simulator.card_defs import Suit
from simulator.game_mode import GameMode, GameContract
from simulator.game_state import Player
from utils.log_util import get_named_logger


def eval_agent(agent: PlayerAgent) -> float:
    """
    Evaluates an agent by playing a large number of games against 3 RuleBasedAgents.

    :param agent: The agent to evaluate.
    :return: The mean win rate of the agent.
    """

    logger = get_named_logger("{}.eval_agent".format(os.path.splitext(os.path.basename(__file__))[0]))
    # logger.setLevel(logging.DEBUG)

    # Main set of players
    players = [
        Player("0-agent", agent=agent),
        Player("1-Zenzi", agent=RuleBasedAgent(1)),
        Player("2-Franz", agent=RuleBasedAgent(2)),
        Player("3-Andal", agent=RuleBasedAgent(3))
    ]

    # Rig the game so Player 0 has the cards to play a Herz-Solo.
    game_mode = GameMode(GameContract.suit_solo, trump_suit=Suit.herz, declaring_player_id=0)
    rng_dealer = DealWinnableHand(game_mode)

    # Run 40k different games. Each game can replicated (via DealExactly) and sampled multiple times.
    # Right now, our baseline (RuleBasedAgent) is almost deterministic, so it's ok to sample each game only once.
    n_games = 40000
    n_agent_samples = 1
    perf_record = np.empty(n_games, dtype=np.float32)

    time_start = timer()
    for i_game in range(n_games):
        if i_game > 0 and i_game % 100 == 0:
            s_elapsed = timer() - time_start
            mean_perf = np.mean(perf_record[:i_game])
            logger.info("Ran {} games. Mean agent winrate={:.3f}. "
                        "Speed is {:.1f} games/second.".format(i_game, mean_perf, i_game/s_elapsed))

        # Deal a single random hand and then create a dealer that will replicate this hand,
        # so we can take multiple samples of this game.
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

        agent_win_rate = sample_games(players, n_agent_samples)
        logger.debug("Agent win rate: {:.1%}.".format(agent_win_rate))

        perf_record[i_game] = agent_win_rate

    s_elapsed = timer() - time_start
    mean_perf = np.mean(perf_record).item()
    logger.info("Finished evaluation. Took {:.0f} seconds.".format(s_elapsed))
    logger.info("Mean agent winrate={:.3f}.".format(mean_perf))

    return mean_perf
