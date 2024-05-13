import logging
import numpy as np
import pymc as pm
import pytensor.tensor as tt

from battle_record import BattleRecord, BattleParticipant, KitKey, TeamKey
from itertools import chain
from numpy.typing import NDArray
from tqdm import tqdm
from typing import Iterable, Callable


"""
This module contains functions for analyzing battle records and collections thereof.
"""


logger = logging.getLogger(__name__)


def all_kits_used(battle_records: Iterable[BattleRecord]) -> set[KitKey]:
    """
    Return the set of all kits used in the given battle records.

    Args:
        battle_records: The battle records to examine.

    Returns:
        The set of all kits used in the given battle records.
    """
    participants: Iterable[BattleParticipant] = chain.from_iterable(
        battle_record.participants.values() for battle_record in battle_records
    )
    return {participant.normalized_kit for participant in participants}


def single_battle_observed_kit_variables(
    battle_record: BattleRecord, kit_index: dict[KitKey, int], team: TeamKey
) -> NDArray[np.int8]:
    """
    Return an array of observed kit variables for a single battle record
    and team.
    These will be the coefficients multiplied by the kit strength factors
    in the model. Right now, they are the count of the number of occurrences
    of the kit in the team. Later will have different factors for the second
    and further appearances of a kit (or a main) on a team.

    Args:
        battle_record: The battle record to examine.
        kit_index: A mapping from kit keys to indices.
        team: The team to whose kits will be returned.

    Returns:
        An array of observed kit variables for the given battle record.
    """
    num_kits = len(kit_index)
    a = np.zeros(num_kits, dtype=np.int8)
    for participant in battle_record.participants.values():
        if participant.team == team:
            a[kit_index[participant.normalized_kit]] += 1
    return a


def single_battle_observed_alpha_win(battle_record: BattleRecord) -> NDArray[np.uint8]:
    """
    Return an array containing a single element, 1 if team alpha won the battle, 0 otherwise.

    Args:
        battle_record: The battle record to examine.

    Returns:
        An array containing a single element, 1 if team alpha won the battle, 0 otherwise.
    """
    return np.array([1 if battle_record.winning_team == "alpha" else 0], dtype=np.uint8)


def all_observed_kit_variables_and_battle_results(
    battle_records: Iterable[BattleRecord], kit_index: dict[KitKey, int]
) -> tuple[NDArray[np.int8], NDArray[np.int8], NDArray[np.uint8]]:
    """
    Return a tuple containing three arrays: the observed kit
    variables for all the given battle records for each team, and the results of those
    battles. The observed kit will be the coefficients multiplied
    by the kit strength factors in the model. The results will be
    as described in ``single_battle_observed_alpha_win``.

    (The data does not contain tri-color battles or battles with draws,
    so there is always a winner.)

    If ``rv`` the return value, then ``rv[0][i]`` will be the observed kit
    variables for the alpha team for a single battle record,
    ``rv[1][i]`` for bravo, and ``rv[2][i]`` will be the result
    of that battle.

    Args:
        battle_records: The battle records to examine.
        kit_index: A mapping from kit keys to indices.

    Returns:
        A tuple containing two arrays: the observed kit variables for all the given battle records
        and the results of those battles.
    """
    observed_kit_variables_alpha = []
    observed_kit_variables_bravo = []
    battle_results = []

    for battle_record in tqdm(battle_records, desc="Processing battle records"):
        observed_kit_variables_alpha.append(
            single_battle_observed_kit_variables(battle_record, kit_index, "alpha")
        )
        observed_kit_variables_bravo.append(
            single_battle_observed_kit_variables(battle_record, kit_index, "bravo")
        )
        battle_results.append(single_battle_observed_alpha_win(battle_record))

    return (
        np.array(observed_kit_variables_alpha),
        np.array(observed_kit_variables_bravo),
        np.array(battle_results),
    )


def kit_only_model(
    battle_record_creator: Callable[[], Iterable[BattleRecord]]
) -> pm.Model:
    """
    Create a PyMC3 model for the given battle record creator.

    Team strength is determined by the kits used by the team members.
    let A = alpha_strength and B = bravo_strength.
    prob_team_alpha_win = exp(a)/(exp(b)+exp(a))
    This can be rewritten as prob_team_alpha_win = (exp(a)/exp(b))/(1+(exp(a)/exp(b)))=inverse_logit(a-b)

    Here is the formula for each battle record:
    kit_strengths = pm.Exp("kit_strengths", 1, dims=("kits",))
    both_sides_kit_strengths = concatenate(kit_strengths, kit_strengths)
    The basic formula is prob_team_alpha_win for each battle = exp(
        dot(kit_strengths, observed_kits)
    )
    Kit strength's prior distributed as a normal variable with mean 1

    Args:
        battle_record_creator: A callable that returns an
            iterable of battle records (generally by re-opening the file)

    Returns:
        A PyMC3 model for the given battle record creator.
    """
    kits = sorted(all_kits_used(battle_record_creator()))
    kit_index = {kit: i for i, kit in enumerate(kits)}
    observed_kit_variables_alpha, observed_kit_variables_bravo, battle_results = (
        all_observed_kit_variables_and_battle_results(
            battle_record_creator(), kit_index
        )
    )
    shuffled_indices = np.random.permutation(battle_results.shape[0])

    logger.info(f"Done reading.")
    logger.info(f"Number of kits: {len(kits)}")
    logger.info(
        f"Shape of observed_kit_variables_alpha: {observed_kit_variables_alpha.shape}"
    )
    logger.info(
        f"Shape of observed_kit_variables_bravo: {observed_kit_variables_bravo.shape}"
    )
    logger.info(f"Shape of battle_results: {battle_results.shape}")

    coords = {"kits": kits}
    with pm.Model(coords=coords) as model:
        observed_kit_data_alpha = pm.Data(
            "observed_kit_data_alpha", observed_kit_variables_alpha[shuffled_indices]
        )
        observed_kit_data_bravo = pm.Data(
            "observed_kit_data_bravo", observed_kit_variables_bravo[shuffled_indices]
        )
        battle_results_data = pm.Data(
            "battle_results_data", battle_results[shuffled_indices]
        )

        starter_kit_strengths = pm.Normal("kit_strengths", mu=0, sigma=1, dims="kits")
        kit_strengths = tt.reshape(starter_kit_strengths, (1, -1))
        logger.info(f"Shape of kit_strengths: {kit_strengths.shape.eval()}")
        all_matches_kit_strengths = tt.repeat(
            kit_strengths,
            observed_kit_data_alpha.shape[0],
            axis=0,
        )
        logger.info(
            f"Shape of all_matches_kit_strengths: {all_matches_kit_strengths.shape.eval()}"
        )

        kit_contributions_alpha = all_matches_kit_strengths * observed_kit_data_alpha
        kit_contributions_bravo = all_matches_kit_strengths * observed_kit_data_bravo
        log_alpha_strength = tt.reshape(
            tt.sum(kit_contributions_alpha, axis=1), (-1, 1)
        )
        log_bravo_strength = tt.reshape(
            tt.sum(kit_contributions_bravo, axis=1), (-1, 1)
        )
        logger.info(f"Shape of log_alpha_strength: {log_alpha_strength.shape.eval()}")

        # Correct for the bias that team alpha wins more than team beta by estimating
        # the amount that alpha wins more often. Multiply this by alpha_win_prob so that
        # in an even match (by kit) - that is 50% by kit, alpha wins at a rate of
        # base_win_probability
        base_win_probability = pm.Beta("base_win_probability", alpha=0.57, beta=0.5)
        pm.Bernoulli(
            "base_win_likelihood", base_win_probability, observed=battle_results_data
        )

        alpha_win_prob = pm.invlogit(log_alpha_strength - log_bravo_strength)
        # noinspection PyTypeChecker
        adjusted_win_prob = alpha_win_prob * (base_win_probability / 0.5)
        # Clip to between 0 and 1 (and exclude the endpoints since those can have unusual results)
        # The right way to do this is to select the priors appropriately to keep
        # the multiplied factor in range for the Bernoulli likelihood, but this is simple enough for now.
        clipped_adjusted_win_prob = tt.clip(adjusted_win_prob, 2**-18, 1 - (2**-18))
        pm.Bernoulli(
            "battle_results", p=clipped_adjusted_win_prob, observed=battle_results_data
        )

    return model
