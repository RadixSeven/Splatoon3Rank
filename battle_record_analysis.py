import logging
import numpy as np
import pymc as pm
import pytensor.tensor as tt

from battle_record import BattleRecord, BattleParticipant, KitKey
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
    battle_record: BattleRecord, kit_index: dict[KitKey, int]
) -> NDArray[np.int8]:
    """
    Return an array of observed kit variables for a single battle record.
    These will be the coefficients multiplied by the kit strength factors
    in the model.  The first half of the array will be the observed
    kit variables for team alpha, and the second half will be the observed
    kit variables for team bravo.

    Importantly, the observed variables for team alpha will be negated.
    Since the team strength is the sum of the kit factors and the
    final probability is 1/(1+exp(bravo_strength-alpha_strength)), this
    embeds the final subtraction in the observed variables.

    Args:
        battle_record: The battle record to examine.
        kit_index: A mapping from kit keys to indices.

    Returns:
        An array of observed kit variables for the given battle record.
    """
    num_kits = len(kit_index)
    a = np.zeros(2 * num_kits, dtype=np.int8)
    for participant in battle_record.participants.values():
        if participant.team == "alpha":
            a[kit_index[participant.normalized_kit]] -= 1
        else:
            a[num_kits + kit_index[participant.normalized_kit]] += 1
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
) -> tuple[NDArray[np.int8], NDArray[np.uint8]]:
    """
    Return a tuple containing two arrays: the observed kit
    variables for all the given battle records and the results of those
    battles. The observed kit will be the coefficients multiplied
    by the kit strength factors in the model. The results will be
    as described in ``single_battle_observed_alpha_win``.

    (The data does not contain tri-color battles or battles with draws,
    so there is always a winner.)

    If ``rv`` the return value, then ``rv[0][i]`` will be the observed kit
    variables for a single battle record and ``rv[1][i]`` will be the result
    of that battle.

    Args:
        battle_records: The battle records to examine.
        kit_index: A mapping from kit keys to indices.

    Returns:
        A tuple containing two arrays: the observed kit variables for all the given battle records
        and the results of those battles.
    """
    observed_kit_variables = []
    battle_results = []

    for battle_record in tqdm(battle_records, desc="Processing battle records"):
        observed_kit_variables.append(
            single_battle_observed_kit_variables(battle_record, kit_index)
        )
        battle_results.append(single_battle_observed_alpha_win(battle_record))

    return np.array(observed_kit_variables), np.array(battle_results)


def kit_only_model(
    battle_record_creator: Callable[[], Iterable[BattleRecord]]
) -> pm.Model:
    """
    Create a PyMC3 model for the given battle record creator.

    Team strength is determined by the kits used by the team members.
    let A = alpha_strength and B = bravo_strength
    prob_team_alpha_win = exp(A)/(exp(A)+exp(B))
    This can be rewritten as prob_team_alpha_win = 1/(1+exp(B)/exp(A)) = 1/(1+exp(B-A))

    Here is the formula for each battle record:
    kit_strength = pm.Exp("kit_strength", 1, dims=("kits",))
    both_sides_kit_strengths = concatenate(kit_strength, kit_strength)
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
    observed_kit_variables, battle_results = (
        all_observed_kit_variables_and_battle_results(
            battle_record_creator(), kit_index
        )
    )
    shuffled_indices = np.random.permutation(battle_results.shape[0])

    logger.info(f"Done reading.")
    logger.info(f"Number of kits: {len(kits)}")
    logger.info(f"Shape of observed_kit_variables: {observed_kit_variables.shape}")
    logger.info(f"Shape of battle_results: {battle_results.shape}")

    coords = {"kits": kits}
    with pm.Model(coords=coords) as model:
        observed_kit_data = pm.Data(
            "observed_kit_data", observed_kit_variables[shuffled_indices]
        )
        battle_results_data = pm.Data(
            "battle_results_data", battle_results[shuffled_indices]
        )

        kit_strength = tt.reshape(
            pm.Normal("kit_strength", mu=1, sigma=0, dims="kits"), (1, -1)
        )
        both_sides_kit_strengths = pm.math.concatenate(
            [kit_strength, kit_strength], axis=1
        )
        logger.info(
            f"Shape of both_sides_kit_strengths: {both_sides_kit_strengths.shape.eval()}"
        )
        all_matches_kit_strengths = tt.repeat(
            both_sides_kit_strengths,
            observed_kit_data.shape[0],
            axis=0,
        )
        logger.info(
            f"Shape of all_matches_kit_strengths: {all_matches_kit_strengths.shape.eval()}"
        )

        kit_contributions = all_matches_kit_strengths * observed_kit_data
        team_beta_advantage = tt.exp(
            tt.reshape(tt.sum(kit_contributions, axis=1), (-1, 1))
        )
        logger.info(f"Shape of team_beta_advantage: {team_beta_advantage.shape.eval()}")

        prob_team_alpha_win = 1 / (1 + team_beta_advantage)
        pm.Bernoulli(
            "battle_results", prob_team_alpha_win, observed=battle_results_data
        )

    return model
