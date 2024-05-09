from itertools import chain
from typing import Iterable, Callable

from numpy.typing import NDArray
from tqdm import tqdm

import pymc as pm
import numpy as np

from battle_record import BattleRecord, BattleParticipant, WeaponKey

"""
This module contains functions for analyzing battle records and collections thereof.
"""


def all_weapons_used(battle_records: Iterable[BattleRecord]) -> set[WeaponKey]:
    """
    Return the set of all weapons used in the given battle records.

    Args:
        battle_records: The battle records to examine.

    Returns:
        The set of all weapons used in the given battle records.
    """
    participants: Iterable[BattleParticipant] = chain.from_iterable(
        battle_record.participants.values()
        for battle_record in tqdm(battle_records, desc="Tallying weapons used")
    )
    return {participant.weapon for participant in participants}


def single_battle_observed_weapon_variables(
    battle_record: BattleRecord, weapon_index: dict[WeaponKey, int]
) -> NDArray[np.int8]:
    """
    Return an array of observed weapon variables for a single battle record.
    These will be the coefficients multiplied by the weapon strength factors
    in the model.  The first half of the array will be the observed
    weapon variables for team alpha, and the second half will be the observed
    weapon variables for team bravo.

    Args:
        battle_record: The battle record to examine.
        weapon_index: A mapping from weapon keys to indices.

    Returns:
        An array of observed weapon variables for the given battle record.
    """
    num_weapons = len(weapon_index)
    a = np.zeros(2 * num_weapons, dtype=np.int8)
    for participant in battle_record.participants.values():
        if participant.team == "alpha":
            a[weapon_index[participant.weapon]] += 1
        else:
            a[num_weapons + weapon_index[participant.weapon]] -= 1
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


def all_observed_weapon_variables_and_battle_results(
    battle_records: Iterable[BattleRecord], weapon_index: dict[WeaponKey, int]
) -> tuple[NDArray[np.int8], NDArray[np.uint8]]:
    """
    Return a tuple containing two arrays: the observed weapon
    variables for all the given battle records and the results of those
    battles. The observed weapon will be the coefficients multiplied
    by the weapon strength factors in the model. The results will be
    as described in ``single_battle_observed_alpha_win``.

    TODO: what about tri-color battles?

    If ``rv`` the return value, then ``rv[0][i]`` will be the observed weapon
    variables for a single battle record and ``rv[1][i]`` will be the result
    of that battle.

    Args:
        battle_records: The battle records to examine.
        weapon_index: A mapping from weapon keys to indices.

    Returns:
        A tuple containing two arrays: the observed weapon variables for all the given battle records
        and the results of those battles.
    """
    observed_weapon_variables = []
    battle_results = []

    for battle_record in tqdm(battle_records, desc="Processing battle records"):
        observed_weapon_variables.append(
            single_battle_observed_weapon_variables(battle_record, weapon_index)
        )
        battle_results.append(single_battle_observed_alpha_win(battle_record))

    return np.array(observed_weapon_variables), np.array(battle_results)


def weapon_only_model(
    battle_record_creator: Callable[[], Iterable[BattleRecord]]
) -> pm.Model:
    """
    Create a PyMC3 model for the given battle record creator.

    Here is the formula for each battle record:
    weapon_strength = pm.Exp("weapon_strength", 1, dims=("weapons",))
    both_sides_weapon_strengths = concatenate(weapon_strength, weapon_strength)
    num_and_den_weapon_strengths = concatenate(both_sides_weapon_strengths, both_sides_weapon_strengths)
    num_and_den_observed_weapons = concatenate(observed_weapon_variables, -abs(observed_weapon_variables))
    The basic formula is prob_team_alpha_win for each battle = exp(
        dot(num_and_den_weapon_strengths, num_and_den_observed_weapons)
    )
    Weapon strength's prior distributed as an exponential variable with mean 1

    Args:
        battle_record_creator: A callable that returns an
            iterable of battle records (generally by re-opening the file)

    Returns:
        A PyMC3 model for the given battle record creator.
    """
    weapons = sorted(all_weapons_used(battle_record_creator()))
    weapon_index = {weapon: i for i, weapon in enumerate(weapons)}
    observed_weapon_variables, battle_results = (
        all_observed_weapon_variables_and_battle_results(
            battle_record_creator(), weapon_index
        )
    )

    coords = {"weapons": weapons}
    with pm.Model(coords=coords) as model:
        observed_weapon_data = pm.Data(
            "observed_weapon_data", observed_weapon_variables
        )
        battle_results_data = pm.Data("battle_results_data", battle_results)

        weapon_strength = pm.Exponential("weapon_strength", 1, dims=("weapons",))
        both_sides_weapon_strengths = pm.math.concatenate(
            weapon_strength, weapon_strength
        )
        num_and_den_weapon_strengths = pm.math.concatenate(
            both_sides_weapon_strengths, both_sides_weapon_strengths
        )
        num_and_den_observed_weapons = pm.math.concatenate(
            observed_weapon_data, -abs(observed_weapon_data)
        )
        prob_team_alpha_win = pm.math.exp(
            pm.math.dot(num_and_den_weapon_strengths, num_and_den_observed_weapons)
        )
        pm.Bernoulli(
            "battle_results", prob_team_alpha_win, observed=battle_results_data
        )

    return model
