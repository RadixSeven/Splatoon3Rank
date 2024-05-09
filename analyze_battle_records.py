from itertools import chain
from typing import Iterable

import numpy as np
from tqdm import tqdm

from battle_record import BattleRecord, BattleParticipant, WeaponKey


def all_weapons_used(battle_records: Iterable[BattleRecord]) -> set[str]:
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
) -> np.ndarray:
    """
    Return an array of observed weapon variables for a single battle record.
    These will be the coefficients multiplied by the weapon strength factors
    in the model.

    Note that since the model is additive, having the same weapon on both
    sides cancels out.

    Args:
        battle_record: The battle record to examine.
        weapon_index: A mapping from weapon keys to indices.

    Returns:
        An array of observed weapon variables for the given battle record.
    """
    a = np.zeros(len(weapon_index))
    for participant in battle_record.participants.values():
        a[weapon_index[participant.weapon]] += 1 if participant.team == "alpha" else -1
    return a


def all_observed_weapon_variables(
    battle_records: Iterable[BattleRecord], weapon_index: dict[WeaponKey, int]
) -> np.ndarray:
    """
    Return an array of observed weapon variables for all the given battle
    records. These will be the coefficients multiplied by the weapon strength
    factors in the model.

    If ``rv`` is the return value, then ``rv[i]`` will be the observed weapon
    variables for a single battle record.

    Args:
        battle_records: The battle records to examine.
        weapon_index: A mapping from weapon keys to indices.

    Returns:
        An array of observed weapon variables for the given battle records.
    """
    return np.array(
        [
            single_battle_observed_weapon_variables(battle_record, weapon_index)
            for battle_record in tqdm(battle_records, desc="Computing weapon variables")
        ]
    )
