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
