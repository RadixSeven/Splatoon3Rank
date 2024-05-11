import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, NewType, TypeGuard

from ability_data import ABILITY_DATA
from kit_data import KIT_DATA
from stage_data import STAGE_DATA

KitKey = NewType("KitKey", str)
AbilityKey = NewType("AbilityKey", str)
StageKey = NewType("StageKey", str)

KIT_KEYS = [kit["key"] for kit in KIT_DATA]
ABILITY_KEYS = [ability["key"] for ability in ABILITY_DATA]
STAGE_KEYS = [stage["key"] for stage in STAGE_DATA]


logger = logging.getLogger(__name__)


def is_kit_key(key: str) -> TypeGuard[KitKey]:
    return key in KIT_KEYS


def is_ability_key(key: str) -> TypeGuard[AbilityKey]:
    return key in ABILITY_KEYS


def is_stage_key(key: str) -> TypeGuard[StageKey]:
    return key in STAGE_KEYS


LOBBY_DESCRIPTIONS: dict[str, str] = {
    "regular": "Regular",
    "bankara_challenge": "Anarchy (Series)",
    "bankara_open": "Anarchy (Open)",
    "xmatch": "X Battle",
    "splatfest_challenge": "Splatfest (Pro)",
    "splatfest_open": "Splatfest (Open)",
    "event": "Challenge Event",
}

LobbyKey = NewType("LobbyKey", str)


def is_lobby_key(key: str) -> TypeGuard[LobbyKey]:
    return key in LOBBY_DESCRIPTIONS.keys()


MODE_DESCRIPTIONS: dict[str, str] = {
    "nawabari": "Turf War",
    "area": "Splat Zones",
    "yagura": "Tower Control",
    "hoko": "Rainmaker",
    "asari": "Clam Blitz",
}

ModeKey = NewType("ModeKey", str)


TEAM_KEYS = ["alpha", "bravo"]

TeamKey = NewType("TeamKey", Literal["alpha", "bravo"])


def is_team_key(key: str) -> TypeGuard[TeamKey]:
    return key in TEAM_KEYS


def is_mode_key(key: str) -> TypeGuard[ModeKey]:
    return key in MODE_DESCRIPTIONS.keys()


RankString = NewType("RankString", str)

# Note that the empty string is a valid rank string.
RankPattern = re.compile(r"(?:[CBAS][+-]?|S\+ [0-9]|S\+ [1-4][0-9]|S\+ 50)?")


def is_rank_string(key: str) -> TypeGuard[RankString]:
    return bool(RankPattern.fullmatch(key))


ColorString = NewType("ColorString", str)

ColorPattern = re.compile(r"[0-9a-f]{8}")


def is_color_string(key: str) -> TypeGuard[ColorString]:
    return bool(ColorPattern.fullmatch(key))


@dataclass
class BattleRecord:
    """
    A record of a battle.

    Derived from:
    https://github.com/fetus-hina/stat.ink/wiki/Spl3-%EF%BC%8D-CSV-Schema-%EF%BC%8D-Battle

    Attributes:
        season: The season of the battle. example: "Chill Season 2022"
        period: During which "period" (2-hour window) did the battle
                take place. (Stored as an ISO 8601 string in the CSV
                file.)
        game_version: The version of the game the battle took place in. example: "2.0.1"
        lobby: The lobby the battle took place in. example: "bankara_open"
        mode: The mode the battle took place in. example: "area"
        stage: The stage the battle took place in. example: "masaba"
        length: The length of the battle in seconds.
        winning_team: The team that won the battle. (Tri-color battles and private battles
           are removed from the data before ``stat.ink`` outputs it.)
        knockout: True if the battle was won by a knockout.
        rank: The rank of the submitter in the battle. "" if not a ranked battle.
        power: The power of the submitter in the battle. Anarchy Power,
            Challenge Power, Splatfest Power, or X Power. None if no power
            available.
        participants: The participants in the battle.
        medals: The medals the player received in the battle.
        event_name: The name of the challenge event. "" if not a challenge event.
    """

    season: str
    period: datetime
    game_version: str
    lobby: LobbyKey
    mode: ModeKey
    stage: StageKey
    length: int
    winning_team: TeamKey
    knockout: bool
    rank: RankString
    power: float | None
    team_characteristics: dict[TeamKey, "TeamCharacteristics"]
    participants: dict[tuple[TeamKey, Literal[1, 2, 3, 4]], "BattleParticipant"]
    medals: list["Medal"]
    event_name: str


@dataclass
class RankedPerformance:
    """Performance in a ranked match.

    Attributes:
        count: The ranked mode result
    """

    count: int


@dataclass
class TurfWarPerformance:
    """Performance in a turf war match.

    Attributes:
        inked: The amount of turf the team inked in points.
        inked_percent: The percentage of the stage the team inked.
    """

    inked: int | None
    inked_percent: float


@dataclass
class TeamCharacteristics:
    """
    The team's performance and characteristics in a battle.

    Attributes:
        team: The team the characteristics belong to.
        color: The color of the team (hexadecimal: rr gg bb aa). Example: d0bf08ff.
        performance: The performance of the team in the battle. (If recorded)
        theme: The splatfest team name. "" if not a splatfest battle.
    """

    team: TeamKey
    color: ColorString | None
    performance: RankedPerformance | TurfWarPerformance | None
    theme: str


MedalGrade = NewType("MedalGrade", Literal["gold", "silver"])


def is_medal_grade(key: str) -> TypeGuard[MedalGrade]:
    return key in ["gold", "silver"]


@dataclass
class Medal:
    """
    The data of a medal a player received in a battle.

    Attributes:
        name: The name of the medal. Known medal names are:
                https://stat.ink/api-info/medal3
                Sometimes this may contain a localized name rather
                than the English name. Using only English would be
                a future improvement.
        grade: The grade of the medal. "gold" or "silver".
    """

    name: str
    grade: MedalGrade


# noinspection SpellCheckingInspection
# The map from kit keys of re-skinned kits to
# their original kits.
_normalized_kit_keys = {
    kit["key"]: kit["reskin_of"] for kit in KIT_DATA if "reskin_of" in kit
}


def log_missing_duplicate_kit_keys():
    """Log the missing duplicate kit keys."""
    kits_by_elements = defaultdict(list)
    for kit in KIT_DATA:
        kits_by_elements[
            (kit["main"], kit["sub"]["key"], kit["special"]["key"])
        ].append(kit["key"])
    for elements, identical_kits in kits_by_elements.items():
        if len(identical_kits) > 1:
            for kit in identical_kits:
                if kit in _normalized_kit_keys.keys():
                    if _normalized_kit_keys[kit] in identical_kits:
                        continue
                    logger.error(
                        f'The normalized kit "{_normalized_kit_keys[kit]}" for {kit}'
                        f" is not in the identical kits list {sorted(identical_kits)}."
                    )
                if kit in _normalized_kit_keys.values():
                    continue
                logger.error(
                    f"Missing duplicate kit key: {kit} with elements {elements}. "
                    f"Other identical kits: {sorted(set(identical_kits) - {kit})}."
                )


# Check for missing duplicate kit keys.
log_missing_duplicate_kit_keys()


@dataclass
class BattleParticipant:
    """
    The data of a participant in a battle.

    Attributes:
        team: The team the participant belongs to.
        team_member_num: The number of the participant in the team. 1 to 4.
        kit: The weapon kit the participant used.
        num_kills_and_assists: num_kills + num_assists (redundant, but here because it's in the data)
        num_kills: The number of kills the participant got.
        num_assists: The number of assists the participant got.
        num_deaths: The number times the participant died.
        num_special_uses: The number of times the participant used their special.
        turf_inked: The amount of turf the participant inked.
    """

    team: TeamKey
    team_member_num: Literal[1, 2, 3, 4]
    kit: KitKey
    num_kills_and_assists: int
    num_kills: int
    num_assists: int
    num_deaths: int
    num_special_uses: int
    turf_inked: int
    abilities: dict[AbilityKey, float]

    @property
    def normalized_kit(self):
        return _normalized_kit_keys.get(self.kit, self.kit)
