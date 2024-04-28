import re
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, NewType, TypeGuard

from ability_data import ABILITY_DATA
from weapon_data import WEAPON_DATA
from stage_data import STAGE_DATA

WeaponKey = NewType("WeaponKey", str)
AbilityKey = NewType("AbilityKey", str)
StageKey = NewType("StageKey", str)

WEAPON_KEYS = [weapon["key"] for weapon in WEAPON_DATA]
ABILITY_KEYS = [ability["key"] for ability in ABILITY_DATA]
STAGE_KEYS = [stage["key"] for stage in STAGE_DATA]


def is_weapon_key(key: str) -> TypeGuard[WeaponKey]:
    return key in WEAPON_KEYS


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
        winning_team: The team that won the battle.
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


@dataclass
class BattleParticipant:
    """
    The data of a participant in a battle.

    Attributes:
        team: The team the participant belongs to.
        team_member_num: The number of the participant in the team. 1 to 4.
        weapon: The weapon the participant used.
        num_kills_and_assists: num_kills + num_assists (redundant, but here because it's in the data)
        num_kills: The number of kills the participant got.
        num_assists: The number of assists the participant got.
        num_deaths: The number times the participant died.
        num_special_uses: The number of times the participant used their special.
        turf_inked: The amount of turf the participant inked.
    """

    team: TeamKey
    team_member_num: Literal[1, 2, 3, 4]
    weapon: WeaponKey
    num_kills_and_assists: int
    num_kills: int
    num_assists: int
    num_deaths: int
    num_special_uses: int
    turf_inked: int
    abilities: dict[AbilityKey, float]
