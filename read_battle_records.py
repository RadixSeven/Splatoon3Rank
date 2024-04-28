import json
import logging
import os
from datetime import datetime
from io import TextIOWrapper
from typing import TypeGuard, Literal, TextIO, Generator, cast
from zipfile import Path, ZipFile

from tqdm import tqdm

from battle_record import (
    TeamKey,
    AbilityKey,
    is_ability_key,
    BattleParticipant,
    is_weapon_key,
    RankedPerformance,
    TurfWarPerformance,
    TeamCharacteristics,
    is_color_string,
    Medal,
    is_medal_grade,
    BattleRecord,
    is_lobby_key,
    is_rank_string,
    is_mode_key,
    is_stage_key,
    is_team_key,
)

BRAVO_TEAM = TeamKey("bravo")
ALPHA_TEAM = TeamKey("alpha")

logger = logging.getLogger(__name__)


class Err:
    """
    An error that has occurred but not been logged or handled yet.
    """

    def __init__(self, msg: str):
        self.msg = msg

    def log(self) -> None:
        logger.error(self.msg)

    def __str__(self) -> str:
        return self.msg

    def __repr__(self) -> str:
        return f"Err({self.msg})"


def parse_abilities(abilities_json: str) -> dict[AbilityKey, float] | Err:
    """
    Parse the abilities JSON string into a dictionary of abilities.

    Args:
        abilities_json: The JSON string representing the abilities.
    Returns:
        A dictionary of abilities if the JSON string is valid, otherwise Err.
    """
    try:
        loaded = json.loads(abilities_json)
    except json.JSONDecodeError as e:
        return Err(f"Failed to decode abilities JSON. {e}")
    bad_keys = {key for key in loaded.keys() if not is_ability_key(key)}
    if len(bad_keys) > 0:
        return Err(f"Invalid ability key found in abilities JSON: {bad_keys}")
    try:
        return {AbilityKey(key): float(value) for key, value in loaded.items()}
    except ValueError as e:
        return Err(f"Failed to convert ability value to float. {e}")


def is_valid_member_num(member_num: int) -> TypeGuard[Literal[1, 2, 3, 4]]:
    return 1 <= member_num <= 4


def create_participant(
    data: dict[str, str], team: TeamKey, member_num: int
) -> BattleParticipant | Err:
    """
    Create a BattleParticipant from a row of CSV data.

    Args:
        data: The row data for a single row of the CSV
        team: The team the participant belongs to.
        member_num: The number of the participant in the team. 1 to 4.
    Returns:
        A BattleParticipant if the data is valid, otherwise an Err.
    """
    # The ID of the participant. e.g., A1, B3, etc.
    id_ = f"{team[0].upper()}{member_num}"
    weapon_key = data[f"{id_}-weapon"]
    if is_valid_member_num(member_num) and is_weapon_key(weapon_key):
        return BattleParticipant(
            team=team,
            team_member_num=member_num,
            weapon=weapon_key,
            num_kills_and_assists=int(data[f"{id_}-kill-assist"]),
            num_kills=int(data[f"{id_}-kill"]),
            num_assists=int(data[f"{id_}-assist"]),
            num_deaths=int(data[f"{id_}-death"]),
            num_special_uses=int(data[f"{id_}-special"]),
            turf_inked=int(data[f"{id_}-inked"]),
            abilities=parse_abilities(data[f"{id_}-abilities"]),
        )
    if not is_valid_member_num(member_num):
        Err("Invalid member number, must be between 1 and 4. {member_num}")
    return Err("Invalid weapon key. {weapon_key}")


def create_team_performance(
    team: TeamKey, raw_inked: str, raw_inked_percent: str, raw_count: str
) -> RankedPerformance | TurfWarPerformance | None | Err:
    """
    Create a performance object for a team from the raw data.

    Args:
        team: The team to create the performance for.
        raw_inked: The raw inked string from the CSV.
        raw_inked_percent: The raw inked percent string from the CSV.
        raw_count: The raw count string from the CSV.
    Returns:
        A RankedPerformance or TurfWarPerformance if the data is present for those modes,
        None if no data is present, or an Err if the data is invalid.
    """
    if raw_count:
        try:
            return RankedPerformance(int(raw_count))
        except ValueError:
            return Err(f"Invalid count data for team {team}: {raw_count}")
    elif raw_inked_percent:
        try:
            return TurfWarPerformance(
                inked=None if raw_inked == "" else int(raw_inked),
                inked_percent=float(raw_inked_percent),
            )
        except ValueError:
            return Err(
                f"Invalid turf-war performance data for team {team}"
                f"inked: {raw_inked}, inked_percent: {raw_inked_percent}"
            )
    if not raw_inked and not raw_inked_percent and not raw_count:
        return None
    return Err(
        f"Invalid performance data for team {team}"
        f"inked: {raw_inked}, inked_percent: {raw_inked_percent},"
        f"count: {raw_count}"
    )


def create_team_characteristics(
    data: dict[str, str], team: TeamKey
) -> TeamCharacteristics | Err:
    """
    Create TeamCharacteristics from a row of CSV data

    Args:
        data: The row data for a single row of the CSV
        team: The team to create the characteristics for.
    Returns:
        A TeamCharacteristics if the data is valid, otherwise an Err.
    """
    raw_color = data[f"{team}-color"]
    color = None if raw_color == "" else raw_color

    raw_inked = data[f"{team}-inked"]
    raw_inked_percent = data[f"{team}-ink-percent"]
    raw_count = data[f"{team}-count"]

    performance_or_err = create_team_performance(
        team, raw_inked, raw_inked_percent, raw_count
    )
    if isinstance(performance_or_err, Err):
        return performance_or_err
    performance = performance_or_err

    if color is None or is_color_string(color):
        return TeamCharacteristics(
            team=team,
            color=color,
            performance=performance,
            theme=data[f"{team}-theme"],
        )
    return Err(f"Invalid color key: '{raw_color}'")


def create_medal(
    data: dict[str, str], medal_num: Literal[1, 2, 3]
) -> Medal | None | Err:
    """
    Create a Medal from a row of CSV data

    Args:
        data: The row data for a single row of the CSV
        medal_num: The medal number to create.
    """
    grade = data[f"medal{medal_num}-grade"]
    name = data[f"medal{medal_num}-name"]

    if not grade:
        return None

    if is_medal_grade(grade):
        return Medal(name=name, grade=grade)

    return Err(f"Invalid medal grade: {grade}")


def is_medal_number(medal_num: int) -> TypeGuard[Literal[1, 2, 3]]:
    return 1 <= medal_num <= 3


def battle_record_for_row(row_num: int, row: dict[str, str]) -> BattleRecord | Err:
    """
    Create a BattleRecord from a row of data.

    Args:
        row_num: The row number.
        row: The row of data.
    Returns:
        A BattleRecord if the row is valid, otherwise an Err.
    """
    lobby_key = row["lobby"]
    mode_key = row["mode"]
    stage_key = row["stage"]
    winner_key = row["win"]
    rank_str = row["rank"]

    participant_keys: set[(TeamKey, Literal[1, 2, 3, 4])] = {
        (team, i) for i in [1, 2, 3, 4] for team in [ALPHA_TEAM, BRAVO_TEAM]
    }
    participants = {k: create_participant(row, k[0], k[1]) for k in participant_keys}
    if any(isinstance(v, Err) for v in participants.values()):
        errs = [v for v in participants.values() if isinstance(v, Err)]
        return Err(f"Invalid participant data, skipping row {row_num}. Errors: {errs}")
    team_characteristics = {
        ALPHA_TEAM: create_team_characteristics(row, ALPHA_TEAM),
        BRAVO_TEAM: create_team_characteristics(row, BRAVO_TEAM),
    }
    if any(isinstance(v, Err) for v in team_characteristics.values()):
        errs = [v for v in team_characteristics.values() if isinstance(v, Err)]
        return Err(
            f"Invalid team characteristics, skipping row {row_num}. Errors: {errs}"
        )

    raw_medals = [create_medal(row, i) for i in range(1, 3) if is_medal_number(i)]
    if any(isinstance(medal, Err) for medal in raw_medals):
        errs = [medal for medal in raw_medals if isinstance(medal, Err)]
        return Err(f"Invalid medal data, skipping row {row_num}. Errors: {errs}")
    medals = [medal for medal in raw_medals if medal is not None]
    if "# season" not in row:
        return Err(f"Missing season field on row {row_num}. Valid Fields: {row.keys()}")

    # The following if statement is required for the type checker to understand that the
    # lobby_key, mode_key, stage_key, and winner_key are valid keys.
    if (
        is_lobby_key(lobby_key)
        and is_rank_string(rank_str)
        and is_mode_key(mode_key)
        and is_stage_key(stage_key)
        and is_team_key(winner_key)
    ):
        return BattleRecord(
            season=row["# season"],
            period=datetime.fromisoformat(row["period"]),
            game_version=row["game-ver"],
            lobby=lobby_key,
            mode=mode_key,
            stage=stage_key,
            length=int(row["time"]),
            winning_team=winner_key,
            knockout=row["knockout"].lower() == "true",
            rank=rank_str,
            power=float(row["power"]) if row["power"] else None,
            team_characteristics=team_characteristics,
            participants=participants,
            medals=medals,
            event_name=row["event"],
        )

    if not is_lobby_key(lobby_key):
        return Err(f"Invalid lobby key: {lobby_key} on row {row_num}")
    if not is_rank_string(rank_str):
        return Err(f"Invalid rank string: {rank_str} on row {row_num}")
    if not is_mode_key(mode_key):
        return Err(f"Invalid mode key: {mode_key} on row {row_num}")
    if not is_stage_key(stage_key):
        return Err(f"Invalid stage key: {stage_key} on row {row_num}")
    return Err(f"Invalid win key: {winner_key} on row {row_num}")


def battles_from_csv(file: TextIO) -> Generator[BattleRecord, None, None]:
    """
    Returns all the battle records from the given CSV file.

    Args:
        file: The CSV file to read from.
    Returns:
        All battle records from the given CSV file.
    """
    from csv import DictReader

    reader = cast(DictReader, DictReader(file))
    with_errs = (
        battle_record_for_row(row_num, row) for row_num, row in enumerate(reader)
    )
    for value in with_errs:
        if isinstance(value, Err):
            value.log()
            continue
        yield value


def battle_records_from_zip(
    zip_file_path: str | Path,
) -> Generator[BattleRecord, None, None]:
    """
    Returns all the battle records from the given zip file.

    Args:
        zip_file_path: The path to the zip file containing the battle records.
    Returns:
        A list of all battle records from the given zip file.
    """
    with ZipFile(zip_file_path, "r") as z:
        csv_bytes = sum(
            i.file_size for i in z.infolist() if i.filename.endswith(".csv")
        )
        with tqdm(
            total=csv_bytes, desc="Reading battles", unit="Bytes", unit_scale=True
        ) as pbar:
            for info in z.infolist():
                filename = info.filename
                base_name = os.path.basename(filename)
                pbar.set_postfix_str(base_name)
                # Ensure it's a csv file (if it ends in .csv, we can assume it's not a directory)
                if filename.endswith(".csv"):
                    with z.open(filename) as file:
                        for battle in battles_from_csv(TextIOWrapper(file, "utf-8")):
                            yield battle
                    pbar.update(info.file_size)
