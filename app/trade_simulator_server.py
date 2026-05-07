from __future__ import annotations

import csv
import json
import mimetypes
import os
import sys
import threading
import unicodedata
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = ROOT / "code"
VENDOR_DIR = CODE_DIR / "vendor"
STATIC_DIR = Path(__file__).resolve().parent / "static"
ASSETS_DIR = ROOT / "assets"
RUNTIME_DIR = Path(__file__).resolve().parent / "runtime"
UNIFIED_DATA_DIR = ROOT / "data" / "unified"
PUBLIC_RUNTIME_DATA_DIR = ROOT / "runtime_data"
AVATAR_INDEX_PATH = ASSETS_DIR / "statmuse_avatars_2025_26" / "index.csv"
PLAYER_NAME_MAP_PATH = ASSETS_DIR / "hupu_player_names_2025_26.csv"
CURRENT_TEAM_OVERRIDES_PATH = ASSETS_DIR / "nba_current_team_overrides_2025_26.csv"
END_ROSTER_PATH = ASSETS_DIR / "nba_end_rosters_2025_26.csv"
PLAYER_ADVANCED_IMPACT_PATH = ASSETS_DIR / "player_advanced_impact_2025_26.csv"
PLAYER_SALARY_PATH = ASSETS_DIR / "player_salaries_2025_26.csv"
TEAM_LOGO_DIR = ASSETS_DIR / "team_logos_2025_26"
CURRENT_YEAR = 2026
NEXT_YEAR = 2027
CURRENT_SEASON_LABEL = "2025-26"
NEXT_SEASON_LABEL = "2026-27"
PUBLIC_PLAYER_DATA_PATH = PUBLIC_RUNTIME_DATA_DIR / "current_players_2025_26.csv"
PUBLIC_TEAM_DATA_PATH = PUBLIC_RUNTIME_DATA_DIR / "current_teams_2025_26.csv"
PUBLIC_WEIGHTED_PLAYER_PATH = PUBLIC_RUNTIME_DATA_DIR / "recent_weighted_rotation_players_2025_26.csv"
PRIVATE_PLAYER_DATA_PATH = UNIFIED_DATA_DIR / "nba_player_stats_unified_1975_76_to_2025_26.csv"
PRIVATE_TEAM_DATA_PATH = UNIFIED_DATA_DIR / "nba_team_stats_unified_1975_76_to_2025_26.csv"
PRIVATE_WEIGHTED_PLAYER_CACHE_PATH = CODE_DIR / "outputs" / "recent_weighted_rotation_players_1975_76_to_2025_26.csv"
USE_PUBLIC_RUNTIME_DATA = PUBLIC_PLAYER_DATA_PATH.exists() or PUBLIC_TEAM_DATA_PATH.exists() or PUBLIC_WEIGHTED_PLAYER_PATH.exists()
PLAYER_DATA_PATH = PUBLIC_PLAYER_DATA_PATH if PUBLIC_PLAYER_DATA_PATH.exists() else PRIVATE_PLAYER_DATA_PATH
TEAM_DATA_PATH = PUBLIC_TEAM_DATA_PATH if PUBLIC_TEAM_DATA_PATH.exists() else PRIVATE_TEAM_DATA_PATH
SITE_METRICS_PATH = RUNTIME_DIR / "site_metrics.json"
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8765"))
MAX_JSON_BODY_BYTES = 32_768
PUBLIC_ASSET_DIRS = {"brand", "statmuse_avatars_2025_26", "team_logos_2025_26"}

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import numpy as np
import pandas as pd

from inference_runtime import (
    PLAYER_INPUT_COLUMNS,
    PLAYER_GROUP_COLUMNS,
    PLAYER_TARGET_COLUMNS,
    RUNTIME_PLAYER_SOURCE_COLUMNS,
    RUNTIME_TEAM_SOURCE_COLUMNS,
    TEAM_INPUT_COLUMNS,
    TEAM_TARGET_COLUMNS,
    TEAM_TARGET_COLUMNS_2,
    aggregate_movement_features,
    build_team_roster_features,
    build_recent_weighted_player_stats,
    filter_effective_rotation_players,
    load_players,
    load_teams,
    aggregate_group,
)
from numpy_resnet import NumpyTabularResNet


TEAM_DISPLAY_COLUMNS = [
    "wins",
    "losses",
    "win_pct",
    "points_per_game",
    "rebounds_per_game",
    "assists_per_game",
    "turnovers_per_game",
    "true_shooting_pct",
    "effective_field_goal_pct",
    "three_point_pct",
    "pace",
    "offensive_rating",
    "defensive_rating",
    "net_rating",
]

PLAYER_STATS_TABLE_COLUMNS = [
    "games_played",
    "minutes_per_game",
    "points_per_game",
    "rebounds_per_game",
    "assists_per_game",
    "steals_per_game",
    "blocks_per_game",
    "turnovers_per_game",
    "field_goal_pct",
    "three_point_pct",
    "free_throw_pct",
    "true_shooting_pct",
    "effective_field_goal_pct",
    "per",
]

PLAYER_IMPACT_TABLE_COLUMNS = [
    "epm",
    "o_epm",
    "d_epm",
    "dpm",
    "o_dpm",
    "d_dpm",
    "rapm",
    "o_rapm",
    "d_rapm",
    "lebron",
    "o_lebron",
    "d_lebron",
    "bpm",
    "obpm",
    "dbpm",
    "ws",
    "ows",
    "dws",
    "ws_per_48",
    "vorp",
    "bbref_ts_pct",
    "bbref_three_point_attempt_rate",
    "bbref_free_throw_attempt_rate",
    "bbref_off_reb_pct",
    "bbref_def_reb_pct",
    "bbref_total_reb_pct",
    "bbref_assist_pct",
    "bbref_steal_pct",
    "bbref_block_pct",
    "bbref_turnover_pct",
    "bbref_usage_pct",
    "nba_off_rating",
    "nba_def_rating",
    "nba_net_rating",
    "nba_est_off_rating",
    "nba_est_def_rating",
    "nba_est_net_rating",
    "nba_assist_pct",
    "nba_assist_to",
    "nba_assist_ratio",
    "nba_off_reb_pct",
    "nba_def_reb_pct",
    "nba_reb_pct",
    "nba_team_turnover_pct",
    "nba_est_turnover_pct",
    "nba_effective_fg_pct",
    "nba_true_shooting_pct",
    "nba_usage_pct",
    "nba_est_usage_pct",
    "nba_est_pace",
    "nba_pace",
    "nba_pie",
    "nba_possessions",
]

PLAYER_IMPACT_SOURCE_COLUMNS = [
    "source_basketball_reference",
    "source_databallr",
    "source_nba_stats",
    "source_epm",
    "source_lebron",
]

TEAM_STATS_TABLE_COLUMNS = [
    "wins",
    "losses",
    "win_pct",
    "points_per_game",
    "rebounds_per_game",
    "assists_per_game",
    "steals_per_game",
    "blocks_per_game",
    "turnovers_per_game",
    "field_goal_pct",
    "three_point_pct",
    "free_throw_pct",
    "true_shooting_pct",
    "effective_field_goal_pct",
    "pace",
    "offensive_rating",
    "defensive_rating",
    "net_rating",
]

PLAYER_DISPLAY_COLUMNS = [
    "games_played",
    "minutes_per_game",
    "points_per_game",
    "rebounds_per_game",
    "assists_per_game",
    "steals_per_game",
    "blocks_per_game",
    "turnovers_per_game",
    "true_shooting_pct",
    "effective_field_goal_pct",
    "per",
    "defensive_rating",
    "db_age",
]

NBA_TEAM_IDS = {
    "ATL": "1610612737",
    "BOS": "1610612738",
    "BKN": "1610612751",
    "CHA": "1610612766",
    "CHI": "1610612741",
    "CLE": "1610612739",
    "DAL": "1610612742",
    "DEN": "1610612743",
    "DET": "1610612765",
    "GSW": "1610612744",
    "HOU": "1610612745",
    "IND": "1610612754",
    "LAC": "1610612746",
    "LAL": "1610612747",
    "MEM": "1610612763",
    "MIA": "1610612748",
    "MIL": "1610612749",
    "MIN": "1610612750",
    "NOP": "1610612740",
    "NYK": "1610612752",
    "OKC": "1610612760",
    "ORL": "1610612753",
    "PHI": "1610612755",
    "PHX": "1610612756",
    "POR": "1610612757",
    "SAC": "1610612758",
    "SAS": "1610612759",
    "TOR": "1610612761",
    "UTA": "1610612762",
    "WAS": "1610612764",
}

TEAM_CONFERENCES = {
    "ATL": "East",
    "BOS": "East",
    "BKN": "East",
    "CHA": "East",
    "CHI": "East",
    "CLE": "East",
    "DET": "East",
    "IND": "East",
    "MIA": "East",
    "MIL": "East",
    "NYK": "East",
    "ORL": "East",
    "PHI": "East",
    "TOR": "East",
    "WAS": "East",
    "DAL": "West",
    "DEN": "West",
    "GSW": "West",
    "HOU": "West",
    "LAC": "West",
    "LAL": "West",
    "MEM": "West",
    "MIN": "West",
    "NOP": "West",
    "OKC": "West",
    "PHX": "West",
    "POR": "West",
    "SAC": "West",
    "SAS": "West",
    "UTA": "West",
}

SALARY_NORMALIZED_ALIASES = {
    "adamaalphabal": ["adamabal"],
    "cameronwhitmore": ["camwhitmore"],
    "carltoncarrington": ["bubcarrington"],
    "dennisschroeder": ["dennisschroder"],
    "herbjones": ["herbertjones"],
    "nicolasclaxton": ["nicclaxton"],
    "ronholland": ["ronaldholland"],
    "santiagoaldama": ["santialdama"],
    "sviatoslavmykhailiuk": ["svimykhailiuk"],
    "yanicniederhauser": ["yanickonanniederhauser"],
}

GAMES_PER_TEAM = 82.0
WIN_PCT_ACCOUNTING_BLEND = 0.35
TRADE_IMPACT_POSITIVE_MULTIPLIER = 1.18
TRADE_IMPACT_NEGATIVE_MULTIPLIER = 0.72
TRADE_IMPACT_WINS_PER_VALUE = 1.55
TRADE_IMPACT_RATING_PER_OFF_VALUE = 0.70
TRADE_IMPACT_RATING_PER_DEF_VALUE = 0.62
TRADE_IMPACT_MAX_WINS_DELTA = 18.0
TRADE_IMPACT_MAX_RATING_DELTA = 8.0


def load_pickle(path: Path):
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)


def normalize_name(text: str) -> str:
    fixed = repair_text(text)
    fixed = unicodedata.normalize("NFKD", fixed)
    fixed = "".join(char for char in fixed if unicodedata.category(char) != "Mn")
    fixed = fixed.lower()
    return "".join(char for char in fixed if char.isalnum())


def roster_name_keys(text: str) -> set[str]:
    key = normalize_name(text)
    if not key:
        return set()
    keys = {key}
    for suffix in ("iii", "iv", "jr", "sr", "ii", "v"):
        if key.endswith(suffix) and len(key) > len(suffix) + 4:
            keys.add(key[: -len(suffix)])
            break
    return keys


def repair_text(text: str) -> str:
    fixed = str(text).strip()
    try:
        fixed = fixed.encode("latin-1").decode("utf-8")
    except Exception:
        pass
    return fixed.strip()


def to_float(value) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def to_optional_float(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def to_optional_int(value) -> int | None:
    num = to_optional_float(value)
    if num is None:
        return None
    return int(round(num))


def compute_losses(row: pd.Series | dict) -> float | None:
    wins = to_float(row["wins"]) if "wins" in row else None
    losses = to_float(row["losses"]) if "losses" in row else None
    games_played = to_float(row["games_played"]) if "games_played" in row else None
    win_pct = to_float(row["win_pct"]) if "win_pct" in row else None
    if losses is not None and losses > 0:
        return losses
    if wins is not None and games_played is not None and games_played >= wins:
        return float(games_played - wins)
    if wins is not None and win_pct is not None and win_pct > 0:
        estimated_games = round(wins / win_pct)
        if estimated_games >= wins:
            return float(estimated_games - wins)
    if wins is not None:
        return float(82.0 - wins)
    return losses


class SiteMetrics:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.data = self._load()

    def _load(self) -> dict[str, int]:
        defaults = {"total_visits": 0, "trade_simulations": 0}
        if not self.path.exists():
            return defaults
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return defaults
        return {
            "total_visits": int(loaded.get("total_visits") or 0),
            "trade_simulations": int(loaded.get("trade_simulations") or 0),
        }

    def _write_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(self.path)

    def snapshot(self) -> dict[str, int]:
        with self.lock:
            return dict(self.data)

    def increment_visit(self) -> dict[str, int]:
        with self.lock:
            self.data["total_visits"] += 1
            self._write_locked()
            return dict(self.data)

    def increment_trade_simulation(self) -> dict[str, int]:
        with self.lock:
            self.data["trade_simulations"] += 1
            self._write_locked()
            return dict(self.data)


class AppState:
    def __init__(self) -> None:
        if USE_PUBLIC_RUNTIME_DATA:
            missing_public_paths = [
                path for path in (PUBLIC_PLAYER_DATA_PATH, PUBLIC_TEAM_DATA_PATH, PUBLIC_WEIGHTED_PLAYER_PATH)
                if not path.exists()
            ]
            if missing_public_paths:
                missing = ", ".join(str(path) for path in missing_public_paths)
                raise FileNotFoundError(f"Public runtime data is incomplete: {missing}")
        player_source_columns = sorted(set(RUNTIME_PLAYER_SOURCE_COLUMNS + PLAYER_STATS_TABLE_COLUMNS))
        team_source_columns = sorted(set(RUNTIME_TEAM_SOURCE_COLUMNS + TEAM_STATS_TABLE_COLUMNS))
        self.players = load_players(PLAYER_DATA_PATH, usecols=player_source_columns)
        self.teams = load_teams(TEAM_DATA_PATH, usecols=team_source_columns)
        self.model_players = self._load_or_build_model_players()
        self.current_players = self.model_players[self.model_players["season_end_year"] == CURRENT_YEAR].copy()
        self.current_teams = self.teams[self.teams["season_end_year"] == CURRENT_YEAR].copy()
        self.team_lookup = {
            row["team_abbreviation"]: row
            for _, row in self.current_teams.iterrows()
        }
        self.roster_lookup = {
            team: frame.reset_index(drop=True)
            for team, frame in self.current_players.groupby("team_abbreviation", dropna=False)
        }
        self.player_lookup = {
            row["player_name"]: row
            for _, row in self.current_players.iterrows()
        }
        self.player_impact_baselines = self._build_player_impact_baselines()
        self.avatar_lookup = self._load_avatar_lookup()
        self.player_name_lookup = self._load_player_name_lookup()
        self.player_advanced_lookup = self._load_player_advanced_lookup()
        self.player_salary_lookup = self._load_player_salary_lookup()
        self.end_roster_team_lookup = self._load_end_roster_team_lookup()
        self.end_roster_player_team_lookup = self._build_end_roster_player_team_lookup()
        self.current_team_override_lookup = self._load_current_team_override_lookup()
        self.default_outgoing_lookup = self._build_default_outgoing_lookup()

        self.player_transfer_bundle = load_pickle(CODE_DIR / "outputs" / "player_transfer_model_v2" / "preprocessing.pkl")
        self.player_retention_bundle = load_pickle(CODE_DIR / "outputs" / "player_retention_model_v1" / "preprocessing.pkl")
        self.team_bundle = load_pickle(CODE_DIR / "outputs" / "team_offseason_update_model_v2" / "preprocessing.pkl")

        self.player_transfer_model = self._load_model(
            CODE_DIR / "outputs" / "player_transfer_model_v2" / "weights.npz",
        )
        self.player_retention_model = self._load_model(
            CODE_DIR / "outputs" / "player_retention_model_v1" / "weights.npz",
        )
        self.team_model = self._load_model(
            CODE_DIR / "outputs" / "team_offseason_update_model_v2" / "weights.npz",
        )
        self.team_win_regularization_offset = self._compute_team_win_regularization_offset()
        self.team_net_rating_regularization_offset = self._compute_team_net_rating_regularization_offset()
        self.team_win_pct_regularization_offset = self._compute_team_win_pct_regularization_offset()

    def _load_avatar_lookup(self) -> dict[str, str]:
        lookup: dict[str, str] = {}
        if not AVATAR_INDEX_PATH.exists():
            return lookup
        with open(AVATAR_INDEX_PATH, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_name = (row.get("file_name") or "").strip()
                if not file_name:
                    continue
                player_name = (row.get("player_name") or "").strip()
                if not player_name:
                    continue
                lookup[normalize_name(player_name)] = f"/assets/statmuse_avatars_2025_26/{file_name}"
        return lookup

    def _load_model(self, path: Path) -> NumpyTabularResNet:
        return NumpyTabularResNet(path)

    def _load_or_build_model_players(self) -> pd.DataFrame:
        if PUBLIC_WEIGHTED_PLAYER_PATH.exists():
            return load_players(PUBLIC_WEIGHTED_PLAYER_PATH)
        if USE_PUBLIC_RUNTIME_DATA:
            raise FileNotFoundError(
                f"Public runtime data is incomplete: missing {PUBLIC_WEIGHTED_PLAYER_PATH}. "
                "Regenerate runtime_data before running the public build."
            )
        if PRIVATE_WEIGHTED_PLAYER_CACHE_PATH.exists() and PRIVATE_WEIGHTED_PLAYER_CACHE_PATH.stat().st_mtime >= PLAYER_DATA_PATH.stat().st_mtime:
            return load_players(PRIVATE_WEIGHTED_PLAYER_CACHE_PATH)
        player_history_columns = sorted(set(PLAYER_INPUT_COLUMNS + PLAYER_TARGET_COLUMNS + PLAYER_GROUP_COLUMNS))
        model_players = build_recent_weighted_player_stats(self.players, numeric_columns=player_history_columns)
        model_players = filter_effective_rotation_players(model_players)
        PRIVATE_WEIGHTED_PLAYER_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        model_players.to_csv(PRIVATE_WEIGHTED_PLAYER_CACHE_PATH, index=False, encoding="utf-8-sig")
        return model_players

    def _load_player_name_lookup(self) -> dict[str, dict[str, str]]:
        lookup: dict[str, dict[str, str]] = {}
        if not PLAYER_NAME_MAP_PATH.exists():
            return lookup
        with open(PLAYER_NAME_MAP_PATH, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                player_name = (row.get("player_name") or "").strip()
                if not player_name:
                    continue
                key = normalize_name(player_name)
                player_name_fixed = repair_text(row.get("player_name_fixed") or player_name)
                player_name_zh = (row.get("player_name_zh") or "").strip()
                lookup[key] = {
                    "player_name_en": player_name_fixed,
                    "player_name_zh": player_name_zh,
                    "display_name": player_name_zh or player_name_fixed,
                }
        return lookup

    def _load_player_advanced_lookup(self) -> dict[tuple[str, str], dict[str, str]]:
        lookup: dict[tuple[str, str], dict[str, str]] = {}
        if not PLAYER_ADVANCED_IMPACT_PATH.exists():
            return lookup
        with open(PLAYER_ADVANCED_IMPACT_PATH, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                player_name = repair_text(row.get("player_name") or "")
                team = (row.get("team_abbreviation") or "").strip().upper()
                if not player_name:
                    continue
                name_key = normalize_name(player_name)
                if team:
                    lookup[(name_key, team)] = row
                lookup.setdefault((name_key, ""), row)
        return lookup

    def _load_player_salary_lookup(self) -> dict[str, dict[str, str]]:
        lookup: dict[str, dict[str, str]] = {}
        if not PLAYER_SALARY_PATH.exists():
            return lookup
        with open(PLAYER_SALARY_PATH, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                player_name = repair_text(row.get("player_name") or "")
                key = (row.get("normalized_name") or "").strip() or normalize_name(player_name)
                if not key:
                    continue
                current_salary = to_optional_int(row.get("salary_2025_26")) or 0
                keys = set(roster_name_keys(player_name))
                keys.add(key)
                keys.update(SALARY_NORMALIZED_ALIASES.get(key, []))
                for candidate_key in keys:
                    existing_salary = to_optional_int(lookup.get(candidate_key, {}).get("salary_2025_26")) or -1
                    if current_salary >= existing_salary:
                        lookup[candidate_key] = row
        return lookup

    def _load_current_team_override_lookup(self) -> dict[str, dict[str, str]]:
        lookup: dict[str, dict[str, str]] = {}
        if not CURRENT_TEAM_OVERRIDES_PATH.exists():
            return lookup
        with open(CURRENT_TEAM_OVERRIDES_PATH, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                player_name = repair_text(row.get("player_name") or "")
                current_team = (row.get("current_team") or "").strip().upper()
                if not player_name or not current_team:
                    continue
                lookup[normalize_name(player_name)] = {
                    "player_name": player_name,
                    "current_team": current_team,
                    "transaction_date": (row.get("transaction_date") or "").strip(),
                    "source": (row.get("source") or "").strip(),
                    "notes": (row.get("notes") or "").strip(),
                }
        return lookup

    def _load_end_roster_team_lookup(self) -> dict[str, dict[str, dict[str, str]]]:
        lookup: dict[str, dict[str, dict[str, str]]] = {}
        if not END_ROSTER_PATH.exists():
            return lookup
        with open(END_ROSTER_PATH, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                team = (row.get("team_abbreviation") or "").strip().upper()
                player_name = repair_text(row.get("player_name") or "")
                if not team or not player_name:
                    continue
                payload = {
                    "player_name": player_name,
                    "team_abbreviation": team,
                    "team_name": (row.get("team_name") or "").strip(),
                    "position": (row.get("position") or "").strip(),
                    "status": (row.get("status") or "").strip(),
                    "source": (row.get("source") or "").strip(),
                    "source_url": (row.get("source_url") or "").strip(),
                }
                team_lookup = lookup.setdefault(team, {})
                for key in roster_name_keys(player_name):
                    team_lookup[key] = payload
        return lookup

    def _build_end_roster_player_team_lookup(self) -> dict[str, list[dict[str, str]]]:
        lookup: dict[str, list[dict[str, str]]] = {}
        for team_lookup in self.end_roster_team_lookup.values():
            for key, payload in team_lookup.items():
                teams = lookup.setdefault(key, [])
                if not any(item["team_abbreviation"] == payload["team_abbreviation"] for item in teams):
                    teams.append(payload)
        return lookup

    def _build_default_outgoing_lookup(self) -> dict[str, list[dict[str, str]]]:
        lookup: dict[str, list[dict[str, str]]] = {}
        if not self.end_roster_team_lookup and not self.current_team_override_lookup:
            return lookup
        for team, frame in self.roster_lookup.items():
            if frame.empty:
                continue
            candidates: list[dict[str, str]] = []
            roster = frame.sort_values(["points_per_game", "minutes_per_game"], ascending=[False, False])
            end_roster_for_team = self.end_roster_team_lookup.get(team)
            for _, row in roster.iterrows():
                player_name = str(row["player_name"])
                player_keys = roster_name_keys(player_name)
                if end_roster_for_team is not None:
                    if any(key in end_roster_for_team for key in player_keys):
                        continue
                    end_roster_matches: list[dict[str, str]] = []
                    for key in player_keys:
                        for payload in self.end_roster_player_team_lookup.get(key, []):
                            if payload["team_abbreviation"] != team and not any(
                                item["team_abbreviation"] == payload["team_abbreviation"] for item in end_roster_matches
                            ):
                                end_roster_matches.append(payload)
                    if end_roster_matches:
                        current_team = "/".join(sorted(item["team_abbreviation"] for item in end_roster_matches))
                        source = end_roster_matches[0].get("source", "")
                        source_url = end_roster_matches[0].get("source_url", "")
                        notes = "Not on selected team's ESPN end roster; present on another ESPN end roster."
                    else:
                        current_team = "FA"
                        source = "ESPN team roster API"
                        source_url = ""
                        notes = "Not on selected team's ESPN end roster; not found on another ESPN end roster."
                    name_fields = self.player_names_for(player_name)
                    candidate = {
                        "player_name": player_name,
                        "player_name_en": name_fields["player_name_en"],
                        "player_name_zh": name_fields["player_name_zh"],
                        "display_name": name_fields["display_name"],
                        "listed_team": team,
                        "current_team": current_team,
                        "transaction_date": "",
                        "source": source,
                        "source_url": source_url,
                        "notes": notes,
                    }
                    candidate.update(self.player_salary_for(player_name))
                    candidates.append(candidate)
                    continue
                override = self.current_team_override_lookup.get(normalize_name(player_name))
                if not override:
                    continue
                current_team = override["current_team"]
                if current_team == team:
                    continue
                name_fields = self.player_names_for(player_name)
                candidate = {
                    "player_name": player_name,
                    "player_name_en": name_fields["player_name_en"],
                    "player_name_zh": name_fields["player_name_zh"],
                    "display_name": name_fields["display_name"],
                    "listed_team": team,
                    "current_team": current_team,
                    "transaction_date": override.get("transaction_date", ""),
                    "source": override.get("source", ""),
                    "notes": override.get("notes", ""),
                }
                candidate.update(self.player_salary_for(player_name))
                candidates.append(candidate)
            if candidates:
                lookup[team] = candidates
        return lookup

    def avatar_url_for(self, player_name: str) -> str | None:
        return self.avatar_lookup.get(normalize_name(player_name))

    def team_logo_url_for(self, team_abbreviation: str) -> str | None:
        team = str(team_abbreviation).strip().upper()
        local_logo = TEAM_LOGO_DIR / f"{team}.svg"
        if local_logo.exists():
            return f"/assets/team_logos_2025_26/{team}.svg"
        team_id = NBA_TEAM_IDS.get(team)
        if not team_id:
            return None
        return f"https://cdn.nba.com/logos/nba/{team_id}/primary/L/logo.svg"

    def player_names_for(self, player_name: str) -> dict[str, str]:
        english_name = repair_text(player_name)
        key = normalize_name(player_name)
        payload = self.player_name_lookup.get(key)
        if payload:
            return payload
        return {
            "player_name_en": english_name,
            "player_name_zh": "",
            "display_name": english_name,
        }

    def player_salary_for(self, player_name: str) -> dict:
        row = None
        for key in roster_name_keys(player_name):
            row = self.player_salary_lookup.get(key)
            if row:
                break
        current_salary = to_optional_int(row.get("salary_2025_26")) if row else None
        next_salary = to_optional_int(row.get("salary_2026_27")) if row else None
        return {
            "salary_2025_26": current_salary,
            "salary_2026_27": next_salary,
            "current_salary": current_salary,
            "next_salary": next_salary,
            "salary_next_defaulted": (str(row.get("next_salary_defaulted") or "").lower() == "true") if row else False,
            "salary_source_url": (row.get("source_url") or "") if row else "",
            "salary_player_id": (row.get("hoopshype_player_id") or "") if row else "",
        }

    def _build_player_impact_baselines(self) -> dict[str, float]:
        baselines: dict[str, float] = {}
        for column, default in [
            ("points_per_game", 8.0),
            ("per", 13.0),
            ("true_shooting_pct", 57.0),
            ("defensive_rating", 116.0),
        ]:
            if column in self.current_players.columns:
                values = pd.to_numeric(self.current_players[column], errors="coerce").dropna()
                baselines[column] = float(values.median()) if not values.empty else default
            else:
                baselines[column] = default
        return baselines

    def _serialize_team(self, row: pd.Series) -> dict:
        data = {
            "team_abbreviation": row["team_abbreviation"],
            "team_name": row["team_name"],
            "season": CURRENT_SEASON_LABEL,
        }
        for column in TEAM_DISPLAY_COLUMNS:
            if column == "losses":
                data[column] = compute_losses(row)
                continue
            data[column] = to_float(row[column]) if column in row else None
        return data

    def _serialize_player_row(self, row: pd.Series, projection_type: str = "current", source_team: str | None = None) -> dict:
        name_fields = self.player_names_for(str(row["player_name"]))
        payload = {
            "player_name": row["player_name"],
            "player_name_en": name_fields["player_name_en"],
            "player_name_zh": name_fields["player_name_zh"],
            "display_name": name_fields["display_name"],
            "team_abbreviation": row.get("team_abbreviation"),
            "team_name": row.get("team_name"),
            "avatar_url": self.avatar_url_for(str(row["player_name"])),
            "projection_type": projection_type,
            "source_team": source_team,
        }
        payload.update(self.player_salary_for(str(row["player_name"])))
        for column in PLAYER_DISPLAY_COLUMNS:
            payload[column] = to_float(row[column]) if column in row else None
        return payload

    def _serialize_projected_player(self, player_name: str, source_team: str, projection_type: str, stats: dict) -> dict:
        name_fields = self.player_names_for(player_name)
        payload = {
            "player_name": player_name,
            "player_name_en": name_fields["player_name_en"],
            "player_name_zh": name_fields["player_name_zh"],
            "display_name": name_fields["display_name"],
            "team_abbreviation": None,
            "team_name": None,
            "avatar_url": self.avatar_url_for(player_name),
            "projection_type": projection_type,
            "source_team": source_team,
        }
        payload.update(self.player_salary_for(player_name))
        for column in PLAYER_DISPLAY_COLUMNS:
            payload[column] = float(stats[column]) if column in stats and stats[column] is not None else None
        return payload

    def _serialize_player_stats_row(self, row: pd.Series) -> dict:
        player_name = str(row["player_name"])
        name_fields = self.player_names_for(player_name)
        payload = {
            "player_name": player_name,
            "player_name_en": name_fields["player_name_en"],
            "player_name_zh": name_fields["player_name_zh"],
            "display_name": name_fields["display_name"],
            "team_abbreviation": row.get("team_abbreviation"),
            "team_name": row.get("team_name"),
            "avatar_url": self.avatar_url_for(player_name),
            "season": CURRENT_SEASON_LABEL,
        }
        payload.update(self.player_salary_for(player_name))
        for column in PLAYER_STATS_TABLE_COLUMNS:
            payload[column] = to_float(row[column]) if column in row else None
        impact = self.player_advanced_lookup.get((normalize_name(player_name), str(row.get("team_abbreviation") or "").upper()))
        impact = impact or self.player_advanced_lookup.get((normalize_name(player_name), ""))
        for column in PLAYER_IMPACT_TABLE_COLUMNS:
            payload[column] = to_optional_float(impact.get(column)) if impact else None
        for column in PLAYER_IMPACT_SOURCE_COLUMNS:
            payload[column] = (impact.get(column) or "") if impact else ""
        return payload

    def _serialize_team_stats_row(self, row: pd.Series) -> dict:
        team = str(row["team_abbreviation"])
        payload = {
            "team_abbreviation": team,
            "team_name": row["team_name"],
            "conference": TEAM_CONFERENCES.get(team, "Unknown"),
            "logo_url": self.team_logo_url_for(team),
            "season": CURRENT_SEASON_LABEL,
        }
        for column in TEAM_STATS_TABLE_COLUMNS:
            if column == "losses":
                payload[column] = compute_losses(row)
                continue
            payload[column] = to_float(row[column]) if column in row else None
        return payload

    def home_payload(self) -> dict:
        top = self.current_players.sort_values(
            ["points_per_game", "minutes_per_game"],
            ascending=[False, False],
        ).drop_duplicates("player_name").head(18)
        return {
            "project_name": "纸上谈球",
            "season_label": CURRENT_SEASON_LABEL,
            "next_season_label": NEXT_SEASON_LABEL,
            "headline_players": [
                {
                    "player_name": self.player_names_for(str(row["player_name"]))["display_name"],
                    "player_name_en": self.player_names_for(str(row["player_name"]))["player_name_en"],
                    "player_name_zh": self.player_names_for(str(row["player_name"]))["player_name_zh"],
                    "display_name": self.player_names_for(str(row["player_name"]))["display_name"],
                    "team_abbreviation": row["team_abbreviation"],
                    "points_per_game": to_float(row["points_per_game"]),
                    "avatar_url": self.avatar_url_for(str(row["player_name"])),
                }
                for _, row in top.iterrows()
            ],
        }

    def list_teams(self):
        teams = self.current_teams[["team_abbreviation", "team_name", "wins", "win_pct", "net_rating"]].drop_duplicates().sort_values(
            ["wins", "team_name"], ascending=[False, True]
        )
        return [
            {
                "team_abbreviation": row["team_abbreviation"],
                "team_name": row["team_name"],
                "wins": to_float(row["wins"]),
                "win_pct": to_float(row["win_pct"]),
                "net_rating": to_float(row["net_rating"]),
            }
            for _, row in teams.iterrows()
        ]

    def player_stats_table(self) -> dict:
        players = self.players[self.players["season_end_year"] == CURRENT_YEAR].copy()
        if players.empty:
            return {"season": CURRENT_SEASON_LABEL, "players": []}
        sort_columns = [column for column in ["points_per_game", "minutes_per_game", "games_played"] if column in players.columns]
        if sort_columns:
            players = players.sort_values(sort_columns, ascending=[False] * len(sort_columns))
        return {
            "season": CURRENT_SEASON_LABEL,
            "players": [self._serialize_player_stats_row(row) for _, row in players.iterrows()],
        }

    def team_stats_table(self) -> dict:
        teams = self.current_teams.copy()
        if teams.empty:
            return {"season": CURRENT_SEASON_LABEL, "conferences": {"East": [], "West": []}, "teams": []}
        teams = teams.sort_values(["wins", "net_rating", "team_name"], ascending=[False, False, True])
        serialized = [self._serialize_team_stats_row(row) for _, row in teams.iterrows()]
        conferences = {
            "East": [row for row in serialized if row["conference"] == "East"],
            "West": [row for row in serialized if row["conference"] == "West"],
        }
        return {
            "season": CURRENT_SEASON_LABEL,
            "conferences": conferences,
            "teams": serialized,
        }

    def team_roster(self, team: str):
        roster = self.roster_lookup.get(team, pd.DataFrame()).copy()
        if roster.empty:
            return []
        roster = roster.sort_values(["points_per_game", "minutes_per_game"], ascending=[False, False])
        return [self._serialize_player_row(row) for _, row in roster.iterrows()]

    def team_view(self, team: str) -> dict:
        if team not in self.team_lookup:
            return {"team": None, "players": [], "default_outgoing": [], "default_outgoing_details": []}
        default_outgoing_details = self.default_outgoing_lookup.get(team, [])
        return {
            "team": self._serialize_team(self.team_lookup[team]),
            "players": self.team_roster(team),
            "default_outgoing": [item["player_name"] for item in default_outgoing_details],
            "default_outgoing_details": default_outgoing_details,
        }

    def players_by_team(self, team: str, exclude_team: str):
        if not team or team == exclude_team:
            return []
        roster = self.roster_lookup.get(team, pd.DataFrame()).copy()
        if roster.empty:
            return []
        roster = roster.sort_values(["points_per_game", "minutes_per_game"], ascending=[False, False])
        return [self._serialize_player_row(row) for _, row in roster.iterrows()]

    def _transform_predict(self, row_df: pd.DataFrame, bundle: dict, model: NumpyTabularResNet) -> np.ndarray:
        features = bundle["feature_columns"]
        row_df = row_df.reindex(columns=features, fill_value=np.nan)
        x = bundle["transformers"]["x_scaler"].transform(
            bundle["transformers"]["imputer"].transform(row_df)
        ).astype("float32")
        pred_scaled = model.predict(x)
        return bundle["transformers"]["y_scaler"].inverse_transform(pred_scaled)

    def _apply_team_win_regularization(self, result: dict) -> dict:
        raw_win_pct = result.get("win_pct")
        wins = float(result["wins"] + self.team_win_regularization_offset)
        wins = float(np.clip(wins, 0.0, GAMES_PER_TEAM))
        result["wins"] = wins
        result["losses"] = float(GAMES_PER_TEAM - wins)
        accounting_pct = float(wins / GAMES_PER_TEAM)
        if raw_win_pct is None:
            result["win_pct"] = accounting_pct
        else:
            blended = (1.0 - WIN_PCT_ACCOUNTING_BLEND) * float(raw_win_pct) + WIN_PCT_ACCOUNTING_BLEND * accounting_pct
            result["win_pct"] = float(np.clip(blended, 0.0, 1.0))
        return result

    def _apply_team_net_rating_regularization(self, result: dict) -> dict:
        offset = float(self.team_net_rating_regularization_offset)
        if "offensive_rating" in result and "defensive_rating" in result:
            result["offensive_rating"] = float(result["offensive_rating"] + offset / 2.0)
            result["defensive_rating"] = float(result["defensive_rating"] - offset / 2.0)
            result["net_rating"] = float(result["offensive_rating"] - result["defensive_rating"])
        elif "net_rating" in result:
            result["net_rating"] = float(result["net_rating"] + offset)
        return result

    def _apply_team_win_pct_regularization(self, result: dict) -> dict:
        result["win_pct"] = float(np.clip(result["win_pct"] + self.team_win_pct_regularization_offset, 0.0, 1.0))
        return result

    def _player_projected_minutes(self, row: pd.Series) -> float:
        minutes = to_float(row["minutes"]) if "minutes" in row else None
        if minutes is not None and minutes > 0.0:
            return minutes
        games = to_float(row["games_played"]) if "games_played" in row else None
        mpg = to_float(row["minutes_per_game"]) if "minutes_per_game" in row else None
        if games is not None and mpg is not None and games > 0.0 and mpg > 0.0:
            return float(games * mpg)
        return 0.0

    def _player_trade_components(self, row: pd.Series) -> dict[str, float]:
        games = to_float(row["games_played"]) if "games_played" in row else None
        mpg = to_float(row["minutes_per_game"]) if "minutes_per_game" in row else None
        minutes = self._player_projected_minutes(row)
        if mpg is None and games and games > 0.0:
            mpg = minutes / games
        games = games or 0.0
        mpg = mpg or 0.0

        availability = float(np.clip(games / 65.0, 0.0, 1.08))
        minute_load = float(np.clip(mpg / 32.0, 0.0, 1.15))
        role_weight = float(np.sqrt(availability) * minute_load)
        if minutes <= 0.0:
            role_weight = 0.0

        ppg = to_float(row["points_per_game"]) if "points_per_game" in row else None
        per = to_float(row["per"]) if "per" in row else None
        ts = to_float(row["true_shooting_pct"]) if "true_shooting_pct" in row else None
        off_dpm = to_float(row["db_o_dpm"]) if "db_o_dpm" in row else None
        def_dpm = to_float(row["db_dpm"]) if "db_dpm" in row else None
        defensive_rating = to_float(row["defensive_rating"]) if "defensive_rating" in row else None

        offense = 0.0
        if ppg is not None:
            offense += 0.22 * (ppg - self.player_impact_baselines["points_per_game"])
        if per is not None:
            offense += 0.10 * (per - self.player_impact_baselines["per"])
        if ts is not None:
            offense += 0.06 * (ts - self.player_impact_baselines["true_shooting_pct"])
        if off_dpm is not None:
            offense += 0.45 * off_dpm

        defense = 0.0
        if def_dpm is not None:
            defense += 0.45 * def_dpm
        if defensive_rating is not None:
            defense += 0.035 * (self.player_impact_baselines["defensive_rating"] - defensive_rating)

        return {
            "offense": float(role_weight * offense),
            "defense": float(role_weight * defense),
            "total": float(role_weight * (offense + defense)),
            "role_weight": role_weight,
        }

    def _sum_trade_components(self, names: list[str]) -> dict[str, float]:
        total = {"offense": 0.0, "defense": 0.0, "total": 0.0, "role_weight": 0.0}
        for name in names:
            player = self.player_lookup.get(name)
            if player is None:
                continue
            components = self._player_trade_components(player)
            for key in total:
                total[key] += components[key]
        return total

    def _compute_trade_impact_adjustment(self, outgoing_names: list[str], incoming_names: list[str]) -> dict[str, float]:
        incoming = self._sum_trade_components(incoming_names)
        outgoing = self._sum_trade_components(outgoing_names)
        offense_delta = incoming["offense"] - outgoing["offense"]
        defense_delta = incoming["defense"] - outgoing["defense"]
        total_delta = offense_delta + defense_delta
        multiplier = TRADE_IMPACT_POSITIVE_MULTIPLIER if total_delta >= 0.0 else TRADE_IMPACT_NEGATIVE_MULTIPLIER

        wins_delta = float(np.clip(
            total_delta * multiplier * TRADE_IMPACT_WINS_PER_VALUE,
            -TRADE_IMPACT_MAX_WINS_DELTA,
            TRADE_IMPACT_MAX_WINS_DELTA,
        ))
        offense_rating_delta = float(np.clip(
            offense_delta * multiplier * TRADE_IMPACT_RATING_PER_OFF_VALUE,
            -TRADE_IMPACT_MAX_RATING_DELTA,
            TRADE_IMPACT_MAX_RATING_DELTA,
        ))
        defense_rating_delta = float(np.clip(
            -defense_delta * multiplier * TRADE_IMPACT_RATING_PER_DEF_VALUE,
            -TRADE_IMPACT_MAX_RATING_DELTA,
            TRADE_IMPACT_MAX_RATING_DELTA,
        ))
        return {
            "wins_delta": wins_delta,
            "offensive_rating_delta": offense_rating_delta,
            "defensive_rating_delta": defense_rating_delta,
            "value_delta": float(total_delta),
            "offense_value_delta": float(offense_delta),
            "defense_value_delta": float(defense_delta),
        }

    def _apply_trade_impact_adjustment(self, result: dict, outgoing_names: list[str], incoming_names: list[str]) -> dict:
        if not outgoing_names and not incoming_names:
            return result
        impact = self._compute_trade_impact_adjustment(outgoing_names, incoming_names)
        result["wins"] = float(np.clip(result["wins"] + impact["wins_delta"], 0.0, GAMES_PER_TEAM))
        if "win_pct" in result:
            result["win_pct"] = float(np.clip(result["win_pct"] + impact["wins_delta"] / GAMES_PER_TEAM, 0.0, 1.0))
        result["offensive_rating"] = float(result["offensive_rating"] + impact["offensive_rating_delta"])
        result["defensive_rating"] = float(result["defensive_rating"] + impact["defensive_rating_delta"])
        result["net_rating"] = float(result["offensive_rating"] - result["defensive_rating"])
        result["trade_impact_adjustment"] = impact
        return result

    def _apply_team_league_regularization(self, result: dict) -> dict:
        result = self._apply_team_net_rating_regularization(result)
        result = self._apply_team_win_regularization(result)
        result = self._apply_team_win_pct_regularization(result)
        return result

    def _restore_team_prediction(self, delta: np.ndarray, row_df: pd.DataFrame) -> dict:
        direct_columns = self.team_bundle.get("direct_target_columns")
        if not direct_columns:
            direct_columns = [
                column[len("target_delta_"):] if column.startswith("target_delta_") else column
                for column in self.team_bundle["target_columns"]
            ]
        row = row_df.iloc[0]
        result = {}
        for idx, col in enumerate(direct_columns):
            result[col] = float(delta[0, idx] + row[f"prev_team_{col}"])
        if "offensive_rating" in result and "defensive_rating" in result:
            result["net_rating"] = float(result["offensive_rating"] - result["defensive_rating"])
        elif "net_rating" not in result:
            result["net_rating"] = float(row["prev_team_net_rating"])
        return result

    def _compute_team_win_regularization_offset(self) -> float:
        if not self.team_lookup:
            return 0.0
        raw_wins = []
        for team in self.team_lookup.keys():
            row_df = self._build_team_sim_row(team, [], [])
            delta = self._transform_predict(row_df, self.team_bundle, self.team_model)
            predicted_wins = self._restore_team_prediction(delta, row_df)["wins"]
            raw_wins.append(predicted_wins)
        expected_total_wins = len(raw_wins) * GAMES_PER_TEAM / 2.0
        predicted_total_wins = float(sum(raw_wins))
        return float((expected_total_wins - predicted_total_wins) / len(raw_wins))

    def _compute_team_net_rating_regularization_offset(self) -> float:
        if not self.team_lookup:
            return 0.0
        raw_net_ratings = []
        for team in self.team_lookup.keys():
            row_df = self._build_team_sim_row(team, [], [])
            delta = self._transform_predict(row_df, self.team_bundle, self.team_model)
            raw_net_ratings.append(self._restore_team_prediction(delta, row_df)["net_rating"])
        return float(-np.mean(raw_net_ratings))

    def _compute_team_win_pct_regularization_offset(self) -> float:
        if not self.team_lookup:
            return 0.0
        win_pcts = []
        for team in self.team_lookup.keys():
            row_df = self._build_team_sim_row(team, [], [])
            delta = self._transform_predict(row_df, self.team_bundle, self.team_model)
            result = self._restore_team_prediction(delta, row_df)
            result = self._apply_team_win_regularization(result)
            win_pcts.append(result["win_pct"])
        expected_total_win_pct = len(win_pcts) / 2.0
        predicted_total_win_pct = float(sum(win_pcts))
        return float((expected_total_win_pct - predicted_total_win_pct) / len(win_pcts))

    def _build_team_sim_row(self, team: str, outgoing_names: list[str], incoming_names: list[str]) -> pd.DataFrame:
        prev_team = self.team_lookup[team]
        prev_roster = self.roster_lookup.get(team, pd.DataFrame()).copy()
        outgoing = prev_roster[prev_roster["player_name"].isin(outgoing_names)].copy()
        retained = prev_roster[~prev_roster["player_name"].isin(outgoing_names)].copy()
        incoming_rows = [self.player_lookup[name] for name in incoming_names if name in self.player_lookup]
        incoming = pd.DataFrame(incoming_rows) if incoming_rows else pd.DataFrame(columns=self.current_players.columns)

        row = {
            "prev_year": CURRENT_YEAR,
            "next_year": NEXT_YEAR,
            "prev_year_numeric": float(CURRENT_YEAR),
            "next_year_numeric": float(NEXT_YEAR),
            "team_abbreviation": team,
            "prev_team_name": prev_team["team_name"],
            "next_team_name": prev_team["team_name"],
        }
        for col in TEAM_TARGET_COLUMNS_2:
            row[f"prev_team_{col}"] = float(prev_team[col])
            row[f"next_team_{col}"] = np.nan
        for col in self.current_teams.columns:
            if col in {"season", "team_abbreviation", "team_name", "season_end_year"}:
                continue
            row[f"prev_team_{col}"] = float(prev_team[col]) if pd.notna(prev_team[col]) else np.nan

        prev_names = set(prev_roster["player_name"].tolist())
        retained_names = set(retained["player_name"].tolist())
        incoming_names_set = set(incoming["player_name"].tolist())
        row["incoming_unknown_count"] = 0.0
        row["retained_ratio"] = float(len(retained_names) / len(prev_names)) if prev_names else np.nan
        row["incoming_ratio"] = float(len(incoming_names_set) / max(len(retained_names) + len(incoming_names_set), 1))
        row["outgoing_ratio"] = float(len(outgoing_names) / len(prev_names)) if prev_names else np.nan

        for prefix, group in [
            ("prev_roster", prev_roster),
            ("retained", retained),
            ("outgoing", outgoing),
            ("incoming", incoming),
        ]:
            row.update(aggregate_group(group, prefix))

        prev_total_minutes = row.get("prev_roster_total_minutes", np.nan)
        retained_total_minutes = row.get("retained_total_minutes", np.nan)
        row["retained_minutes_ratio"] = retained_total_minutes / prev_total_minutes if prev_total_minutes and not np.isnan(prev_total_minutes) else np.nan
        row["incoming_vs_outgoing_minutes"] = row.get("incoming_total_minutes", 0.0) - row.get("outgoing_total_minutes", 0.0)
        row["incoming_vs_outgoing_count"] = row.get("incoming_count", 0.0) - row.get("outgoing_count", 0.0)
        if "incoming_sum_points_per_game" in row and "outgoing_sum_points_per_game" in row:
            row["incoming_vs_outgoing_ppg"] = row["incoming_sum_points_per_game"] - row["outgoing_sum_points_per_game"]
        if "incoming_sum_assists_per_game" in row and "outgoing_sum_assists_per_game" in row:
            row["incoming_vs_outgoing_apg"] = row["incoming_sum_assists_per_game"] - row["outgoing_sum_assists_per_game"]
        if "incoming_sum_rebounds_per_game" in row and "outgoing_sum_rebounds_per_game" in row:
            row["incoming_vs_outgoing_rpg"] = row["incoming_sum_rebounds_per_game"] - row["outgoing_sum_rebounds_per_game"]
        row["base_weight"] = 1.0 + row.get("incoming_count", 0.0) + row.get("outgoing_count", 0.0)
        return pd.DataFrame([row])

    def _predict_team(self, team: str, outgoing_names: list[str], incoming_names: list[str]) -> dict:
        row_df = self._build_team_sim_row(team, outgoing_names, incoming_names)
        delta = self._transform_predict(row_df, self.team_bundle, self.team_model)
        result = self._restore_team_prediction(delta, row_df)
        result = self._apply_trade_impact_adjustment(result, outgoing_names, incoming_names)
        result = self._apply_team_league_regularization(result)
        result["team_abbreviation"] = team
        result["team_name"] = self.team_lookup[team]["team_name"]
        result["season"] = NEXT_SEASON_LABEL
        return result

    def _build_transfer_row(self, player_name: str, dest_team: str, outgoing_names: list[str], incoming_names: list[str]) -> pd.DataFrame:
        player = self.player_lookup[player_name]
        src_team = self.team_lookup[player["team_abbreviation"]]
        dst_team = self.team_lookup[dest_team]
        prev_roster = self.roster_lookup.get(dest_team, pd.DataFrame()).copy()
        outgoing = prev_roster[prev_roster["player_name"].isin(outgoing_names)].copy()
        incoming_rows = [self.player_lookup[name] for name in incoming_names if name in self.player_lookup]
        incoming = pd.DataFrame(incoming_rows) if incoming_rows else pd.DataFrame(columns=self.current_players.columns)
        roster_feat = build_team_roster_features(prev_roster, prefix="dst_prev")
        roster_row = roster_feat.iloc[0].to_dict() if len(roster_feat) else {}
        out_agg = aggregate_movement_features(outgoing.assign(next_year=NEXT_YEAR, source_team_abbreviation=dest_team), "source_team_abbreviation", "dest_outgoing")
        in_agg = aggregate_movement_features(incoming.assign(next_year=NEXT_YEAR, dest_team_abbreviation=dest_team), "dest_team_abbreviation", "dest_incoming")
        row = {
            "player_name": player_name,
            "prev_year": CURRENT_YEAR,
            "next_year": NEXT_YEAR,
            "prev_season": CURRENT_SEASON_LABEL,
            "next_season": NEXT_SEASON_LABEL,
            "source_team_abbreviation": player["team_abbreviation"],
            "dest_team_abbreviation": dest_team,
        }
        for col in PLAYER_INPUT_COLUMNS:
            if col in player.index:
                row[col] = float(player[col]) if pd.notna(player[col]) else np.nan
                row[f"{col}_prev"] = float(player[col]) if pd.notna(player[col]) else np.nan
        for col in TEAM_INPUT_COLUMNS:
            row[f"src_team_{col}"] = float(src_team[col]) if pd.notna(src_team[col]) else np.nan
            row[f"dst_prev_team_{col}"] = float(dst_team[col]) if pd.notna(dst_team[col]) else np.nan
        row.update({k: v for k, v in roster_row.items() if k not in {"season_end_year", "team_abbreviation"}})
        if len(out_agg):
            row.update({k: v for k, v in out_agg.iloc[0].to_dict().items() if k not in {"next_year", "team_abbreviation"}})
        if len(in_agg):
            row.update({k: v for k, v in in_agg.iloc[0].to_dict().items() if k not in {"next_year", "team_abbreviation"}})
        if pd.notna(player.get("db_age", np.nan)):
            row["projected_age_next"] = float(player["db_age"] + 1.0)
            row["projected_age_next_sq"] = row["projected_age_next"] ** 2
            row["projected_age_next_cu"] = row["projected_age_next"] ** 3
        if pd.notna(player.get("minutes", np.nan)) and pd.notna(src_team.get("minutes", np.nan)):
            row["prev_player_minute_share"] = float(player["minutes"] / src_team["minutes"]) if src_team["minutes"] else np.nan
        if pd.notna(player.get("points_per_game", np.nan)) and pd.notna(src_team.get("points_per_game", np.nan)):
            row["prev_player_ppg_share"] = float(player["points_per_game"] / src_team["points_per_game"]) if src_team["points_per_game"] else np.nan
        if pd.notna(player.get("assists_per_game", np.nan)) and pd.notna(src_team.get("assists_per_game", np.nan)):
            row["prev_player_apg_share"] = float(player["assists_per_game"] / src_team["assists_per_game"]) if src_team["assists_per_game"] else np.nan
        if "dest_outgoing_total_minutes" in row and "dst_prev_total_minutes" in row:
            row["dest_role_vacancy_share"] = float(row["dest_outgoing_total_minutes"] / row["dst_prev_total_minutes"]) if row["dst_prev_total_minutes"] else np.nan
        if "minutes" in row and "dest_outgoing_total_minutes" in row:
            row["player_vs_vacancy_minutes_ratio"] = float(row["minutes"] / row["dest_outgoing_total_minutes"]) if row["dest_outgoing_total_minutes"] else np.nan
        if "points_per_game_prev" in row and "dest_outgoing_sum_points_per_game" in row:
            row["player_vs_vacancy_ppg_ratio"] = float(row["points_per_game_prev"] / row["dest_outgoing_sum_points_per_game"]) if row["dest_outgoing_sum_points_per_game"] else np.nan
        if "assists_per_game_prev" in row and "dest_outgoing_sum_assists_per_game" in row:
            row["player_vs_vacancy_apg_ratio"] = float(row["assists_per_game_prev"] / row["dest_outgoing_sum_assists_per_game"]) if row["dest_outgoing_sum_assists_per_game"] else np.nan
        if "dest_incoming_count" in row and "dest_outgoing_count" in row:
            row["dest_competition_delta"] = float((row.get("dest_incoming_count") or 0.0) - (row.get("dest_outgoing_count") or 0.0))
        return pd.DataFrame([row])

    def _predict_incoming_player(self, player_name: str, dest_team: str, outgoing_names: list[str], incoming_names: list[str]) -> dict:
        row_df = self._build_transfer_row(player_name, dest_team, outgoing_names, incoming_names)
        delta = self._transform_predict(row_df, self.player_transfer_bundle, self.player_transfer_model)
        player = self.player_lookup[player_name]
        result = {"player_name": player_name, "source_team": player["team_abbreviation"], "projection_type": "incoming"}
        for idx, col in enumerate(PLAYER_TARGET_COLUMNS):
            base = float(player[col]) if pd.notna(player[col]) else 0.0
            result[col] = float(base + delta[0, idx])
        return result

    def _build_retained_row(self, player_name: str, team: str, outgoing_names: list[str], incoming_names: list[str]) -> pd.DataFrame:
        player = self.player_lookup[player_name]
        team_row = self.team_lookup[team]
        prev_roster = self.roster_lookup.get(team, pd.DataFrame()).copy()
        outgoing = prev_roster[prev_roster["player_name"].isin(outgoing_names)].copy()
        retained = prev_roster[~prev_roster["player_name"].isin(outgoing_names)].copy()
        other_retained = retained[retained["player_name"] != player_name].copy()
        incoming_rows = [self.player_lookup[name] for name in incoming_names if name in self.player_lookup]
        incoming = pd.DataFrame(incoming_rows) if incoming_rows else pd.DataFrame(columns=self.current_players.columns)
        retained_group_columns = [
            "minutes",
            "games_played",
            "minutes_per_game",
            "points_per_game",
            "rebounds_per_game",
            "assists_per_game",
            "steals_per_game",
            "blocks_per_game",
            "turnovers_per_game",
            "true_shooting_pct",
            "effective_field_goal_pct",
            "per",
            "defensive_rating",
            "db_age",
            "db_o_dpm",
            "db_dpm",
            "db_total_rapm",
        ]
        row = {
            "player_name": player_name,
            "prev_year": CURRENT_YEAR,
            "next_year": NEXT_YEAR,
            "prev_season": CURRENT_SEASON_LABEL,
            "next_season": NEXT_SEASON_LABEL,
            "team_abbreviation": team,
            "next_team_abbreviation": team,
        }
        for col in retained_group_columns:
            if col in player.index:
                row[col] = float(player[col]) if pd.notna(player[col]) else np.nan
        for col in TEAM_INPUT_COLUMNS:
            row[f"prev_team_{col}"] = float(team_row[col]) if pd.notna(team_row[col]) else np.nan
        row.update(aggregate_group(prev_roster, "prev_roster"))
        row.update(aggregate_group(other_retained, "other_retained"))
        row.update(aggregate_group(outgoing, "outgoing"))
        row.update(aggregate_group(incoming, "incoming"))
        if pd.notna(player.get("db_age", np.nan)):
            row["projected_age_next"] = float(player["db_age"] + 1.0)
            row["projected_age_next_sq"] = row["projected_age_next"] ** 2
        denom = max(len(prev_roster) - 1, 1)
        row["continuity_ratio_without_player"] = float(len(other_retained) / denom)
        row["incoming_minus_outgoing_minutes"] = row.get("incoming_total_minutes", 0.0) - row.get("outgoing_total_minutes", 0.0)
        row["incoming_minus_outgoing_count"] = row.get("incoming_count", 0.0) - row.get("outgoing_count", 0.0)
        if "minutes" in row and row.get("outgoing_total_minutes", 0.0):
            row["player_vs_outgoing_minutes_ratio"] = float(row["minutes"] / row["outgoing_total_minutes"])
        return pd.DataFrame([row])

    def _predict_retained_player(self, player_name: str, team: str, outgoing_names: list[str], incoming_names: list[str]) -> dict:
        row_df = self._build_retained_row(player_name, team, outgoing_names, incoming_names)
        delta = self._transform_predict(row_df, self.player_retention_bundle, self.player_retention_model)
        player = self.player_lookup[player_name]
        result = {"player_name": player_name, "source_team": team, "projection_type": "retained"}
        for idx, col in enumerate(PLAYER_TARGET_COLUMNS):
            base = float(player[col]) if pd.notna(player[col]) else 0.0
            result[col] = float(base + delta[0, idx])
        return result

    def simulate(self, team: str, outgoing_names: list[str], incoming_names: list[str]) -> dict:
        if team not in self.team_lookup:
            raise ValueError("Unknown team.")
        current_roster = self.roster_lookup.get(team, pd.DataFrame())
        roster_names = set(current_roster["player_name"].tolist())
        valid_outgoing = [name for name in outgoing_names if name in roster_names]
        valid_incoming = [name for name in incoming_names if name in self.player_lookup and self.player_lookup[name]["team_abbreviation"] != team]
        team_projection = self._predict_team(team, valid_outgoing, valid_incoming)
        retained_names = [name for name in current_roster["player_name"].tolist() if name not in valid_outgoing]
        player_projections = [self._predict_retained_player(name, team, valid_outgoing, valid_incoming) for name in retained_names]
        player_projections += [self._predict_incoming_player(name, team, valid_outgoing, valid_incoming) for name in valid_incoming]
        player_projections = sorted(player_projections, key=lambda x: x.get("points_per_game", 0.0), reverse=True)
        serialized = [
            self._serialize_projected_player(
                player_name=item["player_name"],
                source_team=item.get("source_team") or "",
                projection_type=item.get("projection_type") or "projected",
                stats=item,
            )
            for item in player_projections
        ]
        return {
            "team": team_projection,
            "outgoing": valid_outgoing,
            "incoming": valid_incoming,
            "players": serialized,
        }


STATE = AppState()
METRICS = SiteMetrics(SITE_METRICS_PATH)


class Handler(BaseHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        super().end_headers()

    def _send_json(self, payload: dict | list, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str | None = None) -> None:
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type or (mimetypes.guess_type(str(path))[0] or "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            METRICS.increment_visit()
            return self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        if parsed.path.startswith("/assets/"):
            rel_parts = [part for part in parsed.path[len("/assets/"):].split("/") if part]
            if not rel_parts or rel_parts[0] not in PUBLIC_ASSET_DIRS:
                return self.send_error(404)
            asset_path = (ASSETS_DIR.joinpath(*rel_parts)).resolve()
            try:
                asset_path.relative_to(ASSETS_DIR.resolve())
            except ValueError:
                return self.send_error(404)
            if not asset_path.exists() or not asset_path.is_file():
                return self.send_error(404)
            return self._send_file(asset_path)
        if parsed.path == "/api/home":
            return self._send_json(STATE.home_payload())
        if parsed.path == "/api/site_metrics":
            return self._send_json(METRICS.snapshot())
        if parsed.path == "/healthz":
            return self._send_json({"ok": True, "season": CURRENT_SEASON_LABEL})
        if parsed.path == "/api/teams":
            return self._send_json(STATE.list_teams())
        if parsed.path == "/api/player_stats":
            return self._send_json(STATE.player_stats_table())
        if parsed.path == "/api/team_stats":
            return self._send_json(STATE.team_stats_table())
        if parsed.path == "/api/team_view":
            team = parse_qs(parsed.query).get("team", [""])[0]
            return self._send_json(STATE.team_view(team))
        if parsed.path == "/api/roster":
            team = parse_qs(parsed.query).get("team", [""])[0]
            return self._send_json(STATE.team_roster(team))
        if parsed.path == "/api/players_by_team":
            query = parse_qs(parsed.query)
            team = query.get("team", [""])[0]
            exclude_team = query.get("exclude_team", [""])[0]
            return self._send_json(STATE.players_by_team(team, exclude_team))
        self.send_error(404)

    def do_POST(self) -> None:
        if self.path != "/api/simulate":
            return self.send_error(404)
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self._send_json({"error": "Invalid Content-Length."}, status=400)
        if length <= 0:
            return self._send_json({"error": "Empty request body."}, status=400)
        if length > MAX_JSON_BODY_BYTES:
            return self._send_json({"error": "Request body too large."}, status=413)
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return self._send_json({"error": "Invalid JSON body."}, status=400)
        if not isinstance(data, dict):
            return self._send_json({"error": "JSON body must be an object."}, status=400)
        try:
            result = STATE.simulate(
                team=str(data.get("team", "")).strip(),
                outgoing_names=list(data.get("outgoing", [])),
                incoming_names=list(data.get("incoming", [])),
            )
        except Exception as exc:
            return self._send_json({"error": str(exc)}, status=400)
        result["site_metrics"] = METRICS.increment_trade_simulation()
        return self._send_json(result)


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
