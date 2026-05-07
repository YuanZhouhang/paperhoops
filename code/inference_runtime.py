from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


PLAYER_INPUT_COLUMNS = [
    "games_played",
    "minutes",
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
    "db_offposs",
    "db_transition_ts",
]

PLAYER_TARGET_COLUMNS = [
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
]

TEAM_INPUT_COLUMNS = [
    "wins",
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

TEAM_TARGET_COLUMNS = [
    "wins",
    "win_pct",
    "pace",
    "offensive_rating",
    "defensive_rating",
    "net_rating",
]

TEAM_TARGET_COLUMNS_2 = TEAM_TARGET_COLUMNS
GAMES_PER_TEAM = 82.0
ROTATION_MIN_GAMES = 10.0
ROTATION_MIN_MPG = 10.0
ROTATION_MIN_TOTAL_MINUTES = 450.0
HISTORY_LOOKBACK_SEASONS = 3
HISTORY_BASE_WEIGHTS = np.array([0.60, 0.28, 0.12], dtype="float32")

ROSTER_TOP_COLUMNS = [
    "minutes_per_game",
    "points_per_game",
    "rebounds_per_game",
    "assists_per_game",
    "true_shooting_pct",
    "effective_field_goal_pct",
    "per",
    "defensive_rating",
    "db_o_dpm",
    "db_dpm",
    "db_total_rapm",
]

MOVEMENT_AGG_COLUMNS = [
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
    "db_o_dpm",
    "db_dpm",
    "db_total_rapm",
]

PLAYER_GROUP_COLUMNS = [
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

RUNTIME_PLAYER_SOURCE_COLUMNS = sorted(
    {
        "player_name",
        "season",
        "season_end_year",
        "team_abbreviation",
        "team_name",
        *PLAYER_INPUT_COLUMNS,
        *PLAYER_TARGET_COLUMNS,
        *PLAYER_GROUP_COLUMNS,
    }
)

RUNTIME_TEAM_SOURCE_COLUMNS = sorted(
    {
        "season",
        "season_end_year",
        "team_abbreviation",
        "team_name",
        "games_played",
        "losses",
        *TEAM_INPUT_COLUMNS,
        *TEAM_TARGET_COLUMNS,
    }
)


def clean_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype("float32")
    cleaned = (
        series.astype(str)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan, "null": np.nan, "NULL": np.nan})
        .str.replace(",", "", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce").astype("float32")


def estimate_minutes_frame(frame: pd.DataFrame) -> pd.Series:
    if "minutes" in frame.columns:
        minutes = pd.to_numeric(frame["minutes"], errors="coerce").astype("float32")
    else:
        minutes = pd.Series(np.nan, index=frame.index, dtype="float32")
    if {"games_played", "minutes_per_game"}.issubset(frame.columns):
        games = pd.to_numeric(frame["games_played"], errors="coerce").astype("float32")
        mpg = pd.to_numeric(frame["minutes_per_game"], errors="coerce").astype("float32")
        estimated = games * mpg
        minutes = minutes.where(minutes.notna() & minutes.gt(0), estimated)
    return minutes.fillna(0.0).astype("float32")


def add_estimated_minutes(players: pd.DataFrame) -> pd.DataFrame:
    players = players.copy()
    players["minutes"] = estimate_minutes_frame(players)
    return players


def effective_rotation_mask(players: pd.DataFrame) -> pd.Series:
    if len(players) == 0:
        return pd.Series([], index=players.index, dtype=bool)
    games_column = "season_games_played" if "season_games_played" in players.columns else "games_played"
    mpg_column = "season_minutes_per_game" if "season_minutes_per_game" in players.columns else "minutes_per_game"
    minutes_column = "season_minutes" if "season_minutes" in players.columns else "minutes"
    raw_games = pd.to_numeric(players[games_column], errors="coerce").fillna(0.0) if games_column in players else pd.Series(0.0, index=players.index)
    raw_mpg = pd.to_numeric(players[mpg_column], errors="coerce").fillna(0.0) if mpg_column in players else pd.Series(0.0, index=players.index)
    raw_minutes = pd.to_numeric(players[minutes_column], errors="coerce").fillna(0.0) if minutes_column in players else estimate_minutes_frame(players)
    weighted_mpg = pd.to_numeric(players["minutes_per_game"], errors="coerce").fillna(0.0) if "minutes_per_game" in players else raw_mpg
    weighted_points = pd.to_numeric(players["points_per_game"], errors="coerce").fillna(0.0) if "points_per_game" in players else pd.Series(0.0, index=players.index)
    standard_rotation = raw_games.ge(ROTATION_MIN_GAMES) & raw_mpg.ge(ROTATION_MIN_MPG)
    total_minutes_rotation = raw_minutes.ge(ROTATION_MIN_TOTAL_MINUTES)
    injury_exception = raw_games.ge(5.0) & raw_mpg.ge(18.0) & weighted_mpg.ge(18.0) & weighted_points.ge(10.0)
    return standard_rotation | total_minutes_rotation | injury_exception


def filter_effective_rotation_players(players: pd.DataFrame) -> pd.DataFrame:
    if len(players) == 0:
        return players.copy()
    players = add_estimated_minutes(players)
    return players[effective_rotation_mask(players)].copy()


def _history_quality_signal(history: pd.DataFrame) -> pd.Series:
    mpg = pd.to_numeric(history["minutes_per_game"], errors="coerce").fillna(0.0) if "minutes_per_game" in history else pd.Series(0.0, index=history.index)
    ppg = pd.to_numeric(history["points_per_game"], errors="coerce").fillna(0.0) if "points_per_game" in history else pd.Series(0.0, index=history.index)
    per = pd.to_numeric(history["per"], errors="coerce").fillna(13.0) if "per" in history else pd.Series(13.0, index=history.index)
    return mpg + 1.15 * ppg + 0.35 * per


def _dynamic_history_weights(history: pd.DataFrame) -> np.ndarray:
    n = len(history)
    base = HISTORY_BASE_WEIGHTS[:n].astype("float64").copy()
    games = pd.to_numeric(history["games_played"], errors="coerce").fillna(0.0) if "games_played" in history else pd.Series(0.0, index=history.index)
    mpg = pd.to_numeric(history["minutes_per_game"], errors="coerce").fillna(0.0) if "minutes_per_game" in history else pd.Series(0.0, index=history.index)
    reliability = np.sqrt(np.clip(games.to_numpy(dtype="float64") / 65.0, 0.0, 1.0))
    reliability *= np.sqrt(np.clip(mpg.to_numpy(dtype="float64") / 28.0, 0.0, 1.0))
    reliability = np.clip(reliability, 0.25, 1.0)

    signal = _history_quality_signal(history).to_numpy(dtype="float64")
    positive = signal[signal > 0.0]
    if len(positive):
        median_signal = float(np.median(positive))
        low_outlier = (signal < 0.65 * median_signal) & (games.to_numpy(dtype="float64") < 50.0)
        reliability = np.where(low_outlier, reliability * 0.55, reliability)

    weights = base * reliability
    if not np.isfinite(weights).any() or weights.sum() <= 0.0:
        weights = base
    return weights / weights.sum()


def _weighted_history_value(history: pd.DataFrame, column: str, weights: np.ndarray) -> float:
    values = pd.to_numeric(history[column], errors="coerce").to_numpy(dtype="float64")
    valid = np.isfinite(values)
    if not valid.any():
        return np.nan
    local_weights = weights[valid]
    local_weights = local_weights / local_weights.sum()
    value = float(np.dot(values[valid], local_weights))
    if len(values) >= 3 and np.isfinite(values[0]) and np.isfinite(values[-1]):
        recent_signal = _history_quality_signal(history).to_numpy(dtype="float64")
        if np.all(np.diff(recent_signal[::-1]) >= 0.0) or np.all(np.diff(recent_signal[::-1]) <= 0.0):
            span = float(values[0] - values[-1])
            value += 0.15 * span
            finite = values[valid]
            value = float(np.clip(value, finite.min() - abs(span) * 0.25, finite.max() + abs(span) * 0.25))
    return value


def _weighted_history_values(history: pd.DataFrame, numeric_columns: List[str], weights: np.ndarray) -> Dict[str, float]:
    usable_columns = [column for column in numeric_columns if column in history.columns]
    if not usable_columns:
        return {}
    matrix = history[usable_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64")
    valid = np.isfinite(matrix)
    weighted = np.where(valid, matrix * weights.reshape(-1, 1), 0.0)
    denom = np.where(valid, weights.reshape(-1, 1), 0.0).sum(axis=0)
    values = np.full(len(usable_columns), np.nan, dtype="float64")
    has_value = denom > 0.0
    values[has_value] = weighted.sum(axis=0)[has_value] / denom[has_value]

    if len(history) >= 3:
        signal = _history_quality_signal(history).to_numpy(dtype="float64")
        monotonic_signal = np.all(np.diff(signal[::-1]) >= 0.0) or np.all(np.diff(signal[::-1]) <= 0.0)
        if monotonic_signal:
            first = matrix[0]
            last = matrix[-1]
            span = first - last
            finite_span = np.isfinite(span) & np.isfinite(values)
            if finite_span.any():
                adjusted = values + 0.15 * np.where(finite_span, span, 0.0)
                finite_matrix = np.where(valid, matrix, np.nan)
                mins = np.full(len(usable_columns), np.nan, dtype="float64")
                maxs = np.full(len(usable_columns), np.nan, dtype="float64")
                mins[has_value] = np.nanmin(finite_matrix[:, has_value], axis=0)
                maxs[has_value] = np.nanmax(finite_matrix[:, has_value], axis=0)
                margin = np.abs(span) * 0.25
                values = np.where(
                    finite_span,
                    np.clip(adjusted, mins - margin, maxs + margin),
                    values,
                )

    return {
        column: float(value)
        for column, value in zip(usable_columns, values)
        if np.isfinite(value)
    }


def build_recent_weighted_player_stats(
    players: pd.DataFrame,
    numeric_columns: List[str] | None = None,
    lookback: int = HISTORY_LOOKBACK_SEASONS,
) -> pd.DataFrame:
    if len(players) == 0:
        return players.copy()
    numeric_columns = numeric_columns or [
        column
        for column in PLAYER_GROUP_COLUMNS
        if column in players.columns
    ]
    source = add_estimated_minutes(players)
    sort_columns = ["player_name", "season_end_year", "minutes"]
    source = source.sort_values(sort_columns, ascending=[True, True, False])
    player_year_best = source.drop_duplicates(["player_name", "season_end_year"], keep="first")
    history_lookup = {
        name: frame.sort_values("season_end_year", ascending=False).reset_index(drop=True)
        for name, frame in player_year_best.groupby("player_name", dropna=False)
    }

    rows = []
    weighted_cache: Dict[tuple[str, int], Dict[str, float]] = {}
    for _, row in source.iterrows():
        year = int(row["season_end_year"])
        cache_key = (str(row["player_name"]), year)
        history = history_lookup.get(row["player_name"], pd.DataFrame())
        new_row = row.to_dict()
        for raw_column in ["games_played", "minutes", "minutes_per_game", "points_per_game"]:
            if raw_column in row.index:
                new_row[f"season_{raw_column}"] = row[raw_column]
        cached = weighted_cache.get(cache_key)
        if cached is None:
            history_window = history[
                (history["season_end_year"].astype(int) <= year)
                & (history["season_end_year"].astype(int) > year - lookback)
            ].sort_values("season_end_year", ascending=False)
            cached = {}
            if not history_window.empty:
                weights = _dynamic_history_weights(history_window)
                cached.update(_weighted_history_values(history_window, numeric_columns, weights))
                cached["history_weighted_seasons"] = float(len(history_window))
                cached["history_recent_weight"] = float(weights[0]) if len(weights) else 0.0
            weighted_cache[cache_key] = cached
        new_row.update(cached)
        rows.append(new_row)
    return pd.DataFrame(rows)


def project_values_to_sum(
    values: np.ndarray,
    target_sum: float,
    lower: np.ndarray | float,
    upper: np.ndarray | float,
    iterations: int = 80,
) -> np.ndarray:
    """Shift values by a shared offset, with clipping, so the total matches target_sum."""
    values = np.asarray(values, dtype="float64")
    lower_arr = np.broadcast_to(np.asarray(lower, dtype="float64"), values.shape)
    upper_arr = np.broadcast_to(np.asarray(upper, dtype="float64"), values.shape)
    feasible_target = float(np.clip(target_sum, lower_arr.sum(), upper_arr.sum()))
    lo = float(np.min(lower_arr - values))
    hi = float(np.max(upper_arr - values))
    for _ in range(iterations):
        mid = (lo + hi) / 2.0
        current = np.clip(values + mid, lower_arr, upper_arr).sum()
        if current < feasible_target:
            lo = mid
        else:
            hi = mid
    return np.clip(values + (lo + hi) / 2.0, lower_arr, upper_arr).astype("float32")


def normalize_games(games: np.ndarray | pd.Series | None, row_count: int) -> np.ndarray:
    if games is None:
        result = np.full(row_count, GAMES_PER_TEAM, dtype="float32")
    else:
        result = np.asarray(games, dtype="float32")
    return np.where(np.isfinite(result) & (result > 0.0), result, GAMES_PER_TEAM).astype("float32")


def apply_league_team_constraints(
    prediction: np.ndarray,
    games: np.ndarray | pd.Series | None = None,
    target_net_rating_mean: float = 0.0,
) -> np.ndarray:
    prediction = np.asarray(prediction, dtype="float32").copy()
    if len(prediction) == 0:
        return prediction

    idx = {column: index for index, column in enumerate(TEAM_TARGET_COLUMNS)}
    games_arr = normalize_games(games, len(prediction))

    if {"offensive_rating", "defensive_rating", "net_rating"}.issubset(idx):
        net_rating = prediction[:, idx["offensive_rating"]] - prediction[:, idx["defensive_rating"]]
        net_shift = float(target_net_rating_mean - net_rating.mean())
        prediction[:, idx["offensive_rating"]] += net_shift / 2.0
        prediction[:, idx["defensive_rating"]] -= net_shift / 2.0
        prediction[:, idx["net_rating"]] = prediction[:, idx["offensive_rating"]] - prediction[:, idx["defensive_rating"]]

    if "wins" in idx:
        target_wins = float(games_arr.sum() / 2.0)
        prediction[:, idx["wins"]] = project_values_to_sum(
            prediction[:, idx["wins"]],
            target_wins,
            lower=0.0,
            upper=games_arr,
        )

    if "win_pct" in idx:
        prediction[:, idx["win_pct"]] = project_values_to_sum(
            prediction[:, idx["win_pct"]],
            target_sum=float(len(prediction) / 2.0),
            lower=0.0,
            upper=1.0,
        )

    return prediction


def league_constraint_diagnostics(
    prediction: np.ndarray,
    games: np.ndarray | pd.Series | None = None,
) -> Dict[str, float]:
    prediction = np.asarray(prediction, dtype="float32")
    idx = {column: index for index, column in enumerate(TEAM_TARGET_COLUMNS)}
    games_arr = normalize_games(games, len(prediction))
    net_identity_error = prediction[:, idx["net_rating"]] - (
        prediction[:, idx["offensive_rating"]] - prediction[:, idx["defensive_rating"]]
    )
    return {
        "wins_sum": float(prediction[:, idx["wins"]].sum()),
        "target_wins_sum": float(games_arr.sum() / 2.0),
        "wins_sum_error": float(prediction[:, idx["wins"]].sum() - games_arr.sum() / 2.0),
        "win_pct_sum": float(prediction[:, idx["win_pct"]].sum()),
        "target_win_pct_sum": float(len(prediction) / 2.0),
        "win_pct_sum_error": float(prediction[:, idx["win_pct"]].sum() - len(prediction) / 2.0),
        "net_rating_mean": float(prediction[:, idx["net_rating"]].mean()),
        "net_rating_sum": float(prediction[:, idx["net_rating"]].sum()),
        "net_rating_identity_max_abs": float(np.max(np.abs(net_identity_error))),
    }


def load_players(player_path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    players = pd.read_csv(player_path, low_memory=False, usecols=usecols)
    players["season"] = players["season"].astype(str)
    players["team_abbreviation"] = players["team_abbreviation"].astype(str)
    id_columns = {"player_name", "season", "season_end_year", "team_abbreviation", "team_name"}
    for column in players.columns:
        if column in id_columns:
            continue
        players[column] = clean_numeric_series(players[column])
    players["season_end_year"] = clean_numeric_series(players["season_end_year"]).astype("Int64")
    return players


def load_teams(team_path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    teams = pd.read_csv(team_path, low_memory=False, usecols=usecols)
    teams["season"] = teams["season"].astype(str)
    teams["team_abbreviation"] = teams["team_abbreviation"].astype(str)
    for column in teams.columns:
        if column in {"season", "team_abbreviation", "team_name"}:
            continue
        teams[column] = clean_numeric_series(teams[column])
    teams["season_end_year"] = teams["season_end_year"].astype("Int64")
    return teams


def build_team_roster_features(players: pd.DataFrame, prefix: str) -> pd.DataFrame:
    group_keys = ["season_end_year", "team_abbreviation"]
    top_columns = [col for col in ROSTER_TOP_COLUMNS if col in players.columns]
    players = add_estimated_minutes(players)
    sortable = players[group_keys + ["player_name", "minutes"] + top_columns].copy()
    sortable = sortable.sort_values(group_keys + ["minutes"], ascending=[True, True, False])

    rows: List[Dict[str, float]] = []
    for key, group in sortable.groupby(group_keys, dropna=False):
        total_minutes = float(group["minutes"].sum())
        row: Dict[str, float] = {
            "season_end_year": int(key[0]),
            "team_abbreviation": key[1],
            f"{prefix}_roster_size": float(group["player_name"].nunique()),
            f"{prefix}_total_minutes": total_minutes,
            f"{prefix}_top3_min_share": float(group["minutes"].nlargest(3).sum() / total_minutes) if total_minutes > 0 else np.nan,
            f"{prefix}_top5_min_share": float(group["minutes"].nlargest(5).sum() / total_minutes) if total_minutes > 0 else np.nan,
        }
        top_players = group.head(5).reset_index(drop=True)
        for idx in range(5):
            base = f"{prefix}_top_{idx + 1}"
            if idx < len(top_players):
                row[f"{base}_minutes"] = float(top_players.loc[idx, "minutes"])
                for col in top_columns:
                    value = top_players.loc[idx, col]
                    row[f"{base}_{col}"] = float(value) if pd.notna(value) else np.nan
            else:
                row[f"{base}_minutes"] = np.nan
                for col in top_columns:
                    row[f"{base}_{col}"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_group(group: pd.DataFrame, prefix: str) -> Dict[str, float]:
    row: Dict[str, float] = {f"{prefix}_count": float(len(group))}
    if len(group) == 0:
        for col in PLAYER_GROUP_COLUMNS:
            row[f"{prefix}_sum_{col}"] = 0.0
            row[f"{prefix}_mean_{col}"] = np.nan
            row[f"{prefix}_wavg_{col}"] = np.nan
        row[f"{prefix}_total_minutes"] = 0.0
        row[f"{prefix}_top3_min_share"] = np.nan
        row[f"{prefix}_projected_age_mean"] = np.nan
        return row

    minutes = estimate_minutes_frame(group)
    total_minutes = float(minutes.sum())
    row[f"{prefix}_total_minutes"] = total_minutes
    row[f"{prefix}_top3_min_share"] = float(minutes.nlargest(3).sum() / total_minutes) if total_minutes > 0 else np.nan
    for col in [c for c in PLAYER_GROUP_COLUMNS if c in group.columns]:
        series = group[col]
        row[f"{prefix}_sum_{col}"] = float(series.fillna(0.0).sum())
        row[f"{prefix}_mean_{col}"] = float(series.mean()) if series.notna().any() else np.nan
        valid = series.notna() & minutes.gt(0)
        row[f"{prefix}_wavg_{col}"] = float(np.average(series[valid], weights=minutes[valid])) if valid.any() else np.nan
    if "db_age" in group.columns:
        ages = group["db_age"].dropna()
        row[f"{prefix}_projected_age_mean"] = float((ages + 1.0).mean()) if not ages.empty else np.nan
    return row


def aggregate_movement_features(moves: pd.DataFrame, key_col: str, prefix: str) -> pd.DataFrame:
    use_cols = [c for c in MOVEMENT_AGG_COLUMNS if c in moves.columns]
    moves = add_estimated_minutes(moves)
    rows = []
    for key, group in moves.groupby(["next_year", key_col], dropna=False):
        row: Dict[str, float] = {"next_year": int(key[0]), "team_abbreviation": key[1]}
        row[f"{prefix}_count"] = float(len(group))
        weights = estimate_minutes_frame(group)
        total_minutes = float(weights.sum())
        row[f"{prefix}_total_minutes"] = total_minutes
        row[f"{prefix}_top3_min_share"] = float(weights.nlargest(3).sum() / total_minutes) if total_minutes > 0 else np.nan
        for col in use_cols:
            series = group[col]
            row[f"{prefix}_mean_{col}"] = float(series.mean()) if series.notna().any() else np.nan
            row[f"{prefix}_sum_{col}"] = float(series.fillna(0.0).sum())
            if total_minutes > 0:
                valid = series.notna() & weights.gt(0)
                row[f"{prefix}_wavg_{col}"] = float(np.average(series[valid], weights=weights[valid])) if valid.any() else np.nan
            else:
                row[f"{prefix}_wavg_{col}"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)
