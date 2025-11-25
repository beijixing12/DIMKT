"""Convert NPZ interaction-style arrays into a CSV for prepare_dataset.py.

This helper normalizes common aliases used in the provided NPZs and emits a
flat table with columns: user_id, problem_id, correct, skill_id, end_time.
It uses the original record order to synthesize timestamps when no usable
end_time-like field exists.
"""
import argparse
import os
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


ALIASES: Dict[str, Iterable[str]] = {
    "user_id": ["user_id", "uid", "student_id", "user"],
    "problem_id": ["problem_id", "question_id", "item_id", "pid", "item"],
    "correct": ["correct", "y", "attempt", "label", "is_correct", "score"],
    "skill_id": ["skill_id", "skill", "kc", "concept_id"],
    "end_time": ["end_time", "timestamp", "time", "ts", "unix_time", "response_time", "first_response_time"],
}


class MissingColumn(Exception):
    pass


def _find_column(raw_cols: Iterable[str], canon: str, allow_missing: bool = False) -> Optional[str]:
    raw_set = set(raw_cols)
    for alias in ALIASES[canon]:
        if alias in raw_set:
            return alias
    if allow_missing:
        return None
    raise MissingColumn(f"Missing required column for '{canon}'. Acceptable aliases: {ALIASES[canon]}")


def _normalize_columns(raw_cols: Iterable[str]) -> Dict[str, Optional[str]]:
    mapping: Dict[str, Optional[str]] = {}
    mapping["user_id"] = _find_column(raw_cols, "user_id")
    mapping["problem_id"] = _find_column(raw_cols, "problem_id")
    mapping["correct"] = _find_column(raw_cols, "correct")
    mapping["skill_id"] = _find_column(raw_cols, "skill_id")
    mapping["end_time"] = _find_column(raw_cols, "end_time", allow_missing=True)
    return mapping


def _coerce_scalar(val):
    if isinstance(val, (list, tuple, np.ndarray)):
        arr = np.array(val).ravel()
        if arr.size == 0:
            return np.nan
        return arr[0]
    return val


def _load_series(data: np.lib.npyio.NpzFile, name: str) -> pd.Series:
    values = np.array(data[name]).flatten()
    return pd.Series(values).apply(_coerce_scalar)


def _synthesize_timestamps(
    df: pd.DataFrame,
    time_col: Optional[str],
    duration_sum: Optional[pd.Series] = None,
) -> pd.Series:
    # Prefer the sum of first/response durations when both exist and contain data.
    if duration_sum is not None:
        series = pd.to_numeric(duration_sum, errors="coerce")
        if not series.isna().all():
            grouped = series.groupby(df["user_id"])

            def _make_ts(group: pd.Series) -> pd.Series:
                diffs = group.diff().fillna(0)
                if (diffs < 0).any():
                    return group.cumsum()
                return group

            adjusted = grouped.transform(_make_ts)
            if not adjusted.isna().all():
                return adjusted.fillna(0.0)

    if time_col is None:
        # simple deterministic ordering per user
        return df.groupby("user_id").cumcount().astype(float)

    series = pd.to_numeric(df[time_col], errors="coerce")
    if series.isna().all():
        # fallback to deterministic ordering
        return df.groupby("user_id").cumcount().astype(float)

    # If the column looks like a duration (non-strictly increasing), convert to cumulative per user
    grouped = series.groupby(df["user_id"])

    def _make_ts(group: pd.Series) -> pd.Series:
        diffs = group.diff().fillna(0)
        if (diffs < 0).any():
            return group.cumsum()
        return group

    adjusted = grouped.transform(_make_ts)
    return adjusted.fillna(0.0)


def convert(input_path: str, output_csv: str) -> None:
    data = np.load(input_path, allow_pickle=True)
    mapping = _normalize_columns(data.files)

    df_dict = {}
    for canon, raw in mapping.items():
        if raw is None:
            continue
        df_dict[canon] = np.array(data[raw]).flatten()
    df = pd.DataFrame(df_dict)

    for col in df.columns:
        df[col] = df[col].apply(_coerce_scalar)

    # Prefer summing first_response_time + response_time when both exist.
    duration_sum = None
    if "first_response_time" in data.files and "response_time" in data.files:
        first_rt = _load_series(data, "first_response_time")
        resp_time = _load_series(data, "response_time")
        duration_sum = pd.to_numeric(first_rt, errors="coerce").add(
            pd.to_numeric(resp_time, errors="coerce"), fill_value=0
        )

    # Build timestamps
    df["end_time"] = _synthesize_timestamps(
        df, mapping.get("end_time"), duration_sum=duration_sum
    )

    # Ensure canonical column order
    df = df[["user_id", "problem_id", "correct", "skill_id", "end_time"]]
    df.to_csv(output_csv, index=False)
    print(f"Wrote converted CSV to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Convert NPZ arrays into a CSV with canonical columns")
    parser.add_argument("input_npz", help="Path to NPZ file containing user/problem/skill/correct arrays")
    parser.add_argument(
        "--output_csv",
        default=os.path.join("data", "converted.csv"),
        help="Destination CSV path (default: data/converted.csv)",
    )
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    convert(args.input_npz, args.output_csv)


if __name__ == "__main__":
    main()
