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


def _coerce_sequence(val) -> list:
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (list, tuple)):
        return list(val)
    return [val]


def _load_object_array(data: np.lib.npyio.NpzFile, name: str) -> list:
    values = np.array(data[name], dtype=object)
    # The stored arrays are typically shaped (num_users,) with each element a
    # per-user interaction sequence. Preserve that structure so we can expand
    # it later instead of truncating to the first entry.
    return [_coerce_sequence(v) for v in values]


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

    # Expand per-user sequences into per-interaction rows
    user_ids = np.array(data[mapping["user_id"]], dtype=object).tolist()
    problem_seqs = _load_object_array(data, mapping["problem_id"])
    correct_seqs = _load_object_array(data, mapping["correct"])
    skill_seqs = _load_object_array(data, mapping["skill_id"])

    end_time_seqs = None
    if mapping.get("end_time") is not None:
        end_time_seqs = _load_object_array(data, mapping["end_time"])

    first_rt_seqs = resp_time_seqs = None
    if "first_response_time" in data.files and "response_time" in data.files:
        first_rt_seqs = _load_object_array(data, "first_response_time")
        resp_time_seqs = _load_object_array(data, "response_time")

    rows = []
    durations = []

    for idx, uid in enumerate(user_ids):
        problems = problem_seqs[idx]
        corrects = correct_seqs[idx]
        skills = skill_seqs[idx]

        length = len(problems)
        if not (len(corrects) == len(skills) == length):
            raise ValueError(
                f"Mismatched sequence lengths for user {uid}: "
                f"problems={len(problems)}, correct={len(corrects)}, skills={len(skills)}"
            )

        end_times = end_time_seqs[idx] if end_time_seqs is not None else [np.nan] * length

        duration_seq = None
        if first_rt_seqs is not None and resp_time_seqs is not None:
            frt = first_rt_seqs[idx]
            rt = resp_time_seqs[idx]
            if len(frt) != length or len(rt) != length:
                raise ValueError(
                    f"Mismatched response time lengths for user {uid}: "
                    f"first_response_time={len(frt)}, response_time={len(rt)}, expected={length}"
                )
            duration_seq = pd.Series(frt, dtype="float64").add(
                pd.Series(rt, dtype="float64"), fill_value=0
            )

        for j in range(length):
            rows.append(
                {
                    "user_id": uid,
                    "problem_id": problems[j],
                    "correct": corrects[j],
                    "skill_id": skills[j],
                    "end_time": end_times[j],
                }
            )

            if duration_seq is not None:
                durations.append(duration_seq.iloc[j])
            else:
                durations.append(np.nan)

    df = pd.DataFrame(rows)

    # Prefer summing first_response_time + response_time when both exist.
    duration_series = pd.Series(durations, dtype=float)
    duration_sum = duration_series if duration_series.notna().any() else None

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
