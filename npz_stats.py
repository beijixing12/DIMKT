"""Summarize user/problem/interaction counts from NPZ datasets.

This helper normalizes common column aliases (user_id, question_id, skill, y/attempt)
so NPZ files with different field names can be inspected consistently.
"""
import argparse
from typing import Dict, Iterable, Tuple

import numpy as np


ALIASES: Dict[str, Iterable[str]] = {
    "user_id": ["user_id", "uid", "student_id", "user"],
    "problem_id": ["problem_id", "question_id", "item_id", "pid", "item"],
    "skill_id": ["skill_id", "skill", "kc", "concept_id"],
    "correct": ["correct", "y", "attempt", "label", "is_correct", "score"],
}


def _find_column(raw_cols: Iterable[str], canon: str) -> str:
    raw_set = set(raw_cols)
    for alias in ALIASES[canon]:
        if alias in raw_set:
            return alias
    raise ValueError(f"Missing required column for '{canon}'. Acceptable aliases: {ALIASES[canon]}")


def _normalize_columns(raw_cols: Iterable[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    mapping["user_id"] = _find_column(raw_cols, "user_id")
    mapping["problem_id"] = _find_column(raw_cols, "problem_id")
    mapping["skill_id"] = _find_column(raw_cols, "skill_id")
    # correct is optional for counting interactions but we still try to map it
    try:
        mapping["correct"] = _find_column(raw_cols, "correct")
    except ValueError:
        mapping["correct"] = ""
    return mapping


def _load_array(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    values = np.array(data[key]).ravel()
    # Flatten nested/array cells to their first element when needed
    flattened = []
    for val in values:
        if isinstance(val, (list, tuple, np.ndarray)):
            arr = np.array(val).ravel()
            flattened.append(arr[0] if arr.size else np.nan)
        else:
            flattened.append(val)
    return np.asarray(flattened)


def summarize_npz(path: str) -> Tuple[int, int, int]:
    data = np.load(path, allow_pickle=True)
    # Detect non-interaction graph-style NPZ inputs and fail fast
    graph_keys = {"concepts", "exercises", "learners", "concept_exercise", "concept_learner"}
    if graph_keys.issubset(set(data.files)):
        raise ValueError(
            "Provided NPZ appears to store graph/statistics arrays rather than per-interaction records. "
            "Please convert it to interaction rows before computing statistics."
        )

    mapping = _normalize_columns(data.files)
    user_arr = _load_array(data, mapping["user_id"])
    problem_arr = _load_array(data, mapping["problem_id"])
    skill_arr = _load_array(data, mapping["skill_id"])

    # Use the minimum aligned length across required columns to avoid length mismatches
    aligned_len = min(len(user_arr), len(problem_arr), len(skill_arr))
    user_arr = user_arr[:aligned_len]
    problem_arr = problem_arr[:aligned_len]

    num_users = len(np.unique(user_arr))
    num_problems = len(np.unique(problem_arr))
    num_interactions = aligned_len
    return num_users, num_problems, num_interactions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize unique users, unique problems, and total interactions from an NPZ file"
    )
    parser.add_argument("input_npz", help="Path to NPZ file containing interaction arrays")
    args = parser.parse_args()

    users, problems, interactions = summarize_npz(args.input_npz)
    print(f"Users: {users}")
    print(f"Problems: {problems}")
    print(f"Interactions: {interactions}")


if __name__ == "__main__":
    main()
