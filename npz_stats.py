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


def _flatten_interactions(arr: np.ndarray) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """Flatten per-user interaction sequences while tracking their lengths."""

    # Fast path: already a flat numeric array (one interaction per row)
    if arr.dtype != object:
        flat = np.asarray(arr).ravel()
        return flat, tuple(1 for _ in range(len(flat)))

    lengths = []
    sequences = []
    for idx, val in enumerate(arr):
        if isinstance(val, (list, tuple, np.ndarray)):
            seq = np.asarray(val).ravel()
        else:
            seq = np.asarray([val])

        lengths.append(len(seq))
        sequences.append(seq)

        if len(seq) == 0:
            raise ValueError(f"Empty interaction sequence encountered at index {idx}")

    flat = np.concatenate(sequences) if sequences else np.array([], dtype=float)
    return flat, tuple(lengths)


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
    user_arr = np.array(data[mapping["user_id"]]).ravel()
    problem_flat, problem_lengths = _flatten_interactions(np.array(data[mapping["problem_id"]]))
    skill_flat, skill_lengths = _flatten_interactions(np.array(data[mapping["skill_id"]]))

    if len(problem_lengths) != len(user_arr):
        raise ValueError(
            "User array length does not match per-user interaction sequences: "
            f"users={len(user_arr)}, problem sequences={len(problem_lengths)}"
        )
    if problem_lengths != skill_lengths:
        mismatch_idx = next(idx for idx, pair in enumerate(zip(problem_lengths, skill_lengths)) if pair[0] != pair[1])
        raise ValueError(
            "Mismatched interaction lengths between problem_id and skill_id columns at index "
            f"{mismatch_idx}: problem_len={problem_lengths[mismatch_idx]}, skill_len={skill_lengths[mismatch_idx]}"
        )

    num_users = len(np.unique(user_arr))
    num_problems = len(np.unique(problem_flat))
    num_interactions = int(sum(problem_lengths))
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
