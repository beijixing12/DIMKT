"""Combine data_pre.py and data_save.py preprocessing for custom datasets.

This script aligns with the original TensorFlow preprocessing flow:
1) Generate mapping and difficulty files (user2id/problem2id/skill2id,
   difficult2id/sdifficult2id, nones/nonesk) in ``data/``.
2) Slice user interaction sequences into ``train0.npy``, ``valid0.npy``
   (first KFold split) and ``test.npy`` saved in ``data/`` using the
   same filtering rules and random seeds as ``data_save.py``.

It accepts either CSV or NPZ inputs with columns equivalent to
``user_id``, ``problem_id``, ``correct``, ``skill_id``, and
``end_time``. Column alias handling is kept minimal to support common
variants while keeping the rest of the logic identical to the original
scripts.
"""
import argparse
import os
import time
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold, train_test_split

pd.set_option("display.float_format", lambda x: "%.2f" % x)
np.set_printoptions(suppress=True)


def _normalize_columns(raw_cols: Iterable[str], allow_missing_end_time: bool = False) -> Dict[str, str]:
    aliases = {
        "user_id": ["user_id", "uid", "user", "student_id", "stu_id"],
        "problem_id": ["problem_id", "item_id", "pid", "question_id", "item"],
        "correct": [
            "correct",
            "label",
            "is_correct",
            "score",
            "correctness",
            "answer",
            "answers",
            "resp",
            "response",
            "outcome",
            "y",
            "attempt",
        ],
        "skill_id": ["skill_id", "skill", "kc", "concept_id"],
        "end_time": [
            "end_time",
            "timestamp",
            "time",
            "ts",
            "unix_time",
            "response_time",
            "first_response_time",
        ],
    }

    raw_cols_set = set(raw_cols)
    mapping: Dict[str, str] = {}
    for canon, options in aliases.items():
        for col in options:
            if col in raw_cols_set:
                mapping[canon] = col
                break
        if canon not in mapping:
            if canon == "end_time" and allow_missing_end_time:
                mapping[canon] = None
                continue
            raise ValueError(
                f"Missing required column for '{canon}'. Acceptable aliases: {options}"
            )
    return mapping


def _read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="ISO-8859-1", low_memory=False)
    col_map = _normalize_columns(df.columns)
    df = df[[col_map[c] for c in ["user_id", "problem_id", "correct", "skill_id", "end_time"]]]
    df.columns = ["user_id", "problem_id", "correct", "skill_id", "end_time"]
    return df


def _read_npz(path: str) -> pd.DataFrame:
    data = np.load(path, allow_pickle=True)

    # Detect graph/statistics-style NPZ files early and fail with a clearer
    # message since they lack per-interaction logs required by this script.
    graph_like_keys = {"concepts", "exercises", "learners", "concept_exercise", "concept_learner"}
    if graph_like_keys.issubset(set(data.files)):
        available = ", ".join(sorted(data.files))
        raise ValueError(
            "The provided NPZ stores concept/exercise graph statistics rather than per-user interaction "
            "logs. Please convert it to a table with columns user_id, problem_id, correct, skill_id, "
            f"end_time before running this script. Available NPZ arrays: {available}"
        )

    try:
        col_map = _normalize_columns(data.files, allow_missing_end_time=True)
    except ValueError as e:
        available = ", ".join(sorted(data.files))
        raise ValueError(
            "NPZ input is missing required per-interaction columns (user_id, "
            "problem_id, correct, skill_id, end_time). Please convert the file "
            "to include these arrays before running this script. Available arrays: "
            f"{available}. Original error: {e}"
        ) from e
    df_dict = {}
    for canon, raw in col_map.items():
        if raw is None:
            continue
        values = data[raw]
        df_dict[canon] = np.array(values).flatten()
    df = pd.DataFrame(df_dict)

    # If no timestamp-like column existed, synthesize a deterministic ordering
    # so downstream slicing can still sort user interactions.
    if "end_time" not in df.columns:
        df["end_time"] = np.arange(len(df))

    # Normalize any array-like cells to scalars so downstream set()/dict
    # operations work as expected.
    def _coerce_scalar(val):
        if isinstance(val, (list, tuple, np.ndarray)):
            arr = np.array(val).ravel()
            if arr.size == 0:
                return np.nan
            return arr[0]
        return val

    for col in df.columns:
        df[col] = df[col].apply(_coerce_scalar)

    # Ensure correctness values are numeric
    if "correct" in df.columns:
        df["correct"] = pd.to_numeric(df["correct"], errors="coerce")
    return df


def _read_input(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return _read_csv(path)
    if ext == ".npz":
        return _read_npz(path)
    raise ValueError(f"Unsupported file extension '{ext}'. Use .csv or .npz")


def _ensure_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def _parse_ts(val):
        if isinstance(val, (list, tuple, np.ndarray)):
            arr = np.array(val).flatten()
            if arr.size == 0:
                return np.nan
            val = arr[0]
        if pd.isna(val):
            return np.nan
        if isinstance(val, (int, float, np.integer, np.floating)):
            return float(val)
        val_str = str(val)
        try:
            return time.mktime(time.strptime(val_str[:19], "%Y-%m-%d %H:%M:%S"))
        except Exception:
            try:
                return pd.to_datetime(val_str).timestamp()
            except Exception:
                return np.nan

    timestamps = df["end_time"].apply(_parse_ts)
    if timestamps.isna().any():
        if timestamps.dropna().empty:
            timestamps = pd.Series(np.arange(len(df)), index=df.index, dtype=float)
        else:
            fallback = pd.Series(np.arange(len(df)), index=df.index, dtype=float)
            timestamps = timestamps.fillna(fallback)
    df["timestamp"] = timestamps
    return df


def _build_id_maps(df: pd.DataFrame) -> Tuple[Dict, Dict, Dict]:
    users = sorted(set(df["user_id"]))
    problems = sorted(set(df["problem_id"]))
    skills = sorted(set(df["skill_id"]))

    user2id = {u: i + 1 for i, u in enumerate(users)}
    problem2id = {p: i + 1 for i, p in enumerate(problems)}
    skill2id = {s: i for i, s in enumerate(skills)}
    return user2id, problem2id, skill2id


def _compute_difficulty(values: Iterable[float]) -> int:
    avg = int(np.mean(list(values)) * 100) + 1
    return avg


def _build_difficulties(df: pd.DataFrame) -> Tuple[Dict, Dict, np.ndarray, np.ndarray]:
    sdifficult2id: Dict = {}
    difficult2id: Dict = {}
    nones = []
    nonesk = []

    for skill, group in tqdm(df.groupby("skill_id"), desc="Skill difficulty"):
        if len(group) < 30 or group.empty:
            sdifficult2id[skill] = 1.02
            nonesk.append(skill)
            continue
        sdifficult2id[skill] = _compute_difficulty(group["correct"].values)

    for prob, group in tqdm(df.groupby("problem_id"), desc="Question difficulty"):
        if len(group) < 30 or group.empty:
            difficult2id[prob] = 1.02
            nones.append(prob)
            continue
        difficult2id[prob] = _compute_difficulty(group["correct"].values)

    return sdifficult2id, difficult2id, np.array(nones), np.array(nonesk)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess a raw CSV/NPZ file like data_pre.py + data_save.py (outputs in data/)"
        )
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        help="Path to raw CSV or NPZ file (positional argument)",
    )
    parser.add_argument(
        "--input_csv",
        dest="input_flag",
        help="Optional alternative to the positional input path",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=100,
        help="Maximum sequence length when slicing user interactions (default: 100)",
    )
    parser.add_argument("--output_dir", default="data", help=argparse.SUPPRESS)
    args = parser.parse_args()

    input_path = args.input_path or args.input_flag
    if not input_path:
        parser.error("Please provide an input file as a positional argument or with --input_csv")
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    raw_df = _read_input(input_path)
    raw_df["skill_id"].fillna("nan", inplace=True)
    raw_df = raw_df[raw_df["skill_id"] != "nan"].reset_index(drop=True)
    raw_df = _ensure_timestamp(raw_df)

    user2id, problem2id, skill2id = _build_id_maps(raw_df)
    sdifficult2id, difficult2id, nones, nonesk = _build_difficulties(raw_df)
    with open(os.path.join(output_dir, "user2id"), "w", encoding="utf-8") as f:
        f.write(str(user2id))
    with open(os.path.join(output_dir, "problem2id"), "w", encoding="utf-8") as f:
        f.write(str(problem2id))
    with open(os.path.join(output_dir, "skill2id"), "w", encoding="utf-8") as f:
        f.write(str(skill2id))
    with open(os.path.join(output_dir, "difficult2id"), "w", encoding="utf-8") as f:
        f.write(str(difficult2id))
    with open(os.path.join(output_dir, "sdifficult2id"), "w", encoding="utf-8") as f:
        f.write(str(sdifficult2id))

    np.save(os.path.join(output_dir, "nones.npy"), nones)
    np.save(os.path.join(output_dir, "nonesk.npy"), nonesk)

    # Build train/valid/test splits mirroring data_save.py
    seq_len = args.seq_len
    user_ids = np.array(sorted(list(set(raw_df["user_id"]))))
    np.random.seed(100)
    np.random.shuffle(user_ids)
    train_users, test_users = train_test_split(user_ids, test_size=0.2, random_state=5)

    def _slice_users(user_pool, nones_arr, nonesk_arr, filter_low_freq: bool = True):
        sequences = []
        for uid in tqdm(user_pool, desc="Slicing users"):
            user_rows = raw_df[raw_df.user_id == uid].sort_values(by=["timestamp"])
            temp = np.array(user_rows[["user_id", "problem_id", "correct", "skill_id", "timestamp"]])
            if len(temp) < 2:
                continue
            train_q = []
            train_d = []
            train_a = []
            train_skill = []
            train_sd = []
            for row in temp:
                if filter_low_freq and (row[1] in nones_arr or row[3] in nonesk_arr):
                    continue
                train_q.append(problem2id[row[1]])
                train_d.append(difficult2id[row[1]])
                train_a.append(int(row[2]))
                train_skill.append(skill2id[row[3]])
                train_sd.append(sdifficult2id[row[3]])
                if len(train_q) >= seq_len:
                    sequences.append([train_q, train_d, train_a, train_skill, len(train_q), train_sd])
                    train_q, train_d, train_a, train_skill, train_sd = [], [], [], [], []
            if len(train_q) >= 2 and len(train_q) < seq_len:
                sequences.append([train_q, train_d, train_a, train_skill, len(train_q), train_sd])
        np.random.seed(2)
        np.random.shuffle(sequences)
        return np.array(sequences, dtype=object)

    kfold = KFold(n_splits=5, shuffle=True, random_state=5)
    count = 0
    for train_index, valid_index in kfold.split(train_users):
        train_id = train_users[train_index]
        valid_id = train_users[valid_index]
        np.random.shuffle(train_id)
        train_set = _slice_users(train_id, nones, nonesk, filter_low_freq=True)
        valid_set = _slice_users(valid_id, nones, nonesk, filter_low_freq=True)

        # If filtering drops all sequences (common in very small or sparse
        # datasets), retry without removing low-frequency problems/skills so
        # training can proceed with default difficulty values.
        if len(train_set) == 0 or len(valid_set) == 0:
            train_set = _slice_users(train_id, nones, nonesk, filter_low_freq=False)
            valid_set = _slice_users(valid_id, nones, nonesk, filter_low_freq=False)
            if len(train_set) == 0 or len(valid_set) == 0:
                raise ValueError(
                    "No training/validation sequences could be generated. Ensure each user has at least two interactions "
                    "and try reducing --seq_len if the dataset is very small."
                )
        np.save(os.path.join(output_dir, f"train{count}.npy"), np.array(train_set))
        np.save(os.path.join(output_dir, f"valid{count}.npy"), np.array(valid_set))
        count += 1
        break

    test_set = _slice_users(test_users, nones, nonesk, filter_low_freq=True)
    if len(test_set) == 0:
        test_set = _slice_users(test_users, nones, nonesk, filter_low_freq=False)
        if len(test_set) == 0:
            raise ValueError(
                "No test sequences could be generated. Confirm the dataset includes at least two interactions per test user."
            )
    np.save(os.path.join(output_dir, "test.npy"), np.array(test_set))

    print("complete")


if __name__ == "__main__":
    main()
