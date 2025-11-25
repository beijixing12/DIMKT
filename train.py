# -*- coding:utf-8 -*-
"""PyTorch training script for DIM.

This script mirrors the original TensorFlow workflow but trains a PyTorch model
and saves checkpoints in ``.pt`` format.
"""
import argparse
import os
from datetime import datetime
from math import sqrt
import json

import numpy as np
import torch
from sklearn import metrics
from sklearn.metrics import mean_squared_error

from model import DIM


def build_batch(students, start, batch_size, max_steps, num_skills, device):
    input_p = np.zeros((batch_size, max_steps), dtype=np.int64)
    target_p = np.zeros((batch_size, max_steps), dtype=np.int64)
    input_sd = np.zeros((batch_size, max_steps), dtype=np.int64)
    input_d = np.zeros((batch_size, max_steps), dtype=np.int64)
    input_kc = np.zeros((batch_size, max_steps, num_skills), dtype=np.float32)
    x_answer = np.zeros((batch_size, max_steps), dtype=np.int64)
    target_d = np.zeros((batch_size, max_steps), dtype=np.int64)
    target_sd = np.zeros((batch_size, max_steps), dtype=np.int64)
    target_kc = np.zeros((batch_size, max_steps, num_skills), dtype=np.float32)
    target_correctness = []
    target_index = []
    actual_labels = []

    for i in range(batch_size):
        student = students[start + i]
        ppp, problem_ids, correctness, problem_kcs, len_seq, i_s = student
        for j in range(len_seq - 1):
            input_sd[i, j] = i_s[j]
            input_p[i, j] = ppp[j]
            input_d[i, j] = problem_ids[j]
            input_kc[i, j, int(problem_kcs[j])] = 1
            x_answer[i, j] = correctness[j]

            target_sd[i, j] = i_s[j + 1]
            target_p[i, j] = ppp[j + 1]
            target_d[i, j] = problem_ids[j + 1]
            target_kc[i, j, int(problem_kcs[j + 1])] = 1
            target_index.append(i * max_steps + j)
            target_correctness.append(int(correctness[j + 1]))
            actual_labels.append(int(correctness[j + 1]))

    batch_tensors = {
        "input_p": torch.tensor(input_p, device=device),
        "target_p": torch.tensor(target_p, device=device),
        "input_sd": torch.tensor(input_sd, device=device),
        "input_d": torch.tensor(input_d, device=device),
        "input_kc": torch.tensor(input_kc, device=device),
        "x_answer": torch.tensor(x_answer, device=device),
        "target_sd": torch.tensor(target_sd, device=device),
        "target_d": torch.tensor(target_d, device=device),
        "target_kc": torch.tensor(target_kc, device=device),
        "target_index": torch.tensor(target_index, dtype=torch.long, device=device),
        "target_correctness": torch.tensor(target_correctness, dtype=torch.float32, device=device),
    }
    return batch_tensors, actual_labels


def run_epoch(model, optimizer, students, batch_size, max_steps, num_skills, device, train=True):
    data_size = len(students)
    index = 0
    actual_labels = []
    pred_labels = []

    if train:
        np.random.shuffle(students)
        model.train()
    else:
        model.eval()

    while index < data_size:
        current_bs = min(batch_size, data_size - index)
        batch, labels = build_batch(students, index, current_bs, max_steps, num_skills, device)
        index += current_bs

        if train:
            optimizer.zero_grad()
            loss, pred = model.compute_loss(batch)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                loss, pred = model.compute_loss(batch)

        if labels:
            actual_labels.extend(labels)
            pred_labels.extend(pred.detach().cpu().numpy().tolist())

    if not actual_labels:
        return float("nan"), float("nan"), float("nan")

    rmse = sqrt(mean_squared_error(actual_labels, pred_labels))
    auc = metrics.roc_auc_score(actual_labels, pred_labels)
    pred_score = np.greater_equal(pred_labels, 0.5).astype(int)
    acc = np.mean(np.equal(actual_labels, pred_score).astype(int))

    return rmse, auc, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fold", help="Fold number used to pick train/valid split")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.002)
    parser.add_argument("--checkpoint_dir", default="runs_pytorch")
    parser.add_argument(
        "--data_dir",
        default="data",
        help="Directory containing processed train/valid/test .npy files and meta.json",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_path = os.path.join(args.data_dir, f"train{args.fold}.npy")
    valid_path = os.path.join(args.data_dir, f"valid{args.fold}.npy")
    test_path = os.path.join(args.data_dir, "test.npy")
    train_students = np.load(train_path, allow_pickle=True)
    valid_students = np.load(valid_path, allow_pickle=True)
    test_students = np.load(test_path, allow_pickle=True)

    max_num_steps = 100
    max_num_skills = 265
    meta_path = os.path.join(args.data_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        max_num_steps = int(meta.get("max_seq_len", max_num_steps))
        max_num_skills = int(meta.get("num_skills", max_num_skills))

    model = DIM(max_num_steps, max_num_skills, args.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(args.checkpoint_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    best_auc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_rmse, train_auc, train_acc = run_epoch(
            model,
            optimizer,
            train_students,
            args.batch_size,
            max_num_steps,
            max_num_skills,
            device,
            train=True,
        )
        valid_rmse, valid_auc, valid_acc = run_epoch(
            model,
            optimizer,
            valid_students,
            args.batch_size,
            max_num_steps,
            max_num_skills,
            device,
            train=False,
        )

        print(
            f"Epoch {epoch}: Train RMSE {train_rmse:.4f} AUC {train_auc:.4f} ACC {train_acc:.4f} | "
            f"Valid RMSE {valid_rmse:.4f} AUC {valid_auc:.4f} ACC {valid_acc:.4f}"
        )

        if valid_auc > best_auc:
            best_auc = valid_auc
            checkpoint_path = os.path.join(run_dir, "best_model.pt")
            torch.save({"model_state": model.state_dict(), "args": vars(args)}, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")

    test_rmse, test_auc, test_acc = run_epoch(
        model,
        optimizer,
        test_students,
        args.batch_size,
        max_num_steps,
        max_num_skills,
        device,
        train=False,
    )
    print(f"Test: RMSE {test_rmse:.4f} AUC {test_auc:.4f} ACC {test_acc:.4f}")


if __name__ == "__main__":
    main()
