# -*- coding:utf-8 -*-
"""Evaluate a saved PyTorch DIM checkpoint on the test set."""
import argparse
import json
import os
import numpy as np
import torch

from model import DIM
from train import build_batch, run_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to a .pt checkpoint saved by train.py")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--fold", default="1", help="Fold number to align with train/valid files")
    parser.add_argument(
        "--data_dir",
        default="data",
        help="Directory containing processed train/valid/test .npy files and meta.json",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    ckpt_args = checkpoint.get("args", {})
    hidden_size = ckpt_args.get("hidden_size", args.hidden_size)
    fold = ckpt_args.get("fold", args.fold)

    max_num_steps = 100
    max_num_skills = 265
    meta_path = os.path.join(args.data_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        max_num_steps = int(meta.get("max_seq_len", max_num_steps))
        max_num_skills = int(meta.get("num_skills", max_num_skills))

    model = DIM(max_num_steps, max_num_skills, hidden_size).to(device)
    model.load_state_dict(checkpoint["model_state"])

    train_students = np.load(os.path.join(args.data_dir, f"train{fold}.npy"), allow_pickle=True)
    valid_students = np.load(os.path.join(args.data_dir, f"valid{fold}.npy"), allow_pickle=True)
    test_students = np.load(os.path.join(args.data_dir, "test.npy"), allow_pickle=True)

    dummy_opt = torch.optim.Adam(model.parameters(), lr=0.0)

    print("Running evaluation with checkpoint:", args.checkpoint)
    for split_name, data in [
        ("Train", train_students),
        ("Validation", valid_students),
        ("Test", test_students),
    ]:
        rmse, auc, acc = run_epoch(
            model,
            dummy_opt,
            data,
            args.batch_size,
            max_num_steps,
            max_num_skills,
            device,
            train=False,
        )
        print(f"{split_name}: RMSE {rmse:.4f} AUC {auc:.4f} ACC {acc:.4f}")


if __name__ == "__main__":
    main()
