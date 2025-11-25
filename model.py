# -*- coding:utf-8 -*-
"""PyTorch implementation of the DIM model.

The original TensorFlow v1 graph used custom gating operations. This module
re-creates the same computations using PyTorch so that checkpoints can be saved
in ``.pt`` format.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DIM(nn.Module):
    """Difficulty-Influenced Memory model in PyTorch."""

    def __init__(self, num_steps: int, num_skills: int, hidden_size: int, dropout: float = 0.8):
        super().__init__()
        self.num_steps = num_steps
        self.num_skills = num_skills
        self.hidden_size = hidden_size
        self.dropout = dropout

        # Embeddings (padding index 0 keeps parity with the original TF code)
        self.problem_embed = nn.Embedding(53092, hidden_size, padding_idx=0)
        self.sd_embed = nn.Embedding(1011, hidden_size, padding_idx=0)
        self.difficulty_embed = nn.Embedding(1011, hidden_size, padding_idx=0)
        self.answer_embed = nn.Embedding(2, hidden_size)

        # Project KC one-hot vectors to hidden dimension
        self.skill_proj = nn.Linear(num_skills, hidden_size)
        self.target_skill_proj = nn.Linear(num_skills, hidden_size)

        # Shared linear layers for gating operations
        self.linear_l = nn.Linear(hidden_size, hidden_size)
        self.linear_c = nn.Linear(hidden_size, hidden_size)
        self.linear_o = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.linear_q = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.linear_1 = nn.Linear(hidden_size * 4, hidden_size * 4)

        # Output projection for target features
        self.input_proj = nn.Linear(hidden_size * 4, hidden_size)
        self.target_proj = nn.Linear(hidden_size * 4, hidden_size)

        # Knowledge matrix replicated per batch in the original graph
        self.knowledge_w = nn.Parameter(torch.empty(1, hidden_size))
        nn.init.xavier_uniform_(self.knowledge_w)

    def forward(self, batch):
        # Unpack batch
        input_p = batch["input_p"]
        target_p = batch["target_p"]
        input_sd = batch["input_sd"]
        input_d = batch["input_d"]
        input_kc = batch["input_kc"]
        x_answer = batch["x_answer"]
        target_sd = batch["target_sd"]
        target_d = batch["target_d"]
        target_kc = batch["target_kc"]
        target_index = batch["target_index"]

        batch_size = input_p.size(0)

        # Embeddings
        p_embedding = self.problem_embed(input_p)
        target_p_emb = self.problem_embed(target_p)
        sd_embedding = self.sd_embed(input_sd)
        target_sd_emb = self.sd_embed(target_sd)
        diff_embedding = self.difficulty_embed(input_d)
        target_diff_emb = self.difficulty_embed(target_d)
        ans_embedding = self.answer_embed(x_answer)

        skill_embedding = self.skill_proj(input_kc)
        target_skill_emb = self.target_skill_proj(target_kc)

        # Build input/target features
        input_data = torch.cat(
            [p_embedding, skill_embedding, sd_embedding, diff_embedding], dim=-1
        )
        input_data = self.input_proj(input_data)

        target_data = torch.cat(
            [target_p_emb, target_skill_emb, target_sd_emb, target_diff_emb], dim=-1
        )
        target_data = self.target_proj(target_data)

        # Knowledge initialization
        kkk = self.knowledge_w.expand(batch_size, -1)

        outputs = []
        for i in range(self.num_steps):
            sd_i = sd_embedding[:, i, :]
            aa_i = ans_embedding[:, i, :]
            dd_i = diff_embedding[:, i, :]
            q1_i = input_data[:, i, :]

            q = kkk - q1_i
            input_gates = torch.sigmoid(self.linear_l(q))
            c_title = torch.tanh(self.linear_c(q))
            ccc = input_gates * F.dropout(c_title, p=self.dropout, training=self.training)

            x = torch.cat([ccc, aa_i], dim=-1)
            xx = torch.sigmoid(self.linear_o(x))
            xx_title = torch.tanh(self.linear_q(x))
            xx = xx * xx_title

            ins = torch.cat([kkk, aa_i, sd_i, dd_i], dim=-1)
            ooo = torch.sigmoid(self.linear_1(ins))
            kkk = ooo[:, : self.hidden_size] * kkk + (1 - ooo[:, : self.hidden_size]) * xx[:, : self.hidden_size]

            outputs.append(kkk.unsqueeze(1))

        output = torch.cat(outputs, dim=1)
        logits = torch.sum(target_data * output, dim=-1).reshape(-1)
        selected_logits = logits[target_index]
        pred = torch.sigmoid(selected_logits)
        return pred, selected_logits

    def compute_loss(self, batch):
        pred, logits = self.forward(batch)
        labels = batch["target_correctness"]
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        l2 = sum(param.pow(2).sum() for param in self.parameters()) * 1e-6
        return bce + l2, pred
