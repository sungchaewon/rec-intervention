"""SASRec (Kang & McAuley, ICDM 2018) trained with full cross-entropy.

Full softmax CE over the catalog instead of the original one-negative BCE
follows Klenitskiy & Vasilev (RecSys 2023), which shows it is substantially
stronger. Item index i is embedded as token i + 1; token 0 is padding.

Protocol: `fit(train)` holds out each user's last train item as validation for
early stopping (best-epoch weights are kept), then scores test users from their
full train sequence. The test item is never seen during fitting.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from recint.data.interactions import InteractionData, leave_last_out
from recint.metrics.ranking import ndcg_at_k
from recint.models.base import Recommender, mask_seen_items, top_k_items
from recint.utils.device import resolve_device

logger = logging.getLogger(__name__)

PAD_TOKEN = 0
ITEM_TOKEN_OFFSET = 1
ARCHITECTURE_PARAMS = ("max_len", "hidden_dim", "n_blocks", "n_heads")
EMBEDDING_INIT_STD = 0.02


def pad_left(sequences: list[np.ndarray], max_len: int) -> np.ndarray:
    """(n, max_len) token matrix keeping the last `max_len` items, left-padded."""
    tokens = np.full((len(sequences), max_len), PAD_TOKEN, dtype=np.int64)
    for row, sequence in enumerate(sequences):
        tail = sequence[-max_len:]
        if len(tail):
            tokens[row, -len(tail):] = tail + ITEM_TOKEN_OFFSET
    return tokens


def next_item_training_pairs(sequences: list[np.ndarray], max_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Inputs s[:-1] and targets s[1:] per user with >= 2 items."""
    usable = [s for s in sequences if len(s) >= 2]
    return pad_left([s[:-1] for s in usable], max_len), pad_left([s[1:] for s in usable], max_len)


class SASRecNetwork(nn.Module):
    def __init__(
        self, n_items: int, max_len: int, hidden_dim: int, n_blocks: int, n_heads: int, dropout: float
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.embedding_scale = math.sqrt(hidden_dim)
        self.item_embedding = nn.Embedding(n_items + ITEM_TOKEN_OFFSET, hidden_dim, padding_idx=PAD_TOKEN)
        self.position_embedding = nn.Embedding(max_len, hidden_dim)
        self.input_dropout = nn.Dropout(dropout)
        block = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, n_blocks, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(hidden_dim)
        # PyTorch's default N(0, 1) embeddings make initial logits ~ sqrt(hidden_dim)
        # in scale (CE loss far above log(n_items)); use the usual small init.
        for embedding in (self.item_embedding, self.position_embedding):
            nn.init.normal_(embedding.weight, std=EMBEDDING_INIT_STD)
        with torch.no_grad():
            self.item_embedding.weight[PAD_TOKEN].zero_()

    def _attention_mask(self, is_pad: torch.Tensor) -> torch.Tensor:
        """Block future positions and padding keys, but always allow attending to self.

        Keeping the diagonal avoids fully-masked rows (NaNs) for padding queries,
        whose outputs are never used.
        """
        length = is_pad.size(1)
        future = torch.ones(length, length, dtype=torch.bool, device=is_pad.device).triu(1)
        self_only = torch.eye(length, dtype=torch.bool, device=is_pad.device)
        blocked = (future | is_pad[:, None, :]) & ~self_only
        return blocked.repeat_interleave(self.n_heads, dim=0)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """(batch, length) tokens -> (batch, length, hidden) states."""
        is_pad = tokens == PAD_TOKEN
        positions = torch.arange(tokens.size(1), device=tokens.device)
        hidden = self.item_embedding(tokens) * self.embedding_scale + self.position_embedding(positions)
        hidden = self.input_dropout(hidden) * ~is_pad.unsqueeze(-1)
        hidden = self.encoder(hidden, mask=self._attention_mask(is_pad))
        return self.output_norm(hidden)

    def item_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Scores for every real item (padding token excluded)."""
        return hidden @ self.item_embedding.weight[ITEM_TOKEN_OFFSET:].T


class SASRecRecommender(Recommender):
    def __init__(
        self,
        max_len: int,
        hidden_dim: int,
        n_blocks: int,
        n_heads: int,
        dropout: float,
        learning_rate: float,
        weight_decay: float,
        batch_size: int,
        max_epochs: int,
        patience: int,
        eval_k: int,
        eval_batch_size: int,
        device: str,
    ) -> None:
        if hidden_dim % n_heads:
            raise ValueError("hidden_dim must be divisible by n_heads")
        self.params = {
            "max_len": max_len, "hidden_dim": hidden_dim, "n_blocks": n_blocks, "n_heads": n_heads,
            "dropout": dropout, "learning_rate": learning_rate, "weight_decay": weight_decay,
            "batch_size": batch_size, "max_epochs": max_epochs, "patience": patience,
            "eval_k": eval_k, "eval_batch_size": eval_batch_size, "device": device,
        }  # fmt: skip
        self.device = resolve_device(device)
        self.history: list[dict[str, float]] = []
        self._network: SASRecNetwork | None = None
        self._train_tokens: np.ndarray | None = None
        self._train_fingerprint: str | None = None

    def _build_network(self, n_items: int) -> SASRecNetwork:
        p = self.params
        return SASRecNetwork(
            n_items, p["max_len"], p["hidden_dim"], p["n_blocks"], p["n_heads"], p["dropout"]
        ).to(self.device)

    @torch.no_grad()
    def _score_tokens(self, tokens: np.ndarray) -> np.ndarray:
        assert self._network is not None
        self._network.eval()
        scores = []
        for start in range(0, len(tokens), self.params["eval_batch_size"]):
            batch = torch.from_numpy(tokens[start : start + self.params["eval_batch_size"]]).to(self.device)
            last_hidden = self._network(batch)[:, -1]
            scores.append(self._network.item_logits(last_hidden).float().cpu().numpy())
        return np.concatenate(scores).astype(np.float64)

    def _validation_ndcg(self, inner_train: InteractionData, users: np.ndarray, targets: np.ndarray, tokens: np.ndarray) -> float:
        scores = self._score_tokens(tokens)
        mask_seen_items(scores, inner_train.user_item_matrix(), users)
        ranked = top_k_items(scores, self.params["eval_k"])
        return float(ndcg_at_k(ranked, targets, self.params["eval_k"]).mean())

    def fit(self, train: InteractionData) -> SASRecRecommender:
        p = self.params
        inner = leave_last_out(train, min_train_interactions=1)
        inner_sequences = inner.train.user_sequences()
        inputs, targets = next_item_training_pairs(inner_sequences, p["max_len"])
        valid_tokens = pad_left([inner_sequences[u] for u in inner.eval_users], p["max_len"])
        inputs_t, targets_t = torch.from_numpy(inputs), torch.from_numpy(targets)

        self._network = self._build_network(train.n_items)
        optimizer = torch.optim.Adam(
            self._network.parameters(), lr=p["learning_rate"], weight_decay=p["weight_decay"]
        )
        best_ndcg, best_state, epochs_without_improvement = -1.0, None, 0
        self.history = []
        for epoch in range(1, p["max_epochs"] + 1):
            started = time.perf_counter()
            self._network.train()
            order = torch.randperm(len(inputs_t))
            total_loss, total_targets = 0.0, 0
            for start in range(0, len(order), p["batch_size"]):
                batch = order[start : start + p["batch_size"]]
                batch_inputs = inputs_t[batch].to(self.device)
                batch_targets = targets_t[batch].to(self.device)
                is_target = batch_targets != PAD_TOKEN
                logits = self._network.item_logits(self._network(batch_inputs)[is_target])
                loss = F.cross_entropy(logits, batch_targets[is_target] - ITEM_TOKEN_OFFSET)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * int(is_target.sum())
                total_targets += int(is_target.sum())

            valid_ndcg = self._validation_ndcg(inner.train, inner.eval_users, inner.targets, valid_tokens)
            record = {
                "epoch": epoch,
                "train_loss": total_loss / max(total_targets, 1),
                f"valid_ndcg@{p['eval_k']}": valid_ndcg,
                "seconds": time.perf_counter() - started,
            }
            self.history.append(record)
            logger.info(
                "epoch %d  loss %.4f  valid ndcg@%d %.4f  (%.1fs)",
                epoch, record["train_loss"], p["eval_k"], valid_ndcg, record["seconds"],
            )  # fmt: skip
            if valid_ndcg > best_ndcg:
                best_ndcg, epochs_without_improvement = valid_ndcg, 0
                best_state = {k: v.detach().clone() for k, v in self._network.state_dict().items()}
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= p["patience"]:
                    logger.info("early stopping at epoch %d (best valid ndcg %.4f)", epoch, best_ndcg)
                    break

        assert best_state is not None
        self._network.load_state_dict(best_state)
        self._set_train_data(train)
        return self

    def _set_train_data(self, train: InteractionData) -> None:
        self._train_tokens = pad_left(train.user_sequences(), self.params["max_len"])
        self._train_fingerprint = train.fingerprint()

    def score(self, user_indices: np.ndarray) -> np.ndarray:
        if self._network is None or self._train_tokens is None:
            raise RuntimeError("Call fit() or load() before score()")
        return self._score_tokens(self._train_tokens[user_indices])

    def save(self, path: Path) -> None:
        if self._network is None:
            raise RuntimeError("Call fit() before save()")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "params": self.params,
                "n_items": self._network.item_embedding.num_embeddings - ITEM_TOKEN_OFFSET,
                "train_fingerprint": self._train_fingerprint,
                "state_dict": {k: v.cpu() for k, v in self._network.state_dict().items()},
                "history": self.history,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, train: InteractionData, **params: Any) -> SASRecRecommender:
        """Architecture comes from `params` (the config) and must match the checkpoint;
        runtime settings such as `device` may differ."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        mismatched = {
            k: (checkpoint["params"][k], params[k])
            for k in ARCHITECTURE_PARAMS
            if checkpoint["params"][k] != params[k]
        }
        if mismatched:
            raise ValueError(f"Config does not match checkpoint architecture (checkpoint, config): {mismatched}")
        if checkpoint["train_fingerprint"] != train.fingerprint():
            raise ValueError(f"{path} was trained on different data or a different split")
        model = cls(**params)
        model._network = model._build_network(checkpoint["n_items"])
        model._network.load_state_dict(checkpoint["state_dict"])
        model.history = checkpoint["history"]
        model._set_train_data(train)
        return model
