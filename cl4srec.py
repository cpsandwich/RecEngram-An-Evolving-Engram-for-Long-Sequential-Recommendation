"""CL4SRec: Contrastive Learning for Sequential Recommendation (Xie et al., SIGIR 2022).

SASRec base + 3 data augmentations (crop/mask/reorder) + InfoNCE loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import PositionalEncoding, TransformerBlock


class CL4SRec(nn.Module):
    def __init__(
        self,
        item_vocab_size: int,
        hidden_dim: int = 128,
        max_seq_len: int = 500,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.2,
        cl_weight: float = 0.1,
        temperature: float = 0.07,
        aug_crop_ratio: float = 0.3,
        aug_mask_ratio: float = 0.3,
        aug_reorder_ratio: float = 0.3,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cl_weight = cl_weight
        self.temperature = temperature
        self.aug_crop_ratio = aug_crop_ratio
        self.aug_mask_ratio = aug_mask_ratio
        self.aug_reorder_ratio = aug_reorder_ratio

        self.item_emb = nn.Embedding(item_vocab_size, hidden_dim, padding_idx=0)
        self.pos_enc = PositionalEncoding(hidden_dim, max_seq_len)
        self.emb_dropout = nn.Dropout(dropout)
        self.emb_norm = nn.LayerNorm(hidden_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.cl_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.item_emb.weight)

    def _encode(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encode a sequence, return user representation (last valid position)."""
        B, L = seq.shape
        x = self.emb_dropout(self.emb_norm(self.pos_enc(self.item_emb(seq))))
        causal = torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool))
        key_mask = mask.unsqueeze(1)
        attn_mask = causal.unsqueeze(0) & key_mask

        for block in self.blocks:
            x = block(x, attn_mask)

        lengths = mask.sum(dim=-1).clamp(min=1) - 1
        return x[torch.arange(B, device=x.device), lengths]

    def _augment_crop(self, seq: torch.Tensor, mask: torch.Tensor) -> tuple:
        B, L = seq.shape
        lengths = mask.sum(dim=-1).clamp(min=1).long()
        aug_seq = seq.clone()
        aug_mask = mask.clone()
        for b in range(B):
            length = lengths[b].item()
            if length <= 2:
                continue
            crop_len = max(1, int(length * self.aug_crop_ratio * torch.rand(1).item()))
            start = torch.randint(0, length - crop_len, (1,)).item()
            aug_seq[b, start:start + crop_len] = 0
            aug_mask[b, start:start + crop_len] = 0
        return aug_seq, aug_mask

    def _augment_mask(self, seq: torch.Tensor, mask: torch.Tensor) -> tuple:
        B, L = seq.shape
        lengths = mask.sum(dim=-1).clamp(min=1).long()
        aug_seq = seq.clone()
        aug_mask = mask.clone()
        for b in range(B):
            length = lengths[b].item()
            if length <= 1:
                continue
            num_mask = max(1, int(length * self.aug_mask_ratio * torch.rand(1).item()))
            positions = torch.randperm(length)[:num_mask]
            aug_seq[b, positions] = 0
        return aug_seq, aug_mask

    def _augment_reorder(self, seq: torch.Tensor, mask: torch.Tensor) -> tuple:
        B, L = seq.shape
        lengths = mask.sum(dim=-1).clamp(min=1).long()
        aug_seq = seq.clone()
        aug_mask = mask.clone()
        for b in range(B):
            length = lengths[b].item()
            if length <= 3:
                continue
            reorder_len = max(2, int(length * self.aug_reorder_ratio * torch.rand(1).item()))
            start = torch.randint(0, length - reorder_len, (1,)).item()
            perm = torch.randperm(reorder_len, device=seq.device)
            aug_seq[b, start:start + reorder_len] = seq[b, start:start + reorder_len][perm]
        return aug_seq, aug_mask

    def _augment(self, seq: torch.Tensor, mask: torch.Tensor) -> tuple:
        """Apply all 3 augmentations randomly on the same sequence."""
        B, L = seq.shape
        lengths = mask.sum(dim=-1).clamp(min=1).long()
        aug_seq = seq.clone()
        aug_mask = mask.clone()

        for b in range(B):
            length = lengths[b].item()

            if length <= 2:
                continue

            if torch.rand(1).item() < 0.5:
                crop_len = max(1, int(length * self.aug_crop_ratio * torch.rand(1).item()))
                if crop_len < length:
                    start = torch.randint(0, length - crop_len, (1,)).item()
                    aug_seq[b, start:start + crop_len] = 0
                    aug_mask[b, start:start + crop_len] = 0

            if length > 1 and torch.rand(1).item() < 0.5:
                active_positions = aug_mask[b].nonzero(as_tuple=True)[0]
                if len(active_positions) > 1:
                    num_mask = max(1, int(len(active_positions) * self.aug_mask_ratio * torch.rand(1).item()))
                    num_mask = min(num_mask, len(active_positions))
                    idx = torch.randperm(len(active_positions))[:num_mask]
                    aug_seq[b, active_positions[idx]] = 0

            if length > 3 and torch.rand(1).item() < 0.5:
                active_positions = aug_mask[b].nonzero(as_tuple=True)[0]
                if len(active_positions) >= 2:
                    reorder_len = max(2, int(len(active_positions) * self.aug_reorder_ratio * torch.rand(1).item()))
                    reorder_len = min(reorder_len, len(active_positions))
                    start_idx = torch.randint(0, len(active_positions) - reorder_len, (1,)).item()
                    positions = active_positions[start_idx:start_idx + reorder_len]
                    perm = torch.randperm(reorder_len, device=seq.device)
                    aug_seq[b, positions] = seq[b, positions][perm]

        return aug_seq, aug_mask

    def forward(self, batch: dict) -> torch.Tensor:
        seq = batch["seq"]
        mask = batch["mask"]

        h_user = self._encode(seq, mask)

        pos = self.item_emb(batch["target"])
        neg = self.item_emb(batch["negatives"])
        pos_scores = (h_user * pos).sum(dim=-1, keepdim=True)
        neg_scores = torch.einsum("bd,bnd->bn", h_user, neg)
        return torch.cat([pos_scores, neg_scores], dim=-1)

    def compute_cl_loss(self, batch: dict) -> torch.Tensor:
        """Compute contrastive InfoNCE loss over augmented views."""
        seq = batch["seq"]
        mask = batch["mask"]
        B = seq.shape[0]

        aug_seq, aug_mask = self._augment(seq, mask)

        h_orig = self._encode(seq, mask)
        h_aug = self._encode(aug_seq, aug_mask)

        h_orig = self.cl_proj(h_orig)
        h_aug = self.cl_proj(h_aug)

        h_orig = F.normalize(h_orig, dim=-1)
        h_aug = F.normalize(h_aug, dim=-1)

        sim = torch.matmul(h_orig, h_aug.T) / self.temperature
        labels = torch.arange(B, device=sim.device)
        loss_12 = F.cross_entropy(sim, labels)
        loss_21 = F.cross_entropy(sim.T, labels)
        return (loss_12 + loss_21) * 0.5
