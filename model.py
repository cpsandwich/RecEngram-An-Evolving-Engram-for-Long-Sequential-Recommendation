"""RecEngram: Dynamic Memory for Sequential Recommendation.

Combines 3 mutually-reinforcing mechanisms:
1. Target-Guided Memory Activation (bilinear W)
2. Surprise-Driven Memory Evolution (three-tier update + MI critic)
3. Hierarchical Long-Short Term Memory (dual bank + gate)

Paper Tables 1-9. Ablation via use_* flags.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedding import InputEmbedding
from .engram import EngramMemory
from .target_guided import TargetGuidedActivation
from .surprise import SurpriseEvolution, MICritic, variational_mi_lower_bound
from .hierarchical import HierarchicalMemory
from ..common import TransformerBlock


class RecEngram(nn.Module):
    def __init__(
        self,
        item_vocab_size: int,
        hidden_dim: int = 128,
        max_seq_len: int = 500,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.2,
        num_memory_slots: int = 128,
        num_long_slots: int = 32,
        num_short_slots: int = 96,
        write_strength: float = 0.1,
        window_size: int = 20,
        surprise_low_percentile: float = 0.30,
        surprise_high_percentile: float = 0.70,
        surprise_window: int = 5000,
        long_update_interval: int = 5,
        critic_hidden_dim: int = 256,
        mi_loss_weight: float = 0.1,
        use_target_guided: bool = True,
        use_surprise: bool = True,
        use_hierarchy: bool = True,
        category_vocab_size: int = 5000,
        price_vocab_size: int = 10,
        brand_vocab_size: int = 50000,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_memory_slots = num_memory_slots
        self.mi_loss_weight = mi_loss_weight
        self.use_target_guided = use_target_guided
        self.use_surprise = use_surprise
        self.use_hierarchy = use_hierarchy
        self.window_size = window_size

        self.input_emb = InputEmbedding(
            item_vocab_size, category_vocab_size, price_vocab_size, brand_vocab_size,
            hidden_dim, max_seq_len, dropout,
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        if use_hierarchy:
            self.memory = HierarchicalMemory(
                num_long_slots, num_short_slots, hidden_dim,
                write_strength, long_update_interval,
            )
        else:
            self.memory = EngramMemory(num_memory_slots, hidden_dim, write_strength)

        if use_target_guided:
            self.tga = TargetGuidedActivation(hidden_dim)

        if use_surprise:
            self.surprise = SurpriseEvolution(
                hidden_dim, critic_hidden_dim, surprise_window,
                surprise_low_percentile, surprise_high_percentile, write_strength,
            )
            self.mi_critic = MICritic(hidden_dim, critic_hidden_dim)

        self.norm_out = nn.LayerNorm(hidden_dim)

    def _encode_sequence(self, batch: dict) -> tuple:
        """Encode input sequence through embeddings and transformer blocks.

        Returns:
            x: (B, L, d) encoded sequence
            h_user: (B, d) last valid position
        """
        seq = batch["seq"]
        mask = batch["mask"]
        B, L = seq.shape
        device = seq.device

        x = self.input_emb(batch)

        causal = torch.tril(torch.ones(L, L, device=device, dtype=torch.bool))
        key_mask = mask.unsqueeze(1)
        attn_mask = causal.unsqueeze(0) & key_mask

        for block in self.blocks:
            x = block(x, attn_mask)

        lengths = mask.sum(dim=-1).clamp(min=1) - 1
        h_user = x[torch.arange(B, device=device), lengths]
        return x, h_user

    def _read_memory(
        self, h_user: torch.Tensor, target_emb: torch.Tensor = None
    ) -> tuple:
        """Read from memory with optional target-guided activation.

        Returns:
            readout: (B, d) memory-augmented user representation
            aux: dict with intermediate values for loss computation
        """
        aux = {}

        if self.use_hierarchy:
            if self.use_target_guided and target_emb is not None:
                all_slots = self.memory.get_all_slots()
                activation, tg_scores = self.tga(target_emb, all_slots)
                query = h_user + F.normalize(
                    torch.einsum("bk,kd->bd", activation, all_slots), dim=-1
                )
                aux["tg_activation"] = activation
                aux["tg_scores"] = tg_scores
            else:
                query = h_user

            combined, long_read, short_read, gate = self.memory.read(query)
            aux["long_read"] = long_read
            aux["short_read"] = short_read
            aux["gate"] = gate
            readout = combined
        else:
            if self.use_target_guided and target_emb is not None:
                slots = self.memory.get_slots()
                activation, tg_scores = self.tga(target_emb, slots)
                query = h_user + F.normalize(
                    torch.einsum("bk,kd->bd", activation, slots), dim=-1
                )
                aux["tg_activation"] = activation
                aux["tg_scores"] = tg_scores
            else:
                query = h_user

            readout, attn_w = self.memory.read(query)
            aux["attn_weights"] = attn_w

        return readout, aux

    def forward(self, batch: dict) -> tuple:
        """Forward pass.

        Returns:
            all_scores: (B, 1+N) pos/neg scores
            aux: dict with memory readout, attention weights, etc.
        """
        x, h_user = self._encode_sequence(batch)

        pos = self.item_emb(batch["target"])
        neg = self.item_emb(batch["negatives"])
        target_emb = torch.cat([pos.unsqueeze(1), neg], dim=1).mean(dim=1)

        memory_read, aux = self._read_memory(h_user, target_emb)

        h_user = self.norm_out(h_user + memory_read)

        pos_scores = (h_user * pos).sum(dim=-1, keepdim=True)
        neg_scores = torch.einsum("bd,bnd->bn", h_user, neg)
        all_scores = torch.cat([pos_scores, neg_scores], dim=-1)

        aux["h_user"] = h_user
        aux["memory_read"] = memory_read

        return all_scores, aux

    @property
    def item_emb(self):
        return self.input_emb.item_emb

    def write_memory(self, h_user: torch.Tensor):
        """Write user state to memory after each batch (called by trainer)."""
        if self.use_hierarchy:
            self.memory.write(h_user)
        else:
            self.memory.write(h_user)

    def compute_mi_loss(self, h_user: torch.Tensor) -> torch.Tensor:
        """Compute mutual information lower bound between user state and memory."""
        if not self.use_surprise:
            return torch.tensor(0.0, device=h_user.device)

        if self.use_hierarchy:
            memory_state = self.memory.get_all_slots().unsqueeze(0).expand(
                h_user.shape[0], -1, -1
            )
        else:
            memory_state = self.memory.get_slots().unsqueeze(0).expand(
                h_user.shape[0], -1, -1
            )

        return variational_mi_lower_bound(self.mi_critic, h_user, memory_state)

    def compute_surprise(
        self, h_user: torch.Tensor, all_scores: torch.Tensor
    ) -> tuple:
        """Compute prediction error and evolve memory via surprise mechanism."""
        if not self.use_surprise:
            return h_user.new_zeros(()), h_user.new_zeros(h_user.shape[0], dtype=torch.long)

        pos_score = all_scores[:, 0]
        pos_prob = torch.sigmoid(pos_score)
        pred_error = 1.0 - pos_prob.detach()

        if self.use_hierarchy:
            long_slots = self.memory.long_memory.get_slots()
            short_slots = self.memory.short_memory.get_slots()
            updated_long, long_surprise = self.surprise(
                h_user, long_slots, pred_error
            )
            updated_short, short_surprise = self.surprise(
                h_user, short_slots, pred_error
            )
            self.memory.long_memory.slots.data = updated_long
            self.memory.short_memory.slots.data = updated_short
            surprise_levels = long_surprise
        else:
            slots = self.memory.get_slots()
            updated_slots, surprise_levels = self.surprise(
                h_user, slots, pred_error
            )
            self.memory.slots.data = updated_slots

        return pred_error.mean(), surprise_levels
