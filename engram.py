"""EngramMemory: content-based read/write over K memory slots.

Paper Section 3.1: memory stores evolving user interest representations.
Read: attention over slots given query vector.
Write: update most similar slot via content-based addressing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EngramMemory(nn.Module):
    def __init__(
        self,
        num_slots: int,
        hidden_dim: int,
        write_strength: float = 0.1,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.hidden_dim = hidden_dim
        self.write_strength = write_strength

        self.slots = nn.Parameter(torch.randn(num_slots, hidden_dim))
        nn.init.xavier_uniform_(self.slots.unsqueeze(0))

    def read(
        self, query: torch.Tensor, mask: torch.Tensor = None
    ) -> tuple:
        """Read from memory via attention.

        Args:
            query: (B, d) query vector
            mask: (B, K) optional boolean mask for available slots

        Returns:
            readout: (B, d) attended memory vector
            attn_w: (B, K) attention weights
        """
        K, d = self.slots.shape
        B = query.shape[0]

        slots_norm = F.normalize(self.slots, dim=-1).unsqueeze(0).expand(B, -1, -1)
        query_norm = F.normalize(query, dim=-1)

        attn = torch.einsum("bd,bkd->bk", query_norm, slots_norm) / (d ** 0.5)

        if mask is not None:
            attn = attn.masked_fill(~mask, float("-inf"))

        attn_w = F.softmax(attn, dim=-1)
        readout = torch.einsum("bk,bkd->bd", attn_w, slots_norm)
        return readout, attn_w

    def write(
        self, value: torch.Tensor, slot_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Write a new value to the most similar slot.

        Args:
            value: (B, d) new memory entry
            slot_mask: (B, K) optional boolean mask for writable slots

        Returns:
            write_weights: (B, K) distribution over slots written to
        """
        K, d = self.slots.shape
        B = value.shape[0]

        slots_norm = F.normalize(self.slots, dim=-1).unsqueeze(0).expand(B, -1, -1)
        value_norm = F.normalize(value, dim=-1)

        sim = torch.einsum("bd,bkd->bk", value_norm, slots_norm)

        if slot_mask is not None:
            sim = sim.masked_fill(~slot_mask, float("-inf"))

        write_w = F.softmax(sim, dim=-1)

        with torch.no_grad():
            delta = value_norm.unsqueeze(1) - slots_norm
            update = self.write_strength * write_w.unsqueeze(-1) * delta
            self.slots.data = self.slots.data + update.sum(dim=0)

        return write_w

    def get_slots(self) -> torch.Tensor:
        return self.slots
