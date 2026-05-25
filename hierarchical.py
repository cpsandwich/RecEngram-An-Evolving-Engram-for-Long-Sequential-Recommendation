"""HierarchicalMemory: long/short-term dual-bank memory.

Paper Section 3.3 + Equation 9-10:
- Long-term bank: K_long slots, updated every Δt_long steps
- Short-term bank: K_short slots, updated every step
- Adaptive gating: learnable gate combines long and short readouts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .engram import EngramMemory


class HierarchicalMemory(nn.Module):
    def __init__(
        self,
        num_long_slots: int,
        num_short_slots: int,
        hidden_dim: int,
        write_strength: float = 0.1,
        long_update_interval: int = 5,
    ):
        super().__init__()
        self.num_long = num_long_slots
        self.num_short = num_short_slots
        self.hidden_dim = hidden_dim
        self.long_update_interval = long_update_interval

        self.long_memory = EngramMemory(num_long_slots, hidden_dim, write_strength)
        self.short_memory = EngramMemory(num_short_slots, hidden_dim, write_strength)

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

        self.register_buffer("step_counter", torch.tensor(0, dtype=torch.long))

    def read(self, query: torch.Tensor) -> tuple:
        """Read from both long and short memory, combine via learned gate.

        Args:
            query: (B, d) query vector

        Returns:
            combined: (B, d) gated combination
            long_read: (B, d) long-term readout
            short_read: (B, d) short-term readout
            gate: (B, d) gate values
        """
        long_mask = self._long_mask(query.device)
        short_mask = self._short_mask(query.device)

        long_read, long_attn = self.long_memory.read(query, long_mask)
        short_read, short_attn = self.short_memory.read(query, short_mask)

        g = self.gate(torch.cat([query, long_read], dim=-1))
        combined = g * long_read + (1 - g) * short_read
        return combined, long_read, short_read, g

    def write(self, value: torch.Tensor, force_long: bool = False):
        """Write to short memory every step, long memory every Δt_long steps.

        Args:
            value: (B, d) new memory entry
            force_long: if True, also write to long memory
        """
        short_mask = self._short_mask(value.device)
        self.short_memory.write(value, short_mask)

        self.step_counter += 1
        if force_long or (self.step_counter % self.long_update_interval == 0):
            long_mask = self._long_mask(value.device)
            self.long_memory.write(value, long_mask)

    def get_all_slots(self) -> torch.Tensor:
        """Concatenate all slots from both banks. Returns (K_long+K_short, d)."""
        return torch.cat([self.long_memory.get_slots(), self.short_memory.get_slots()], dim=0)

    def _long_mask(self, device: torch.device) -> torch.Tensor:
        m = torch.ones(self.num_long, dtype=torch.bool, device=device).unsqueeze(0)
        return m

    def _short_mask(self, device: torch.device) -> torch.Tensor:
        m = torch.ones(self.num_short, dtype=torch.bool, device=device).unsqueeze(0)
        return m
