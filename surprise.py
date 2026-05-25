"""SurpriseEvolution: three-tier memory update + MI critic.

Paper Section 3.2: surprise-driven memory evolution.
- Fine-tune (low surprise): small L2 adjustment to most similar slot
- GRU re-encode (medium surprise): re-encode slot via learned GRU
- Allocate new (high surprise): replace least-used slot
Paper Equation 8: mutual information pruning via variational bound.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque


class MICritic(nn.Module):
    """Critic network for mutual information estimation.

    Estimates I(s; M) = E_{p(s,M)}[log f(s,M)] - E_{p(s)p(M)}[log f(s,M)]
    where f is a learnable critic network.
    """

    def __init__(self, hidden_dim: int, critic_hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, critic_hidden),
            nn.GELU(),
            nn.Linear(critic_hidden, critic_hidden),
            nn.GELU(),
            nn.Linear(critic_hidden, 1),
        )

    def forward(self, state: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Compute critic score f(s, M) for MI estimation.

        Args:
            state: (B, d) current user state
            memory: (B, K, d) or (B, d) memory representation

        Returns:
            (B, 1) critic scores
        """
        if memory.dim() == 3:
            memory = memory.mean(dim=1)
        x = torch.cat([state, memory], dim=-1)
        return self.net(x)


class SurpriseEvolution(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        critic_hidden: int = 256,
        surprise_window: int = 5000,
        surprise_low: float = 0.30,
        surprise_high: float = 0.70,
        write_strength: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.surprise_window = surprise_window
        self.surprise_low = surprise_low
        self.surprise_high = surprise_high
        self.write_strength = write_strength

        self.critic = MICritic(hidden_dim, critic_hidden)
        self.reencoder = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)

        self.register_buffer("prediction_errors", torch.zeros(surprise_window))
        self.register_buffer("error_ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("error_count", torch.tensor(0, dtype=torch.long))

        self.register_buffer("slot_ages", torch.zeros(0))

    def update_error_stats(self, errors: torch.Tensor):
        """Update running buffer of prediction errors."""
        with torch.no_grad():
            n = errors.numel()
            ptr = self.error_ptr.item()
            remaining = min(n, self.surprise_window - ptr)
            self.prediction_errors[ptr:ptr + remaining] = errors.flatten()[:remaining]
            if n > remaining:
                wrap = n - remaining
                self.prediction_errors[:wrap] = errors.flatten()[remaining:]
            self.error_ptr = torch.tensor((ptr + n) % self.surprise_window)
            self.error_count = torch.tensor(min(
                self.error_count.item() + n, self.surprise_window
            ))

    def _get_thresholds(self) -> tuple:
        """Get low and high surprise thresholds from historical errors."""
        if self.error_count < 100:
            return 0.0, float("inf")
        errors = self.prediction_errors[:self.error_count].clone()
        sorted_errors, _ = errors.sort()
        low_idx = int(self.surprise_low * len(sorted_errors))
        high_idx = int(self.surprise_high * len(sorted_errors))
        return sorted_errors[low_idx].item(), sorted_errors[high_idx].item()

    def evolve(
        self,
        new_state: torch.Tensor,
        memory_slots: torch.Tensor,
        surprise_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply three-tier evolution to memory slots.

        Args:
            new_state: (B, d) new state to write
            memory_slots: (K, d) current memory slots
            surprise_mask: (B,) long tensor with values 0 (fine-tune), 1 (re-encode), 2 (allocate)

        Returns:
            updated_slots: (K, d) updated memory
        """
        K, d = memory_slots.shape
        B = new_state.shape[0]
        updated_slots = memory_slots.clone()

        if B == 0:
            return updated_slots

        state_norm = F.normalize(new_state, dim=-1)
        slots_norm = F.normalize(memory_slots, dim=-1)
        sim = torch.einsum("bd,kd->bk", state_norm, slots_norm)
        nearest_slot = sim.argmax(dim=-1)

        for b in range(B):
            idx = nearest_slot[b]
            tier = surprise_mask[b].item()

            if tier == 0:
                delta = self.write_strength * (new_state[b] - memory_slots[idx])
                updated_slots[idx] = memory_slots[idx] + delta
            elif tier == 1:
                ctx = memory_slots[idx].unsqueeze(0).unsqueeze(0)
                inp = new_state[b].unsqueeze(0).unsqueeze(0)
                _, h = self.reencoder(inp, ctx)
                updated_slots[idx] = h.squeeze(0).squeeze(0)
            else:
                ages = self.slot_ages
                if ages.numel() != K:
                    ages = torch.zeros(K, device=memory_slots.device)
                oldest = ages.argmax().item()
                updated_slots[oldest] = new_state[b]

                if self.slot_ages.numel() == K:
                    self.slot_ages[oldest] = 0

        return updated_slots

    def forward(
        self,
        h_user: torch.Tensor,
        memory_slots: torch.Tensor,
        prediction_error: torch.Tensor,
    ) -> tuple:
        """Compute surprise levels and evolve memory.

        Returns:
            updated_slots: (K, d)
            surprise_levels: (B,) tensor with tier assignments
        """
        self.update_error_stats(prediction_error.detach())
        low_thresh, high_thresh = self._get_thresholds()

        surprise_mask = torch.zeros_like(prediction_error, dtype=torch.long)
        surprise_mask[prediction_error > low_thresh] = 1
        surprise_mask[prediction_error > high_thresh] = 2

        updated_slots = self.evolve(h_user, memory_slots, surprise_mask)
        return updated_slots, surprise_mask


def variational_mi_lower_bound(
    critic: MICritic,
    state: torch.Tensor,
    memory: torch.Tensor,
    shuffle: bool = True,
) -> torch.Tensor:
    """Variational lower bound of mutual information I(state; memory).

    E_{p(s,M)}[log f] - log E_{p(s)p(M)}[exp(f)]

    Args:
        critic: MICritic network
        state: (B, d) user state
        memory: (B, K, d) memory representation
        shuffle: if True, shuffle memory to approximate p(s)p(M)

    Returns:
        scalar MI lower bound
    """
    joint = critic(state, memory).squeeze(-1)

    if shuffle:
        idx = torch.randperm(memory.shape[0], device=memory.device)
        memory_shuffled = memory[idx]
    else:
        memory_shuffled = memory

    marginal = critic(state, memory_shuffled).squeeze(-1)

    mi = joint.mean() - marginal.exp().mean().log()
    return mi
