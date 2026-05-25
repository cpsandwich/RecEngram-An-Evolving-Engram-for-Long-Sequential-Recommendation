"""DIEN: Deep Interest Evolution Network (Zhou et al., AAAI 2019).

Extends DIN with interest extractor (GRU) and interest evolving (AUGRU).
AUGRU = GRU with attention-weighted update gate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InterestExtractor(nn.Module):
    """GRU-based interest extraction from behavior sequence."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B, L, d = x.shape
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=L)
        return out


class AUGRU(nn.Module):
    """GRU with Attention Update gate: u'_t = a_t * u_t."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(
        self, x: torch.Tensor, attn_scores: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        B, L, d = x.shape
        device = x.device
        h = torch.zeros(B, d, device=device)

        mask = torch.arange(L, device=device).unsqueeze(0) < lengths.unsqueeze(1)

        for t in range(L):
            xt = x[:, t, :]
            ht = self.gru(xt, h)
            a_t = attn_scores[:, t]
            u_t = torch.sigmoid(
                self.gru.weight_ih[d:, :].mm(xt.T).T +
                self.gru.weight_hh[d:, :].mm(h.T).T +
                self.gru.bias_ih[d:].unsqueeze(0) +
                self.gru.bias_hh[d:].unsqueeze(0)
            )
            u_t_mod = a_t.unsqueeze(-1) * u_t
            r_t = torch.sigmoid(
                self.gru.weight_ih[:d, :].mm(xt.T).T +
                self.gru.weight_hh[:d, :].mm(h.T).T +
                self.gru.bias_ih[:d].unsqueeze(0) +
                self.gru.bias_hh[:d].unsqueeze(0)
            )
            h_tilde = torch.tanh(
                self.gru.weight_ih[d:2*d, :].mm(xt.T).T +
                r_t * (self.gru.weight_hh[d:2*d, :].mm(h.T).T +
                       self.gru.bias_hh[d:2*d].unsqueeze(0)) +
                self.gru.bias_ih[d:2*d].unsqueeze(0)
            )
            h_new = (1 - u_t_mod) * h + u_t_mod * h_tilde
            h = torch.where(mask[:, t].unsqueeze(-1), h_new, h)

        return h


class DIEN(nn.Module):
    def __init__(
        self,
        item_vocab_size: int,
        hidden_dim: int = 128,
        max_seq_len: int = 500,
        dropout: float = 0.2,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.item_emb = nn.Embedding(item_vocab_size, hidden_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout)

        self.interest_extractor = InterestExtractor(hidden_dim)
        self.interest_evolving = AUGRU(hidden_dim)

        self.attn_fc = nn.Sequential(
            nn.Linear(hidden_dim * 4, 80),
            nn.GELU(),
            nn.Linear(80, 40),
            nn.GELU(),
            nn.Linear(40, 1),
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 3, 200),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(200, 80),
            nn.GELU(),
            nn.Linear(80, hidden_dim),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, batch: dict) -> torch.Tensor:
        seq = batch["seq"]
        mask = batch["mask"]
        B, L = seq.shape

        seq_emb = self.emb_dropout(self.item_emb(seq))
        target_emb = self.item_emb(batch["target"])
        neg_emb = self.item_emb(batch["negatives"])

        lengths = mask.sum(dim=-1).clamp(min=1)

        interest_states = self.interest_extractor(seq_emb, lengths)

        target_expand = target_emb.unsqueeze(1).expand(-1, L, -1)
        attn_input = torch.cat([
            target_expand, interest_states,
            target_expand - interest_states,
            target_expand * interest_states,
        ], dim=-1)
        attn_scores = self.attn_fc(attn_input).squeeze(-1)

        pad_mask = torch.arange(L, device=seq.device).unsqueeze(0) >= lengths.unsqueeze(1)
        attn_scores = attn_scores.masked_fill(pad_mask, float("-inf"))
        attn_w = F.softmax(attn_scores, dim=-1)

        h_user = self.interest_evolving(interest_states, attn_w, lengths)

        concat = torch.cat([h_user, target_emb, h_user * target_emb], dim=-1)
        h_out = self.fc(concat)

        pos_scores = (h_out * target_emb).sum(dim=-1, keepdim=True)
        neg_scores = torch.einsum("bd,bnd->bn", h_out, neg_emb)
        return torch.cat([pos_scores, neg_scores], dim=-1)
