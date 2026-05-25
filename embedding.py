"""InputEmbedding: item + category + price_tier + brand + position."""

import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    def __init__(
        self,
        item_vocab_size: int,
        category_vocab_size: int,
        price_vocab_size: int,
        brand_vocab_size: int,
        hidden_dim: int = 128,
        max_seq_len: int = 500,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.item_emb = nn.Embedding(item_vocab_size, hidden_dim, padding_idx=0)
        self.cat_emb = nn.Embedding(category_vocab_size, hidden_dim, padding_idx=0)
        self.price_emb = nn.Embedding(price_vocab_size, hidden_dim, padding_idx=0)
        self.brand_emb = nn.Embedding(brand_vocab_size, hidden_dim, padding_idx=0)

        self.pos_emb = nn.Embedding(max_seq_len + 1, hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for emb in [self.item_emb, self.cat_emb, self.price_emb, self.brand_emb, self.pos_emb]:
            nn.init.xavier_uniform_(emb.weight)

    def forward(self, batch: dict) -> torch.Tensor:
        """Sum item, category, price, brand, and position embeddings.

        Returns: (B, L, d) tensor.
        """
        seq = batch["seq"]
        B, L = seq.shape
        device = seq.device

        x = self.item_emb(seq)

        cat = batch.get("category")
        if cat is not None:
            x = x + self.cat_emb(cat)

        price = batch.get("price_tier")
        if price is not None:
            x = x + self.price_emb(price)

        brand = batch.get("brand")
        if brand is not None:
            x = x + self.brand_emb(brand)

        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_emb(positions)

        return self.dropout(self.norm(x))
