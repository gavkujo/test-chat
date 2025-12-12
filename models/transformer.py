# models/transformer.py

'''
Custom Transformer-based Sequence-to-Sequence Model

This module defines a lightweight transformer architecture for sequence-to-sequence
tasks, such as NL2Func or code generation. It includes:

1. PositionalEncoding:
   Adds sinusoidal positional encodings to token embeddings to inject order information
   into the model, which is crucial since transformers are permutation-invariant.

2. MiniTransformer:
   A compact Transformer model with:
   - Encoder and decoder stacks
   - Multi-head attention
   - Feedforward layers
   - Token embeddings with positional encodings
   - Output projection layer to vocabulary logits

Key Features:
- Supports causal masking for autoregressive decoding.
- Handles padding masks for variable-length sequences.
- Designed for fast experimentation on small datasets or low-resource settings.
- Suitable as a lightweight alternative to full-scale models like T5 or GPT.
'''

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        pe = self.get_buffer('pe')
        x = x + pe[:, :x.size(1), :].to(x.device)
        return x

class MiniTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 512,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        # embeddings + positional
        self.src_tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        # transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        # output projection
        self.generator = nn.Linear(d_model, vocab_size)

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        # src: (batch, src_len)
        return (src == self.pad_idx).to(src.device)  # True where padding

    def make_tgt_mask(self, tgt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # standard triangular causal mask + padding mask
        tgt_pad_mask = (tgt == self.pad_idx).to(tgt.device)
        seq_len = tgt.size(1)
        causal_mask = (
            torch.triu(torch.ones((seq_len, seq_len), device=tgt.device), diagonal=1)
            .bool()
        )
        return causal_mask, tgt_pad_mask

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
    ) -> torch.Tensor:
        """
        src: (batch, src_len)
        tgt: (batch, tgt_len)
        returns: (batch, tgt_len, vocab_size)
        """
        src_mask = None
        src_key_padding_mask = self.make_src_mask(src)

        tgt_mask, tgt_key_padding_mask = self.make_tgt_mask(tgt)

        # embed + pos
        src_emb = self.positional_encoding(self.src_tok_emb(src) * math.sqrt(self.d_model))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt) * math.sqrt(self.d_model))

        output = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_key_padding_mask=src_key_padding_mask,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        logits = self.generator(output)
        return logits
