# infer.py

'''
infer.py — Greedy Decoding for NL2Func Transformer

This module provides a minimal inference pipeline for a transformer-based NL2Func model.

Features:
1. Loads a trained MiniTransformer model from checkpoint.
2. Uses SentencePiece tokenizer to encode/decode natural language input.
3. Implements greedy decoding for sequence generation.
4. Decodes the model's output and optionally parses it via parse_and_build.
5. Handles JSON output where the model predicts structured data.
6. Prints both the raw decoded text and the parsed function representation.

Functions:
- greedy_decode(model, src_ids, sp, max_len, device):
    Performs autoregressive greedy decoding on a source sequence.
    Stops at EOS token or when max_len is reached.

- infer(args):
    Loads model and tokenizer, encodes input text, generates output with greedy decoding, 
    decodes the token sequence, prints text, parses using parse_and_build, and prints the parsed structure.

Usage:
    python infer.py --text "Your NL input here" --tokenizer path/to/tokenizer --model_path path/to/model.pt
'''

import torch
import sentencepiece as spm
import argparse
from models.transformer import MiniTransformer
from data.parser_test import parse_and_build as pb

def greedy_decode(model, src_ids, sp, max_len, device):
    model.eval()
    src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)
    src_mask = (src == sp.PieceToId('[PAD]')).to(device)

    # encoder output
    with torch.no_grad():
        memory = model.transformer.encoder(
            model.positional_encoding(model.src_tok_emb(src) * (model.d_model ** 0.5)),
            src_key_padding_mask=src_mask
        )

    # start with BOS
    tgt_ids = [sp.PieceToId('[BOS]')]
    for _ in range(max_len):
        tgt_tensor = torch.tensor(tgt_ids, dtype=torch.long, device=device).unsqueeze(0)
        tgt_mask = torch.triu(torch.ones((len(tgt_ids), len(tgt_ids)), device=device), diagonal=1).bool()
        out = model.transformer.decoder(
            model.positional_encoding(model.tgt_tok_emb(tgt_tensor) * (model.d_model ** 0.5)),
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_mask
        )
        logits = model.generator(out[:, -1, :])  # last token
        next_id = logits.argmax(-1).item()
        if next_id == sp.PieceToId('[EOS]'):
            break
        tgt_ids.append(next_id)

    return tgt_ids

def infer(args):
    # load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer + '.model')

    # load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.model_path, map_location=device)
    vocab_size = sp.GetPieceSize()
    model = MiniTransformer(
        vocab_size=vocab_size,
        d_model=checkpoint['model_state'][list(checkpoint['model_state'].keys())[0]].shape[1],
        nhead=args.nhead,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.ff_dim,
        dropout=args.dropout,
        max_len=args.max_len,
        pad_idx=sp.PieceToId('[PAD]')
    ).to(device)
    model.load_state_dict(checkpoint['model_state'])

    # tokenize input
    raw = args.text
    src_ids = [sp.PieceToId('[BOS]')] + sp.EncodeAsIds(raw) + [sp.PieceToId('[EOS]')]
    src_ids = src_ids[:args.max_len]  # Ensure input fits positional encoding
    # generate
    out_ids = greedy_decode(model, src_ids, sp, args.max_len, device)
    # decode
    tokens = [sp.IdToPiece(i) for i in out_ids[1:]]  # exclude BOS
    # join and clean
    import json
    text = sp.DecodePieces(tokens)
    try:
        result = json.loads(text)
        print("result: ", result)
    except Exception:
        print(text)

    print(pb(raw,text))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference on NL2Func Transformer")
    parser.add_argument('--tokenizer', type=str, default='tokenizer/tokenizer',
                        help="SentencePiece model prefix")
    parser.add_argument('--model_path', type=str, default='saved/best_model.pt',
                        help="Path to trained model checkpoint")
    parser.add_argument('--text', type=str, required=True,
                        help="Raw input text to convert")
    parser.add_argument('--max_len', type=int, default=512,
                        help="Max output length (in tokens)")
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--enc_layers', type=int, default=3)
    parser.add_argument('--dec_layers', type=int, default=3)
    parser.add_argument('--ff_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    args = parser.parse_args()
    infer(args)


