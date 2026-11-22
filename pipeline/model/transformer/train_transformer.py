import argparse
import json
import math
import os
import random
import time
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_corpus_lines(corpus_dir: str) -> List[str]:
    lines: List[str] = []
    for fname in sorted(os.listdir(corpus_dir)):
        if not fname.endswith(".txt"):
            continue
        fpath = os.path.join(corpus_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    lines.append(stripped)
    return lines


def tokenize_and_pack(
    lines: List[str],
    tokenizer,
    block_size: int,
    batch_encode_size: int = 2048,
) -> Tuple[torch.Tensor, int]:
    """Tokenize lines with explicit <BOS>/<EOS> and pack into contiguous blocks."""
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    all_ids: List[int] = []
    for i in range(0, len(lines), batch_encode_size):
        batch = lines[i : i + batch_encode_size]
        encoded = tokenizer(batch, add_special_tokens=False)["input_ids"]
        for ids in encoded:
            if ids:
                all_ids.append(bos_id)
                all_ids.extend(ids)
                all_ids.append(eos_id)

    num_blocks = len(all_ids) // block_size
    trimmed = all_ids[: num_blocks * block_size]
    blocks = torch.tensor(trimmed, dtype=torch.long).view(num_blocks, block_size)

    # Exact character count of the packed sequence for accurate BPC calculation
    total_chars = sum(
        len(tokenizer.decode(row.tolist(), skip_special_tokens=True))
        for row in blocks
    )
    return blocks, total_chars


class PackedTokenDataset(Dataset):
    def __init__(self, blocks: torch.Tensor):
        self.blocks = blocks

    def __len__(self) -> int:
        return self.blocks.size(0)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.blocks[idx]


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float):
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2 or "norm" in name.lower() or "bias" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
    )


@torch.no_grad()
def evaluate(
    model: LlamaForCausalLM,
    val_loader: DataLoader,
    val_total_chars: int,
    device: torch.device,
    use_amp: bool,
) -> dict:
    model.eval()
    total_nll = 0.0
    total_pred_tokens = 0

    for batch in val_loader:
        batch = batch.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(input_ids=batch, labels=batch)

        # HF LlamaForCausalLM shifts internally: predicts (seq_len - 1) tokens per sequence
        n_tokens = batch.size(0) * (batch.size(1) - 1)
        total_nll += outputs.loss.item() * n_tokens
        total_pred_tokens += n_tokens

    avg_nll = total_nll / max(1, total_pred_tokens)
    ppl = math.exp(min(avg_nll, 20.0))
    bpc = total_nll / (max(1, val_total_chars) * math.log(2.0))

    return {
        "val_loss": round(avg_nll, 4),
        "val_perplexity": round(ppl, 4),
        "val_bpc": round(bpc, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Train a small causal Transformer for MehranLM.")
    parser.add_argument("--corpus_dir", type=str, default="Corpus/Cleaned")
    parser.add_argument("--tokenizer_dir", type=str, default="pipeline/tokenization/files")
    parser.add_argument("--output_dir", type=str, default="checkpoints/mehranlm_transformer")
    parser.add_argument(
        "--metrics_path",
        type=str,
        default="pipeline/model/transformer/training_metrics.json",
    )
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--intermediate_size", type=int, default=688)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Using device: {device} | AMP fp16: {use_amp}")

    # 1. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    print(f"Loaded tokenizer from {args.tokenizer_dir} (vocab_size={len(tokenizer)})")

    # 2. Load and split corpus at the line level (prevents leakage across train/val)
    lines = load_corpus_lines(args.corpus_dir)
    random.shuffle(lines)
    val_size = max(1, int(len(lines) * args.val_ratio))
    val_lines = lines[:val_size]
    train_lines = lines[val_size:]
    print(f"Lines -> train: {len(train_lines):,d} | val: {len(val_lines):,d}")

    # 3. Tokenize and pack into fixed-length blocks
    print("Tokenizing and packing train/val splits...")
    train_blocks, train_chars = tokenize_and_pack(train_lines, tokenizer, args.block_size)
    val_blocks, val_chars = tokenize_and_pack(val_lines, tokenizer, args.block_size)
    print(
        f"Train blocks: {len(train_blocks):,d} ({train_blocks.numel():,d} tokens) | "
        f"Val blocks: {len(val_blocks):,d} ({val_blocks.numel():,d} tokens, {val_chars:,d} chars)"
    )

    train_loader = DataLoader(
        PackedTokenDataset(train_blocks),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=use_amp,
    )
    val_loader = DataLoader(
        PackedTokenDataset(val_blocks),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=use_amp,
    )

    # 4. Initialize compact Llama-style causal Transformer with tied embeddings
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_heads,
        max_position_embeddings=args.block_size,
        attention_dropout=args.dropout,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        tie_word_embeddings=True,
    )
    model = LlamaForCausalLM(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Initialized MehranLM Transformer: {total_params:,d} parameters ({total_params / 1e6:.2f}M)")

    # 5. Optimizer & cosine LR schedule with warmup
    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_loader) // args.grad_accum_steps)
    total_opt_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(0.05 * total_opt_steps))

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        progress = (current_step - warmup_steps) / float(max(1, total_opt_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # 6. Training loop
    os.makedirs(args.output_dir, exist_ok=True)
    history = []
    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        num_batches = 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, start=1):
            batch = batch.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outputs = model(input_ids=batch, labels=batch)
                loss = outputs.loss / args.grad_accum_steps

            scaler.scale(loss).backward()
            running_loss += outputs.loss.item()
            num_batches += 1

            if step % args.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

        train_loss = running_loss / max(1, num_batches)
        val_metrics = evaluate(model, val_loader, val_chars, device, use_amp)
        elapsed = time.time() - t0

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_perplexity": round(math.exp(min(train_loss, 20.0)), 4),
            **val_metrics,
            "lr": round(scheduler.get_last_lr()[0], 6),
            "epoch_time_sec": round(elapsed, 1),
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch}/{args.epochs} ({elapsed:.1f}s) | "
            f"Train Loss: {epoch_record['train_loss']:.4f} (PPL: {epoch_record['train_perplexity']:.2f}) | "
            f"Val Loss: {epoch_record['val_loss']:.4f} (PPL: {epoch_record['val_perplexity']:.2f}, "
            f"BPC: {epoch_record['val_bpc']:.4f})"
        )

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            print(f"  -> Saved new best checkpoint to {args.output_dir}")

    # 7. Save compact training report (safe to commit to Git)
    report = {
        "architecture": "LlamaForCausalLM",
        "parameters": total_params,
        "config": vars(args),
        "dataset": {
            "train_lines": len(train_lines),
            "val_lines": len(val_lines),
            "train_tokens": int(train_blocks.numel()),
            "val_tokens": int(val_blocks.numel()),
            "val_chars": int(val_chars),
        },
        "best_val_loss": round(best_val_loss, 4),
        "history": history,
    }
    os.makedirs(os.path.dirname(args.metrics_path), exist_ok=True)
    with open(args.metrics_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nTraining complete! Metrics saved to {args.metrics_path}")


if __name__ == "__main__":
    main()
