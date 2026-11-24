# MehranLM Small Transformer Baseline

A reproducible causal Transformer baseline using HuggingFace
`LlamaForCausalLM`, trained on `Corpus/Cleaned/` using
`MehranLM-Tokenizer` from `pipeline/tokenization/files/`.

## Architecture & Setup

- **Model:** HuggingFace `LlamaForCausalLM`
- **Parameters:** 13,063,424 (~13.06M)
- **Config:** `vocab_size=32488`, `hidden_size=256`,
  `intermediate_size=688`, `num_layers=6`, `num_heads=4`,
  `block_size=256`, `dropout=0.1`
- **Data Split:** 95% training (139,578 lines,
  6,046,976 tokens) / 5% validation
  (7,346 lines, 319,232 tokens).
- **Optimization:** AdamW, cosine learning-rate decay,
  5% warmup, fp16 mixed precision.

## Validation Results

| Epoch | Train Loss | Train PPL | Val Loss | Val PPL | Val BPC | Time |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6.8026 | 900.15 | 5.5535 | 258.14 | 2.0490 | 99.5s |
| 2 | 5.2366 | 188.03 | 5.0104 | 149.96 | 1.8486 | 104.3s |
| 3 | 4.8413 | 126.64 | 4.7963 | 121.06 | 1.7696 | 104.0s |
| 4 | 4.6294 | 102.45 | 4.7013 | 110.09 | 1.7345 | 103.7s |
| 5 | 4.5130 | 91.19 | 4.6692 | 106.61 | 1.7227 | 103.9s |


## Held-Out Evaluation (`Corpus/Testing-data/`)

Evaluated on held-out testing datasets excluded from training.

| Dataset | Lines | Loss | Perplexity | BPC |
|---|---:|---:|---:|---:|
| `sindhi_sentiment_parsed.txt` | 1,904 | 5.6064 | 272.16 | 2.1259 |
| `sindhi_misspelling_corrected_sentences.txt` | 100,000 | 6.1145 | 452.39 | 2.3881 |
| `sindhi_misspelled_sentences.txt` | 100,000 | 7.6863 | 2178.25 | 4.0367 |


## Usage

### Train

```bash
cd pipeline
python model/transformer/train_transformer.py --epochs 5
```

### Test Generation and Perplexity / BPC

```bash
cd pipeline
python model/transformer/testing_transformer.py --max_new_tokens 80
```

Sample outputs are saved in `text_generated_by_transformer.txt`.
Training and evaluation metrics are stored in `training_metrics.json`.

## Notes

This is an initial experimental baseline. Results should be interpreted
in the context of the available Sindhi corpus and tokenizer.

Further experiments should investigate longer training, corpus diversity,
and performance across different Sindhi text domains.
