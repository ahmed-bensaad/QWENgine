# Model file inventory

Exact facts about the two GGUF files the engine runs, from `tools/gguf_inventory.py` (an independent Python
implementation of the GGUF spec) and `build/inspect_gguf` (the engine's parser). Regenerate the full per-tensor
table with `python3 tools/gguf_inventory.py <file.gguf> --markdown out.md` (or `--csv out.csv`).

## Files

| | `Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf` | `Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf` |
|---|---|---|
| Size | 10,094,357,632 bytes | 10,442,827,840 bytes |
| GGUF | v3, 52 metadata keys, 851 tensors | v3, 53 keys (+ `qwen35.nextn_predict_layers` = 1), 866 tensors |
| Tensor data | at 0xa7c480, alignment 32, densely packed | same trunk tensors + 15 tensors of `blk.64` |
| Tensor bytes | 10,083,362,816 (9.391 GiB) | + 348,469,248 (332.3 MiB, MTP block) |
| Origin | ISTA DASLab, "GSQ-RCO non-uniform mixed-precision GGUF", base model Qwen/Qwen3.8-27B, Apache-2.0 | same |

851 = 48 DeltaNet layers × 14 tensors + 16 attention layers × 11 + 3 model-level tensors.

## Hyperparameters (`qwen35.*`)

| Key | Value |
|---|---|
| `block_count` | 64 (the MTP file: 65, with `nextn_predict_layers` = 1) |
| `embedding_length` / `feed_forward_length` | 5120 / 17408 |
| `attention.head_count` / `head_count_kv` | 24 / 4 |
| `attention.key_length` / `value_length` | 256 / 256 |
| `attention.layer_norm_rms_epsilon` | 1e-6 |
| `full_attention_interval` | 4 (layers 3, 7, ..., 63) |
| `rope.freq_base` / `dimension_count` / `dimension_sections` | 1e7 / 64 / [11, 11, 10, 0] |
| `ssm.conv_kernel` / `state_size` / `group_count` / `time_step_rank` / `inner_size` | 4 / 128 / 16 / 48 / 6144 |
| `context_length` | 262144 (training) |
| Tokenizer | `gpt2` BPE, pre-tokenizer `qwen35`, 248,320 tokens, 247,587 merges, EOS 248046, BOS/PAD 248044 |

## Storage formats

| Format | Tensors | Bytes | MiB | Bits per weight |
|---|---|---|---|---|
| IQ3_S | 97 | 2,264,064,000 | 2159.2 | 3.4375 |
| IQ3_XXS | 83 | 2,043,166,720 | 1948.5 | 3.0625 |
| IQ2_S | 72 | 2,000,957,440 | 1908.3 | 2.5625 |
| IQ4_XS | 33 | 1,449,738,240 | 1382.6 | 4.25 |
| Q2_K | 28 | 622,755,840 | 593.9 | 2.625 |
| IQ2_XS | 32 | 616,816,640 | 588.2 | 2.3125 |
| IQ2_XXS | 30 | 567,705,600 | 541.4 | 2.0625 |
| Q4_K | 19 | 383,385,600 | 365.6 | 4.5 |
| BF16 | 96 | 47,185,920 | 45.0 | 16 |
| IQ1_M | 4 | 40,140,800 | 38.3 | 1.75 |
| IQ1_S | 4 | 36,864,000 | 35.2 | 1.5625 |
| F32 | 353 | 10,582,016 | 10.1 | 32 |

Formats vary per tensor within every layer (for example layer 0: `attn_qkv` IQ4_XS, `ffn_gate` IQ1_S, `ffn_up`
IQ1_M, `ffn_down` IQ2_XS).

## Tensors by role

Shapes in ggml order (ne0 = row length K, ne1 = rows N).

| Role | Shape | Count | MiB | Formats (tensor count) |
|---|---|---|---|---|
| `token_embd.weight` | 5120 × 248320 | 1 | 388.4 | IQ2_S |
| `output.weight` (LM head) | 5120 × 248320 | 1 | 644.1 | IQ4_XS |
| `output_norm.weight` | 5120 | 1 | 0.02 | F32 |
| `attn_norm`, `post_attention_norm` | 5120 | 64 + 64 | 2.5 | F32 |
| **DeltaNet layers (48)** | | | | |
| `attn_qkv.weight` | 5120 × 10240 | 48 | 877.0 | IQ2_S 15, IQ3_S 10, IQ3_XXS 9, IQ2_XS 7, IQ4_XS 4, IQ2_XXS 2, Q2_K 1 |
| `attn_gate.weight` (z) | 5120 × 6144 | 48 | 515.2 | IQ3_XXS 12, IQ3_S 7, Q2_K 7, IQ2_S 7, IQ2_XS 5, IQ4_XS 3, IQ1_M 3, IQ2_XXS 3, Q4_K 1 |
| `ssm_alpha.weight`, `ssm_beta.weight` | 5120 × 48 | 48 + 48 | 45.0 | BF16 |
| `ssm_conv1d.weight` | 4 × 10240 | 48 | 7.5 | F32 |
| `ssm_a`, `ssm_dt.bias` | 48 | 48 + 48 | 0.02 | F32 |
| `ssm_norm.weight` | 128 | 48 | 0.02 | F32 |
| `ssm_out.weight` | 6144 × 5120 | 48 | 627.2 | IQ3_S 25, IQ3_XXS 10, IQ4_XS 8, IQ2_S 3, Q4_K 2 |
| **Attention layers (16)** | | | | |
| `attn_q.weight` (Q and gate) | 5120 × 12288 | 16 | 349.2 | IQ3_XXS 8, Q2_K 6, IQ2_XXS 1, IQ4_XS 1 |
| `attn_k.weight`, `attn_v.weight` | 5120 × 1024 | 16 + 16 | 74.0 | IQ4_XS 7, Q4_K 10, IQ3_S 7, IQ3_XXS 3, IQ2_S 2, IQ1_S 2, IQ2_XXS 1 |
| `attn_q_norm`, `attn_k_norm` | 256 | 16 + 16 | 0.03 | F32 |
| `attn_output.weight` | 6144 × 5120 | 16 | 180.0 | IQ3_S 10, IQ2_XS 3, IQ2_XXS 2, IQ2_S 1 |
| **FFN (all 64 layers)** | | | | |
| `ffn_gate.weight` | 5120 × 17408 | 64 | 1859.4 | IQ2_S 16, Q2_K 10, IQ3_XXS 10, IQ2_XXS 9, IQ3_S 7, IQ2_XS 6, IQ1_S 2, IQ4_XS 2, Q4_K 2 |
| `ffn_up.weight` | 5120 × 17408 | 64 | 1957.0 | IQ3_S 15, IQ3_XXS 14, IQ2_S 12, IQ2_XXS 10, IQ2_XS 6, Q2_K 2, IQ4_XS 2, Q4_K 2, IQ1_M 1 |
| `ffn_down.weight` | 17408 × 5120 | 64 | 2089.8 | IQ3_XXS 17, IQ3_S 16, IQ2_S 15, IQ2_XS 5, IQ4_XS 5, IQ2_XXS 2, Q2_K 2, Q4_K 2 |

Weight bytes read per generated token: all tensors except the embedding table (of which one row is read), about
9.68 GB.

## MTP block (`blk.64` in the `-mtp` file)

| Tensor | Shape | Format | Bytes |
|---|---|---|---|
| `nextn.eh_proj.weight` | 10240 × 5120 | Q6_K | 43,008,000 |
| `attn_q.weight` | 5120 × 12288 | Q6_K | 51,609,600 |
| `attn_k.weight`, `attn_v.weight` | 5120 × 1024 | Q6_K | 4,300,800 each |
| `attn_output.weight` | 6144 × 5120 | Q6_K | 25,804,800 |
| `ffn_gate.weight`, `ffn_up.weight` | 5120 × 17408 | Q6_K | 73,113,600 each |
| `ffn_down.weight` | 17408 × 5120 | Q6_K | 73,113,600 |
| `nextn.enorm`, `nextn.hnorm`, `nextn.shared_head_norm`, `attn_norm`, `post_attention_norm` | 5120 | F32 | 20,480 each |
| `attn_q_norm`, `attn_k_norm` | 256 | F32 | 1,024 each |

The engine requantizes the Q6_K matrices to Q4_K at load (332 → 228 MiB; cached in `data/cache/`). The MTP block
shares `token_embd` and `output.weight` with the trunk.
