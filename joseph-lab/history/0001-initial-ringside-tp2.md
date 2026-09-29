# 0001 — Initial RiNGSiDE TP2 validation

**Date:** 2026-09-29  
**Status:** QUALIFIED as the checkpoint-substitution baseline  
**Upstream base:** `othexmr/GLM-5.3-Flash-NVFP4-2x-4x-DGX-Sparks-RiNGSiDE@6606f7638aff044a8170e414818477679934a1c3`

## Purpose

Establish the author's TP2 release profile on the local two-DGX-Spark system before changing the target checkpoint. No G3 patches or unrelated tuning were to be introduced during this phase.

## Target and draft

Target:

- repository: `RedHatAI/GLM-5.3-Flash-NVFP4`
- revision: `18d55bfd5a2194887738da73753975c9d3842f46`
- format: compressed-tensors NVFP4

Draft/speculator:

- DFlash2 enabled
- K = 7
- draft revision used by this comparison lane: `7d74cdd8` (short form retained in the historical receipt)

## Fixed TP2 serving profile

The measured baseline used the RiNGSiDE TP2 profile at the pinned upstream commit, including:

- 2 × DGX Spark / GB10
- TP2
- DCP1
- B12X
- sparse-MLA
- async scheduling OFF
- prefix-cache retention interval = 4608
- CUDA graph capture = `8,16,24,32,40,48`
- max model length = 262144
- max sequences = 6
- max batched tokens = 14336
- KV dtype = `fp8_e4m3`
- configured KV pool = 11,274,289,152 bytes/rank (10.5 GiB/rank)

Production DeepSeek remained untouched/inactive during this work.

## First correctness result

| Gate | Result |
|---|---|
| P1 | PASS |
| P2 | PASS |
| P3 | PASS |
| P4 | PASS |
| P6 | historical PASS; later scorer review found a possible schema false-positive risk |

The P6 caveat is preserved because this is a historical record, not a retroactive reclassification.

## First measured throughput

### Decode

| Workload | Throughput |
|---|---:|
| C1 code | 57.534 tok/s |
| C1 prose | 33.692 tok/s |
| C1 structured | 83.056 tok/s |

### Prefill

| Prompt size | Throughput |
|---|---:|
| 8K | 2243.23 tok/s |
| 32K | 2408.24 tok/s |
| 64K | 2387.68 tok/s |

## Memory / swap

Minimum observed `MemAvailable` during the formal run:

| Rank | Minimum MemAvailable |
|---|---:|
| rank0 | approximately 1.19 GiB |
| rank1 | approximately 0.79 GiB |

Observed page-out delta:

| Rank | `pswpout` delta |
|---|---:|
| rank0 | +67,165 |
| rank1 | +38,330 |

This established that the baseline ran correctly but had very little RAM headroom and did page out during the measured run.

## DFlash2 telemetry

- K = 7
- acceptance = 39.68%

An older reported value of `58.333` was not retained as accepted tokens/speculative step because it is inconsistent with the K=7 interpretation. The true historical accepted-tokens/spec-step value for this Phase A record is therefore `UNKNOWN`.

## KV layout/capacity

The same fixed profile was later formally clarified as:

- configured KV budget/rank = 11,274,289,152 bytes
- actual block-aligned backing/rank = 11,270,983,680 bytes
- padding = 3,305,472 bytes/rank
- number of blocks = 408
- bytes/block = 27,624,960
- target block = 4608 tokens
- drafter logical block = 1152 tokens
- target shared capacity = 1,880,064 tokens
- drafter logical capacity = 470,016 tokens

The target capacity is a shared TP2 capacity and must not be doubled across ranks. The drafter logical capacity must also not be added to the target capacity.

## Disposition

This Phase A state was accepted as the clean baseline for the next operation: changing only the target checkpoint while keeping the RiNGSiDE TP2 serving profile fixed.
