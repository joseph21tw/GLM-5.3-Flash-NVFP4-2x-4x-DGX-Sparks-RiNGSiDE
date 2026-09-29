# 0002 — Switch target checkpoint to NVFP4-Spark

**Date:** 2026-09-29  
**Status:** QUALIFIED for correctness and serving  
**Base:** history entry `0001`, upstream RiNGSiDE `6606f7638aff044a8170e414818477679934a1c3`

## Change

Only the target checkpoint was intentionally replaced.

From:

- `RedHatAI/GLM-5.3-Flash-NVFP4`
- revision `18d55bfd5a2194887738da73753975c9d3842f46`

To:

- `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark`
- revision `a608241037e4c2565356bff7ca293f2133888f88`
- local path `/home/joseph/ai/models/GLM-5.3-Flash-NVFP4-Spark`

The RiNGSiDE runtime/profile was kept matched to Phase A: same pinned repository commit, image/runtime, B12X, sparse-MLA, TP2, DCP1, DFlash2 K=7, scheduler, prefix/replay behavior, graph capture, `fp8_e4m3` KV, 11,274,289,152-byte/rank KV pool, model length, sequence count, batched-token limit, prompts, and measurement method.

No G3 optimization was intentionally ported as part of this checkpoint substitution.

## Direct-load failure

The checkpoint did not load directly under the original Phase A checkpoint contract.

First blocking error:

```text
KeyError:
layers.0.self_attn.in_proj_qkvbfg_a.weight_scale
```

The incompatibility was handled with a generalized checkpoint-contract adapter rather than by importing the G3 runtime wholesale.

## Compatibility adaptation

The qualified adaptation covered the checkpoint contract needed by:

- KDA
- Indexer
- MLA handling

Qualification receipts reported:

```text
KDA_ADAPTER=PASS
INDEXER_ADAPTER=PASS
MLA_HANDLER rank0/rank1 PASS
A_SEMANTIC_BEHAVIOR_UNCHANGED=YES
```

An intermediate deployment error was also discovered: rank0 had the generalized adapter while rank1 still had stale KDA/Indexer adapter content.

A prelaunch overlay parity gate was then added/used and the two ranks were made identical:

```text
PRELAUNCH_OVERLAY_PARITY=PASS
ADAPTER_SHA256_RANK0=4f3d1462905f52a339c67dba73c61003d501e2e5f845167a1ff934c6679a1d43
ADAPTER_SHA256_RANK1=4f3d1462905f52a339c67dba73c61003d501e2e5f845167a1ff934c6679a1d43
ADAPTER_HASH_MATCH=YES
IMAGE_DIGEST_MATCH=YES
IMPORT_PATH_MATCH=YES
```

Final load result:

```text
B_ADAPTED_LOAD=PASS
GENERALIZED_ADAPTER=QUALIFIED_FOR_CORRECTNESS
```

The exact historical adapter source/patch bytes have not yet been recovered into this fork. See `../../patches/joseph/checkpoint-nvfp4-spark/README.md`. The SHA256 above is retained as the identity gate for recovering the exact artifact rather than recreating it from memory.

## First correctness result after adaptation

| Gate | Result |
|---|---|
| P1 | PASS |
| P2 | PASS |
| P3 | PASS |
| P4 | PASS |
| P6 | PASS |

## First matched throughput result

### Decode

| Workload | Phase A RedHat | Phase B NVFP4-Spark | Relative delta |
|---|---:|---:|---:|
| C1 code | 57.534 | 56.055 | -2.57% |
| C1 prose | 33.692 | 33.613 | -0.23% |
| C1 structured | 83.056 | 80.263 | -3.36% |

### Prefill

| Prompt size | Phase A RedHat | Phase B NVFP4-Spark | Relative delta |
|---|---:|---:|---:|
| 8K | 2243.23 | 2207.816 | -1.58% |
| 32K | 2408.24 | 2347.310 | -2.53% |
| 64K | 2387.68 | 2322.489 | -2.73% |

At this initial substitution stage, NVFP4-Spark was therefore approximately 0–3% slower in the matched RiNGSiDE workloads. This entry does not attribute that difference to a single kernel family.

## Memory / swap impact

Minimum observed `MemAvailable`:

| Rank | Phase A RedHat | Phase B NVFP4-Spark | Approx. improvement |
|---|---:|---:|---:|
| rank0 | ~1.19 GiB | 2,133,920 kB (~2.04 GiB) | ~+0.85 GiB |
| rank1 | ~0.79 GiB | 2,557,140 kB (~2.44 GiB) | ~+1.65 GiB |

Page-out behavior:

| Rank | Phase A | Phase B |
|---|---:|---:|
| rank0 `pswpout` delta | +67,165 | 0 |
| rank1 `pswpout` delta | +38,330 | 0 |

The checkpoint substitution materially improved RAM headroom and eliminated observed benchmark page-out in this run.

## KV capacity

The fixed RiNGSiDE KV profile remained unchanged:

- configured KV budget = 10.5 GiB/rank
- blocks = 408
- target shared capacity = 1,880,064 tokens
- drafter logical capacity = 470,016 tokens

DFlash2 remained ON. Display-KV was OFF.

## DFlash2 telemetry

Phase B:

- K = 7
- acceptance = 51.71339563862929%
- accepted tokens/speculative step = 1.8863636363636365

This was higher acceptance than the Phase A historical 39.68% value, although Phase A's comparable accepted-tokens/spec-step value is `UNKNOWN`.

## Compatibility materialization observed

For MLA compatibility, the adapter/runtime materialized the following per rank:

| Tensor | Bytes/rank |
|---|---:|
| `q_a_proj` | 3,047,424 |
| `kv_a_proj_with_mqa` | 1,015,808 |
| `q_b_proj` | 12,189,696 |
| `kv_b_proj` | 8,126,464 |
| **Total** | **24,379,392 B = 23.25 MiB/rank** |

This 23.25 MiB/rank allocation is too small by itself to explain the observed 2–3% workload slowdown. Later profiling showed that several MXFP8 attention/KDA/MLA/Indexer sources were materialized to BF16 runtime targets, but no valid per-family attribution was available at this milestone.

## Historical evidence path

The formal Phase B comparison report was recorded at:

```text
/home/joseph/ai/projects/glm53-ringside-tp2-evidence/
b-adapted-20260929/PHASE_A_VS_B_ADAPTED_FINAL.md
```

That filesystem path is retained as historical provenance; the report itself is not claimed to be present in this Git repository unless separately imported.

## Disposition

NVFP4-Spark was accepted as a usable RiNGSiDE TP2 target checkpoint after the generalized checkpoint-contract adaptation. Later tuning work must be recorded separately rather than rewriting this initial substitution result.
