# Direct MXFP8 Runtime Equivalence Investigation — PENDING

**Status:** `PENDING`  
**Archived:** 2026-10-02  
**Decision:** Pause the Direct MXFP8 runtime-equivalence investigation. Do not continue into H10+. Preserve the validated evidence, known causal findings, unresolved boundary, and an explicit resume point.

> This investigation concerns only the experimental Direct MXFP8 loader/preparation path. The Formal RiNGSiDE Phase B serving profile was not changed.

## 1. Objective

The experimental goal was to take checkpoint-native MXFP8 non-MoE projections and prepare them directly into the existing RiNGSiDE Dual serving representation:

```text
MXFP8 checkpoint source
        ↓
Direct preparation
        ↓
existing RiNGSiDE Dual serving representation
```

The intended benefit was to avoid full intermediate BF16 source materialization while preserving all of the following:

- Formal BASE request-time execution path;
- target numerical output;
- DFlash behavior;
- existing DualPath / Marlin serving semantics.

Direct target equivalence was **not achieved** before this lane was paused.

## 2. Frozen numerical reference

The investigation used one fixed 24-token request:

```text
PROMPT_TOKEN_COUNT=24
PROMPT_TOKEN_IDS_HASH=71ef84276af9f966650b216cb848d50ae58a5aa472680e625eba33352b8587cd
```

Formal BASE initial target:

```text
TOKEN=154842
LOGITS_SHA256=16665cc6f59c927aa20eae58e768ca32d364a729f57eb924bda6675a03a9f662
```

Original source-normalized DIRECT:

```text
TOKEN=25136
LOGITS_SHA256=3f7384eff32eb13b429f5e55afbff7f5e8d9e53c404343df6a2d49df1f74bf4b
```

The BASE/DIRECT divergence was deterministic and repeatable.

## 3. Elimination ledger

### H1 — CUDA graph path

```text
H1_RESULT=EXCLUDED
RESULT=DIVERGENCE_IS_GRAPH_INDEPENDENT
```

With CUDA graph disabled on both candidates, eager execution remained repeatable and different:

```text
BASE   = 154842 / 16665cc6...
DIRECT = 25136  / 3f7384ef...
```

Therefore CUDA graph capture/replay is not a necessary cause.

### H2 — Environment

```text
H2_RESULT=EXCLUDED
RESULT=NO_RUNTIME_RELEVANT_ENV_DIVERGENCE
REQUEST_TIME_DIRECT_ENV_DIFF_COUNT=0
```

All observed environment differences were proven load-only or diagnostic-only.

### H3 — Startup / effective runtime configuration

```text
H3_RESULT=EXCLUDED
RESULT=NO_RUNTIME_RELEVANT_CONFIG_DIVERGENCE
```

Each candidate had 137 normalized config entries. Three differences were classified as expected source preparation, Direct load-time method selection, or diagnostic instrumentation. No request-time configuration difference remained.

### H4 — Effective runtime source

```text
H4_RESULT=EXCLUDED
RESULT=NO_REQUEST_TIME_SOURCE_DIVERGENCE
```

The static effective-source/import audit covered 9,842 entries per candidate. Relevant installed compiled extensions and shared JIT artifacts matched. No request-time semantic source difference remained. The effective KDA source was also normalized to the same source identity.

### H5 — Direct persistent state leakage

H5 found a real persistent BF16 state difference in Sparse MLA indexer weights:

```text
self_attn.indexer.wk_weights_proj.weight
11 unique parameters
22 TP-rank-local copies
```

Thus:

```text
H5_RESULT=SUPPORTED
RESULT=DIRECT_PERSISTENT_STATE_LEAKAGE_FOUND
```

A causal normalization made all 22 rank-local copies byte-identical to BASE, but STEP0 remained exactly the original DIRECT result:

```text
TOKEN=25136
LOGITS_SHA256=3f7384eff32eb13b429f5e55afbff7f5e8d9e53c404343df6a2d49df1f74bf4b
```

Therefore:

```text
H5_CAUSAL_RESULT=NOT_CAUSAL_FOR_STEP0_OUTPUT
```

This is a real loader/state defect, but it did not explain the frozen STEP0 divergence.

### H6 — Request-time module instance semantic state

```text
H6_RESULT=EXCLUDED
RESULT=NO_REQUEST_TIME_MODULE_STATE_DIVERGENCE
```

Layer 0 module tree and quant-method instance state matched. Recorded attribute differences were load-history or diagnostic-only.

### H7

```text
H7_STATUS=SUBSUMED_BY_H6
```

The DualPath quant-method internal instance-state question was already covered by H6 and was not run separately.

### H8 — Marlin prepared/runtime metadata

```text
H8_RESULT=EXCLUDED
RESULT=NO_MARLIN_RUNTIME_METADATA_DIVERGENCE
```

For the first Layer 0 Marlin invocation, BASE and DIRECT matched on input metadata, packed-weight metadata, scale/global-scale metadata, workspace metadata, auxiliary state, compiled extension, and effective call arguments.

Observed effective call included:

```text
M/N/K=24/12608/4096
quant_type=float4_e2m1f
quant_type_id=562949953487106
use_atomic_add=false
use_fp32_reduce=true
```

H8 deliberately did not compare the large weight/scale payload contents; that was deferred to H9.

## 4. H9 — First request-time numerical-state divergence

A bounded SHA-256 sweep of Layer 0 request-time numerical state found:

```text
41 tensors per candidate per TP rank
82 rank-matched comparisons
78 equal
4 different
```

The four differences were:

```text
rank0 self_attn.f_b_proj.weight
rank0 self_attn.g_b_proj.weight
rank1 self_attn.f_b_proj.weight
rank1 self_attn.g_b_proj.weight
```

Tensor metadata matched; contents differed.

Classification:

```text
LOAD_DEQUANTIZATION_DIFFERENCE
```

Therefore:

```text
H9_RESULT=SUPPORTED
RESULT=LAYER0_NUMERICAL_STATE_DIVERGENCE_FOUND
```

## 5. Layer 0 f_b/g_b causal normalization

The construction difference was proven.

BASE:

```text
MXFP8 E4M3 weight
+ U8/E8M0 weight_scale
→ apply scale / dequantize
→ BF16
→ assign parameter
```

Original DIRECT:

```text
MXFP8 E4M3 weight
→ required weight_scale omitted
→ ordinary loader
→ BF16 parameter
```

The intervention normalized only Layer 0 `f_b_proj.weight` and `g_b_proj.weight`.

Hard gates:

```text
4/4 affected rank-local parameters == BASE byte-exact
78/78 non-target Layer 0 state hashes unchanged
```

The normalized DIRECT output became:

```text
TOKEN=46461
LOGITS_SHA256=7a6d3b868a99dd99ba2bf854e60e0b100fe194ad5ccd946fd0a485461e02bb9a
```

This differed from both the original DIRECT result and BASE. Therefore:

```text
H9_CAUSAL_RESULT=CONFIRMED_CONTRIBUTOR_NOT_SUFFICIENT
RESULT=LAYER0_FB_GB_STATE_CONTRIBUTES_BUT_OTHER_DIVERGENCE_REMAINS
```

This is the strongest causal result from the investigation: the Layer 0 `f_b/g_b` MXFP8 scale-route defect is causally visible in STEP0, but is not by itself sufficient to restore BASE equivalence.

## 6. Model-wide f_b/g_b defect family

The same construction defect was then audited model-wide.

```text
68 semantic f_b/g_b projections
34 KDA layers
```

All 68 semantic projections matched the same defect pattern:

```text
MXFP8 weight exists
U8/E8M0 weight_scale exists
BASE applies scale before BF16 assignment
DIRECT drops equivalent scale application before BF16 assignment
```

Runtime state comparison:

```text
136 rank-local BASE/DIRECT comparisons
136 different
0 equal
```

Therefore the Layer 0 defect is not an isolated instance; it is a model-wide loader defect family.

## 7. Model-wide normalization attempt

The corrected per-pair normalization path successfully handled:

```text
60 / 68 semantic targets per TP rank
```

with:

```text
0 unexpected hits
0 non-f_b/g_b hits
```

Eight semantic targets per rank were missed:

```text
layers 5, 6, 8, 9
× f_b_proj / g_b_proj
```

This corresponds to 16 rank-local parameters.

The model-wide expansion therefore remained:

```text
H9_FB_GB_EXPANSION_RESULT=INCONCLUSIVE
RESULT=MODEL_WIDE_FB_GB_HIT_SET_INVALID
```

The 136-state equality gate and model-wide normalized STEP0 were not run.

## 8. Missed-eight upstream route investigation

Static inspection established:

- layers 5, 6, 8, and 9 are KDA `Glm5NextDecoderLayer / Glm5NextLinearAttention` layers;
- layer 7 is MLA and has no `f_b/g_b`, so it is not a same-family control;
- all eight weight/scale pairs exist in `model-hf-nonexpert-00004-of-00004.safetensors`.

Live tracing established the actual upstream boundary:

1. all 16 raw keys exist in the checkpoint index/header;
2. shard `model-hf-nonexpert-00004-of-00004.safetensors` is present in the InstantTensor input-file list;
3. runtime `safe_open(...).tensors()` does **not** enumerate those 16 entries;
4. therefore they never reach the prefix mapper, `AutoWeightsLoader`, `Glm5NextModel.load_weights`, or the existing normalization hook;
5. same-family controls in layers 4 and 10 do reach iterator → mapper → AutoWeightsLoader → normalization hook.

Classification:

```text
MISSED_8_COMMON_ROUTE_CAUSE_PROVEN
CLASS=EARLIER_LOADER_INTERCEPT
```

Current proven boundary:

```text
checkpoint header
        ↓
InstantTensor input shard present
        ↓
InstantTensor iterator / name enumeration
        ↓
[layers 5,6,8,9 f_b/g_b entries disappear here]
        ↓
prefix mapper
        ↓
AutoWeightsLoader
        ↓
normalization hook
```

The recommended future repair point is the InstantTensor iterator/name-enumeration boundary, using an exact raw-key allowlist for only the unresolved 16 keys while leaving the already-working 60-target path unchanged.

**No InstantTensor fix was implemented before this lane was paused.**

## 9. Work not completed

The following gates remain unrun:

- exact `68/68` semantic normalization hits per rank;
- `136/136` affected BF16 state hashes equal to BASE;
- final non-target isolation guard;
- model-wide normalized frozen STEP0;
- DFlash smoke / acceptance qualification after target equivalence;
- benchmark;
- correctness suite.

Therefore full model-wide causal sufficiency remains:

```text
UNKNOWN
```

What is proven is narrower:

```text
Layer0 f_b/g_b scale-route defect is a causal STEP0 contributor.
The same construction defect exists across all 68 model-wide f_b/g_b projections.
Full model-wide sufficiency is not yet proven.
```

## 10. SparkB protected-file identity drift — unresolved provenance gate

Protected path:

```text
src/tp2/vllm/model_executor/layers/quantization/glm53_dual_fp8_dense.py
```

Last observed host hashes:

```text
SparkA SHA256=7ca94d1beb1a1b0675b456b0720ef4d96b960e0f312c95b3f9a34456cd740aa4
SparkB SHA256=4e9c12fb110b3eb56e7b4e8d090c8d4df395235ce773f290062c87b1d0b9fe9e
```

The provenance and effective-runtime relevance of the SparkB drift were not fully audited before pausing. The final missed-eight trace did not mount or write this protected file.

Before resuming H9 causal work, audit:

- provenance of the SparkB variant;
- whether any prior H9 runtime actually imported/mounted that host copy;
- whether any existing H9 evidence needs to be invalidated or rerun.

Do **not** blindly `git checkout`, restore, or copy the SparkA file over SparkB before provenance is established.

## 11. Graph-safe activation-trace branch

An earlier attempt to qualify a graph-capture-safe cryptographic activation fingerprint did not find an existing usable device SHA-256 primitive in the pinned runtime/tooling.

```text
H26_KDA_INTERNAL_TRACE=INCONCLUSIVE
RESULT=GRAPH_SAFE_CRYPTO_HASH_NOT_AVAILABLE
```

This is not the preferred resume path for the Direct loader investigation.

## 12. Evidence roots

The detailed receipts and raw evidence remain in the experiment evidence tree. Important roots include:

```text
ai/projects/glm53-ringside-tp2-evidence/native-mxfp8-origin-5family-20260930/h1-cuda-graph-single-variable-elimination-20261001/
ai/projects/glm53-ringside-tp2-evidence/native-mxfp8-origin-5family-20260930/h2-environment-difference-audit-20261001/
ai/projects/glm53-ringside-tp2-evidence/native-mxfp8-origin-5family-20260930/h3-startup-runtime-config-audit-20261001/
ai/projects/glm53-ringside-tp2-evidence/native-mxfp8-origin-5family-20260930/h4-effective-runtime-source-manifest-audit-20261001/
ai/projects/glm53-ringside-tp2-evidence/native-mxfp8-origin-5family-20260930/h5-direct-request-time-branch-leakage-audit-20261001/
ai/projects/glm53-ringside-tp2-evidence/native-mxfp8-origin-5family-20260930/h5-indexer-state-causal-normalization-20261001/
ai/projects/glm53-ringside-tp2-evidence/native-mxfp8-origin-5family-20260930/h6-request-time-module-state-audit-20261001/
ai/projects/glm53-ringside-tp2-evidence/native-mxfp8-origin-5family-20260930/h8-marlin-prepared-runtime-metadata-identity-20261001/
ai/projects/glm53-ringside-tp2-evidence/native-mxfp8-origin-5family-20260930/h9-layer0-request-time-numerical-state-identity-20261001/
ai/projects/glm53-ringside-tp2-evidence/native-mxfp8-origin-5family-20260930/h9-layer0-fb-gb-causal-normalization-20261001/
ai/projects/glm53-ringside-tp2-evidence/native-mxfp8-origin-5family-20260930/h9-fb-gb-model-wide-same-bug-expansion-20261001/
ai/projects/glm53-ringside-tp2-evidence/native-mxfp8-origin-5family-20260930/h9-fb-gb-model-wide-corrected-normalization-20261001/
ai/projects/glm53-ringside-tp2-evidence/native-mxfp8-origin-5family-20260930/h9-fb-gb-missed-8-route-audit-20261001/
ai/projects/glm53-ringside-tp2-evidence/native-mxfp8-origin-5family-20260930/h9-fb-gb-missed-8-live-route-trace-20261001/
```

These paths are evidence references; they are not a claim that every local evidence artifact is committed into this repository.

## 13. Final status

```text
DIRECT_MXFP8_LANE_STATUS=PENDING
FORMAL_PHASE_B_CHANGED=NO
DFLASH_FINAL_VALIDATION=NOT_RUN
BENCHMARK=NOT_RUN
CORRECTNESS_SUITE=NOT_RUN

DIRECT_TARGET_EQUIVALENCE=NOT_ACHIEVED
PROVEN_CAUSAL_BUG=LAYER0_FB_GB_MXFP8_SCALE_ROUTE_DROP
MODEL_WIDE_SAME_BUG=68/68_SEMANTIC_PROJECTIONS_CONFIRMED
FULL_MODEL_CAUSAL_SUFFICIENCY=UNKNOWN
NEXT_LOADER_BOUNDARY=INSTANTTENSOR_ITERATOR_NAME_ENUMERATION
UNRESOLVED_RAW_KEYS=16
UNRESOLVED_SEMANTIC_TARGETS=8
PENDING=YES
```

## 14. Resume point

Do **not** rerun H1-H8 and do **not** jump directly to H10.

Resume in this order:

### Step 0 — provenance gate

Audit the SparkB protected `glm53_dual_fp8_dense.py` drift and prove its effective-runtime relevance to prior H9 experiments.

### Step 1 — exact InstantTensor recovery

Implement exact recovery at the InstantTensor iterator/name-enumeration boundary for only:

```text
layers 5,6,8,9
f_b_proj.weight
f_b_proj.weight_scale
g_b_proj.weight
g_b_proj.weight_scale
```

Total unresolved raw keys: `16`.

Preserve the existing 60-target path.

### Step 2 — normalization hit-set gate

Require per rank:

```text
68/68 semantic normalization hits
0 missed
0 unexpected
```

### Step 3 — numerical-state equality gate

Require:

```text
136/136 affected rank-local BF16 hashes == BASE
```

and all non-target isolation guards unchanged.

### Step 4 — frozen eager STEP0

Run the frozen 24-token eager request twice.

Required BASE identity:

```text
TOKEN=154842
LOGITS_SHA256=16665cc6f59c927aa20eae58e768ca32d364a729f57eb924bda6675a03a9f662
```

### Step 5 — decision

If exact BASE target equivalence is restored, proceed to DFlash smoke / acceptance qualification.

If exact BASE target equivalence is **not** restored, stop the Direct lane again unless a new independently compelling reason justifies further investigation.

## 15. Archive decision

This lane is paused because:

1. reproducible causal evidence has been obtained;
2. the model-wide loader defect family has been identified;
3. the upstream boundary for the last eight semantic targets has been localized;
4. remaining work requires additional loader surgery;
5. the marginal value of continuing the serial investigation is currently lower than other RiNGSiDE tuning work.

Final archive state:

```text
STATUS=PENDING
RESUME_POINT=DOCUMENTED
EVIDENCE_PRESERVED=YES
DO_NOT_AUTO_RESUME=YES
```
