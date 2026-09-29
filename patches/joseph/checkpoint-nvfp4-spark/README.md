# NVFP4-Spark checkpoint compatibility adaptation

This directory is reserved for the exact compatibility patch/overlay used to run:

`local-inference-lab/GLM-5.3-Flash-NVFP4-Spark`

revision:

`a608241037e4c2565356bff7ca293f2133888f88`

on the RiNGSiDE TP2 runtime pinned at:

`6606f7638aff044a8170e414818477679934a1c3`

## Why an adaptation was required

The target checkpoint initially failed to load with:

```text
KeyError:
layers.0.self_attn.in_proj_qkvbfg_a.weight_scale
```

A generalized checkpoint-contract adapter was then used for the KDA / Indexer / MLA checkpoint-layout differences.

Qualified receipts reported:

```text
KDA_ADAPTER=PASS
INDEXER_ADAPTER=PASS
MLA_HANDLER rank0/rank1 PASS
A_SEMANTIC_BEHAVIOR_UNCHANGED=YES
B_ADAPTED_LOAD=PASS
GENERALIZED_ADAPTER=QUALIFIED_FOR_CORRECTNESS
```

The final adapter deployed to both ranks had SHA256:

```text
4f3d1462905f52a339c67dba73c61003d501e2e5f845167a1ff934c6679a1d43
```

Parity gates also passed:

```text
PRELAUNCH_OVERLAY_PARITY=PASS
ADAPTER_HASH_MATCH=YES
IMAGE_DIGEST_MATCH=YES
IMPORT_PATH_MATCH=YES
```

## Current recovery status

`PATCH_BYTES_NOT_RECOVERED`

The historical records currently available to this repository preserve the adapter identity, purpose, and qualification results, but not the complete source/patch bytes.

Do **not** recreate a patch from memory and label it as the historical qualified adapter.

The exact artifact should be recovered from the DGX evidence/worktree and accepted here only if its SHA256 and provenance match the recorded qualified artifact.

Primary historical evidence location:

```text
/home/joseph/ai/projects/glm53-ringside-tp2-evidence/
b-adapted-20260929/
```

Formal comparison report:

```text
PHASE_A_VS_B_ADAPTED_FINAL.md
```

## Recovery gate

When the exact patch/source is recovered, record at minimum:

1. original filesystem path;
2. source/import path used by rank0 and rank1;
3. SHA256;
4. diff against upstream RiNGSiDE `6606f763...` source;
5. affected runtime files;
6. whether it is an overlay, direct source replacement, or generated adaptation;
7. proof that rank0 and rank1 use identical bytes;
8. the correctness receipt tied to that exact artifact.

Only after those checks should the real patch files be added beside this README.

## Non-goal

This compatibility adaptation must not be conflated with later G3 MXFP8 lifecycle/performance work such as `c998406f2dbf48476c42b53c086a77bb0036b2a4`. That later work was identified as a possible semantic-port candidate during performance analysis, but it was not part of the initial Phase B checkpoint-enablement step recorded here.
