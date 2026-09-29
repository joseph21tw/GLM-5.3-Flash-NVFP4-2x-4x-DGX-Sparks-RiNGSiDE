# NVFP4-Spark checkpoint compatibility adaptation

This directory preserves the exact compatibility overlay used to run:

`local-inference-lab/GLM-5.3-Flash-NVFP4-Spark`

revision:

`a608241037e4c2565356bff7ca293f2133888f88`

on the RiNGSiDE TP2 runtime pinned at:

`6606f7638aff044a8170e414818477679934a1c3`

## Recovery status

`RECOVERED_EXACT_RUNTIME_OVERLAY`

The 2026-09-29 evidence archive was recovered and inspected. It does not contain a literal `.patch` or `.diff` for this adapter. The actual checkpoint-enablement artifact was a complete `model.py` file mounted read-only over the installed vLLM file on both ranks.

Exact recovered artifact in this repository:

```text
exact-overlay/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/model.py
```

Runtime mount target:

```text
/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/model.py
```

Qualified SHA256:

```text
4f3d1462905f52a339c67dba73c61003d501e2e5f845167a1ff934c6679a1d43
```

The recovered file hashes to exactly the recorded qualified runtime identity above.

## Why an adaptation was required

The target checkpoint initially failed to load with:

```text
KeyError:
layers.0.self_attn.in_proj_qkvbfg_a.weight_scale
```

The recovered generalized checkpoint-contract adapter handles the KDA / Indexer / MLA checkpoint-layout differences required by this checkpoint without importing the G3 runtime wholesale.

Qualified receipts reported:

```text
KDA_ADAPTER=PASS
INDEXER_ADAPTER=PASS
MLA_HANDLER rank0/rank1 PASS
A_SEMANTIC_BEHAVIOR_UNCHANGED=YES
B_ADAPTED_LOAD=PASS
GENERALIZED_ADAPTER=QUALIFIED_FOR_CORRECTNESS
```

## Proof that this exact file was used

The archived launch contract describes a single read-only bind mount of this `model.py`.

Both rank-specific compose overrides used the same host source and the same runtime target. The archived prelaunch parity and parity-qualified load receipts recorded:

```text
ADAPTER_SHA256_RANK0=4f3d1462905f52a339c67dba73c61003d501e2e5f845167a1ff934c6679a1d43
ADAPTER_SHA256_RANK1=4f3d1462905f52a339c67dba73c61003d501e2e5f845167a1ff934c6679a1d43
ADAPTER_HASH_MATCH=YES
MODEL_PY_RUNTIME_SHA256_RANK0=4f3d1462905f52a339c67dba73c61003d501e2e5f845167a1ff934c6679a1d43
MODEL_PY_RUNTIME_SHA256_RANK1=4f3d1462905f52a339c67dba73c61003d501e2e5f845167a1ff934c6679a1d43
IMPORT_PATH_MATCH=YES
OVERLAY_MOUNT_MATCH=YES
PRELAUNCH_OVERLAY_PARITY=PASS
```

The archived launch labels identify the adapter as:

```text
ringside.tp2.adapter=generalized-mxfp8-checkpoint-contract
```

## Historical evidence source

Recovered from the user-supplied archive of:

```text
/home/joseph/ai/projects/glm53-ringside-tp2-evidence/
b-adapted-20260929/
```

Relevant archived files include:

```text
overlay/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/model.py
launch/LAUNCH_CONTRACT.md
launch/rank0/compose.override.yaml
launch/rank1/compose.override.yaml
contracts/prelaunch-overlay-parity.txt
logs/B_ADAPTED_PARITY_LOAD_GATE.txt
PHASE_A_VS_B_ADAPTED_FINAL.md
report/B_ADAPTER_REPORT.md
```

## Artifact type

This was an **overlay source replacement**, not a standalone patch file. Do not rewrite the historical record to imply that a `.patch` existed at qualification time.

A derived diff against the exact RiNGSiDE preimage may be generated later for review convenience, but such a derived diff must be labelled as derived and must not replace this exact recovered runtime artifact as the authoritative historical bytes.

## Non-goal

This compatibility adaptation must not be conflated with later G3 MXFP8 lifecycle/performance work such as `c998406f2dbf48476c42b53c086a77bb0036b2a4`. That later work was identified as a possible semantic-port candidate during performance analysis, but it was not part of the initial Phase B checkpoint-enablement step recorded here.