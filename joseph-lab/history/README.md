# RiNGSiDE TP2 History

This directory is the durable milestone history for the local two-DGX-Spark RiNGSiDE lane.

It is intentionally selective: routine probes and noise-level tuning attempts do not belong here unless they materially explain a later engineering decision.

| ID | Date | Event | From | To | Status |
|---|---|---|---|---|---|
| 0001 | 2026-09-29 | Initial RiNGSiDE TP2 validation | upstream TP2 release profile | measured RedHat NVFP4 baseline | QUALIFIED as substitution baseline |
| 0002 | 2026-09-29 | Target checkpoint substitution | RedHatAI GLM-5.3-Flash-NVFP4 | local-inference-lab GLM-5.3-Flash-NVFP4-Spark | QUALIFIED for correctness and serving |

## Status meanings

- `QUALIFIED`: passed the stated gate and was accepted for the purpose described in that history entry.
- `REJECTED`: tested and deliberately not adopted.
- `SUPERSEDED`: historically valid but replaced by a later qualified state.
- `INCONCLUSIVE`: evidence was insufficient for a causal or adoption decision.

## Evidence rule

Measured results, static/source findings, and historical/public data must remain distinguishable. Missing historical values are recorded as `UNKNOWN` rather than reconstructed from assumptions.
