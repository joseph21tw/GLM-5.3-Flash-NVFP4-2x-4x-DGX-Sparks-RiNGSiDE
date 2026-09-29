# Joseph RiNGSiDE Lab Records

This directory records the local TP2 integration history built on top of the upstream RiNGSiDE repository.

## Branch policy

- `main` is kept as an upstream mirror and should not contain Joseph-specific runtime changes.
- `tp2-qualified` is the local qualified TP2 integration line.
- Experimental changes should be developed on separate branches and merged only after qualification.

## Record policy

`history/` is not a complete experiment log. It records:

1. the first validated state of a serving configuration;
2. later changes that materially changed the runtime, checkpoint, memory/KV behavior, correctness contract, or qualified serving profile;
3. rejected or superseded work only when the result is important enough to explain why that path is not used.

Each history entry should preserve, when available:

- upstream/base commit;
- target and draft checkpoint revisions;
- exact runtime/profile assumptions;
- the reason for the change;
- files/overlays/patches changed to make it work;
- first successful correctness result;
- first matched decode/prefill result;
- RAM, swap, and KV capacity impact;
- links or paths to evidence receipts;
- final disposition (`QUALIFIED`, `REJECTED`, `SUPERSEDED`, or `INCONCLUSIVE`).

The history files describe what happened. Git commits, source files, and patch files are the authoritative record of what bytes changed.

## Current historical base

The local TP2 work documented here started from upstream RiNGSiDE commit:

`6606f7638aff044a8170e414818477679934a1c3`

The first checkpoint-substitution work kept the RiNGSiDE TP2 serving profile fixed and changed only the target checkpoint from the upstream RedHat NVFP4 target to:

`local-inference-lab/GLM-5.3-Flash-NVFP4-Spark`

revision:

`a608241037e4c2565356bff7ca293f2133888f88`

See `history/` for the measured baseline and substitution record.
