# Bounded Boulder-anchor cloud source bundle — 2026-08-12

## Purpose

The authorised World Climbing video lane already has a bounded cloud verifier:
three 2026 events, up to 12 fresh 60-second windows per event, one-frame-per-
second review, event-scoped checkpoints, and a mandatory dry run before a
Gemini Flash-Lite request.  It is **not** on `comp-v2/main`, so it cannot be
launched from the remote repository yet.

This note fixes the transfer boundary.  A future focused branch must copy only
the files below from source commit `3c166d670762f4350fc4635b23f1790ead016557`,
verify every size and SHA-256, and run the listed checks before it is pushed.
It must not use the broad historical video commit as a proxy for this bundle.

## Required source files

| Path | Bytes | SHA-256 |
| --- | ---: | --- |
| `video_boulder_anchor_discovery.py` | 31,426 | `798b1aad3c2f74b2018fd738cf0ae85bd852f300c28e852e286eee7455d3be9f` |
| `video_boulder_anchor_gemini_compat.py` | 5,239 | `25e68ae1e285c5f7be7ef093655778df55fe263154915bb3c91e3cfeb4b97e1a` |
| `video_boulder_segmentation.py` | 36,473 | `30eb0eb3bf847dca5cf8aa341481ab009dd29bb77b246182a27c676e91e4b2d7` |
| `video_boulder_anchor_verification_merge.py` | 17,393 | `a2818655949d0da4a9cb59fab38ab68b80a81684a8376d2003ddb2a424ee292e` |
| `scripts/run_boulder_anchor_discovery.py` | 29,159 | `105bc9b6a9b1ec9158bec6078be614aa209b44e3d3372e78181571eb8a023f7e` |
| `scripts/validate_boulder_anchor_discovery_packet.py` | 4,588 | `b542665450c64945c4097c468cf3dac34b97b991544dabdcafec769dab0cf4a1` |
| `scripts/validate_boulder_anchor_discovery.py` | 3,654 | `f80b7cbea1145e45d27494a30f2db9f3267e2e49b96d6b7f862c5a19e20d5551` |
| `scripts/merge_boulder_anchor_verification_checkpoints.py` | 990 | `0adfb935356ed834022f4e88b5d7b6785c119619d4f2e5933d1e5ed4bcb467dc` |
| `tests/test_video_boulder_anchor_discovery.py` | 18,699 | `d573e0ec4464d3c01ea2d58f52c1ed24cc87ef899c4751e080402f0086cb7080` |
| `tests/test_boulder_anchor_verification_workflow.py` | 19,715 | `3593e0bee6d1e8582d5f2142ed5c846da51b814cfc7ced7232fbce8454c5a412` |
| `.github/workflows/video-2026-boulder-anchor-discovery.yml` | 4,590 | `c80a37e7a33a5d6a3ad4645652427001bc4e98410de739bdff886410779e5528` |
| `.github/workflows/video-2026-boulder-anchor-verification.yml` | 8,380 | `eec856805bed61fa274feb02b32e9c9206b3ce0b132ae8a482d2d61737b1b5c8` |

The bundle also needs the seven frozen files in
`data/video_boulder_anchor_discovery_2026_v1/`, including the exact
`merged_review_windows.jsonl` hash
`49fdce904c9b5a720c14f38033253378def7b22d95b291ac0cb30b72daa0e3ab`,
and `data/video_2026_source_manifest.csv` hash
`8f6ce23cfd3da6d01ef75dbf795e24cba3040aee8aa143add8b7e69feadc8772`.

## Required acceptance

1. `python scripts/validate_boulder_anchor_discovery_packet.py` passes.
2. Both focused test modules pass.
3. The workflow's mandatory event-scoped dry run passes with `execute=false`.
4. A human checks that `execute=true` is still capped at 12 windows per event,
   uses only the repository secret `GEMINI_API_KEY`, and writes no public
   athlete or coaching claim.

Only after those checks may the workflow be pushed and manually dispatched.
The dispatch remains bounded research acquisition, not model promotion or a
public deployment.

## Local release-candidate evidence

On 2026-08-12 this exact bundle passed the packet validator (`77/77` planned
windows, zero missing/quarantined, all downstream safety gates closed), 31
focused anchor-workflow tests, and the complete clean candidate suite (`56`
tests plus `22` subtests).  The mandatory local dry run for event `1479`
planned 73 verification windows across its four official broadcasts and made
no model request.  It produced a plan only; a plan is intentionally not
validated as an executed checkpoint.
