# Post-publication routing rerun: 1 success out of 3

Exactly seeds 0, 1 and 2, each from a separate original-source copy at repository commit fbdcb10b289001baf2c0e444bb963e421262a4fd. Native 6-bit weights unchanged. A separate venv used MLX-LM 0.32.0 with the supplied server/parser patch, temperature 1.0, 16,384 completion tokens per turn, up to 12 turns, and reasoning replayed as reasoning_content. Fixed routing regressions and all repository tests ran in the same pinned offline unprivileged container. Original checkout and original inference venv were not modified.

| Seed | Outcome | Final tests |
| --- | --- | --- |
| 0 | Failed; 12-turn budget exhausted | 8 regression failures |
| 1 | Passed | All 36 tests passed |
| 2 | Failed; syntax error and malformed repair call | 6 import errors caused by syntax error |

Seed 2's invalid JSON was returned to the client as tool-call text by the patched server. The harness still classified a no-call stop response as model_finished and ended without repairing the syntax error. Its final tests correctly report failure. The patch exposes the lost call; it does not make invalid quotes valid or establish automatic client repair.

All transcripts, before/after source, generated patches, test outputs, parser warnings and receipts are retained. runtime.json records patched source and patch checksums. This is one authored task with three seeds, not a general agent reliability estimate. Multiple settings changed from the initial trials, so the result cannot identify which change caused the successful run.
