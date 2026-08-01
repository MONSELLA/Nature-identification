"""Is a model's extraction parse-failure rate caused by TRUNCATION?

Usage:  python scripts/diagnostics/extraction_failure_shape.py <vlm_responses_*.jsonl>

Extraction runs under vLLM GUIDED DECODING (StructuredOutputsParams(json=...)),
which makes malformed JSON impossible by construction — the model literally
cannot emit a token that violates ObjectExtractionResponse. So the ONLY way
json.loads can fail is a generation cut off before the JSON closed, i.e. it hit
--max_new_tokens_extraction.

The failed records keep no raw text (extract_objects_batch discards it), so the
failures cannot be measured directly. Instead this measures the SUCCESSES: if
their output-length distribution is piled up against the token ceiling, the
failures are the tail that fell off the end (-> raise the budget). If the
successes sit comfortably below it, truncation is NOT the story and the cause
is elsewhere (e.g. the model burning its budget before the JSON starts).

Token counts are a ~3.7 chars/token approximation, so compare SHAPE across
models rather than trusting absolute values.
"""
import json, sys, statistics as st
path = sys.argv[1]
recs = [json.loads(l) for l in open(path)]
recs = [r for r in recs if r.get("record_type") not in ("header", "footer")]
ok  = [r for r in recs if not r.get("extraction_parse_failed")]
bad = [r for r in recs if r.get("extraction_parse_failed")]
print(f"{path.split('/')[-1]}: {len(recs)} images | extraction parse-fail {len(bad)} ({len(bad)/len(recs):.1%})")
rl = [len(r.get("extraction_reasoning") or "") for r in ok]
ol = [len(r.get("objects") or []) for r in ok]
# reconstruct roughly what the model had to emit under guided decoding
tot = [len(json.dumps({"reasoning": r.get("extraction_reasoning") or "",
                       "objects": r.get("objects") or []})) / 3.7 for r in ok]
tot.sort()
if tot:
    n = len(tot)
    print(f"  reasoning chars : median {st.median(rl):.0f}  max {max(rl)}")
    print(f"  objects/image   : median {st.median(ol):.1f}  max {max(ol)}")
    print(f"  approx OUTPUT tokens: median {st.median(tot):.0f} | p90 {tot[int(.9*n)]:.0f} "
          f"| p99 {tot[min(int(.99*n),n-1)]:.0f} | MAX {tot[-1]:.0f}")
    for cap in (256, 512, 768):
        print(f"    within {cap}: {sum(1 for t in tot if t < cap)/n:.1%}")
cap_ratio = sum(1 for t in tot if t > 450) / len(tot) if tot else 0
print(f"  >>> {'TRUNCATION-SHAPED (mass piled at the ceiling)' if cap_ratio > .10 else 'not truncation-shaped'}"
      f"  ({cap_ratio:.1%} of successes above ~450 approx-tok)")
