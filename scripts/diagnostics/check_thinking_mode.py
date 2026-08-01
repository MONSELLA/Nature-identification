"""Pre-flight: is chat-template "thinking" mode OFF for every model in a run?

Usage:
    python scripts/diagnostics/check_thinking_mode.py \
        Qwen/Qwen3.5-9B google/gemma-4-12B-it \
        mistralai/Ministral-3-8B-Instruct-2512-BF16 \
        lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct OpenGVLab/InternVL3_5-8B

Loads each model's TOKENIZER ONLY (a few MB — no weights, no GPU), renders a
sample prompt with thinking off and on, and reports whether the two differ.

WHY THIS MATTERS FOR THE CROSS-MODEL COMPARISON
Thinking mode is not a free capability boost, it is a CONFOUND: a model allowed
to reason before answering gets extra inference-time compute the others don't,
so any measured difference conflates "better at nature identification" with
"was allowed to think first". This pipeline's schemas already give every model
the same explicit reasoning fields (ObjectExtractionResponse.reasoning,
TaxonomyResponse's two-step nature_reasoning/sub_axes_reasoning), so CoT
belongs there — inside the measured protocol, identical across families.

It also breaks the pipeline mechanically. Qwen3.5-9B renders a prompt ending in
an already-open '<think>' block; under guided decoding the model is then forced
into JSON from inside that block and cannot close it (measured ~10% extraction
parse-failure vs <1% for a non-thinking model at identical settings), while the
free-form caption call has no grammar at all, so chain-of-thought text lands in
the caption itself.

VERDICTS
  OK (no thinking)   - template never emits <think>; nothing to disable.
  OK (switched off)  - template emits <think> by default, and a switch this
                       codebase knows about actually suppresses it.
  ACTION NEEDED      - template emits <think> with NO switch this codebase
                       recognizes, or the switch does not change the rendered
                       prompt. That model would reason where the others cannot,
                       so the comparison is not fair until it is handled.
"""

import sys

sys.path.insert(0, ".")
from src.models.vlm_models import THINKING_SWITCH_NAMES, find_thinking_switches  # noqa: E402

SAMPLE = [{"role": "user", "content": "Describe this image, including any text."}]


def check(model_name):
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        return f"❓ {model_name}\n     could not load tokenizer: {type(e).__name__}: {e}"

    template = tok.chat_template or ""
    switches = find_thinking_switches(template)

    def render(**kw):
        try:
            return tok.apply_chat_template(SAMPLE, tokenize=False,
                                            add_generation_prompt=True, **kw)
        except Exception as e:
            return f"<render failed: {type(e).__name__}: {e}>"

    default = render()
    emits_by_default = "<think>" in default

    lines = [f"   {model_name}",
             f"     switches found : {switches or 'none'}",
             f"     emits <think> by default: {emits_by_default}"]

    if not emits_by_default and not switches:
        return "✅ OK (no thinking)\n" + "\n".join(lines)

    if switches:
        off = render(**{n: False for n in switches})
        off_emits = "<think>" in off
        lines.append(f"     with switch(es) OFF, emits <think>: {off_emits}")
        lines.append(f"     prompt changed when toggled: {off != default}")
        if not off_emits:
            return "✅ OK (switched off)\n" + "\n".join(lines)
        return "🚨 ACTION NEEDED — switch did not suppress <think>\n" + "\n".join(lines)

    lines.append(f"     no known switch (looked for {list(THINKING_SWITCH_NAMES)})")
    return "🚨 ACTION NEEDED — thinking on, no switch to disable it\n" + "\n".join(lines)


def main():
    models = sys.argv[1:]
    if not models:
        print(__doc__)
        sys.exit(1)
    results = [check(m) for m in models]
    print("\n" + "=" * 70)
    for r in results:
        print(r)
        print("-" * 70)
    if any(r.startswith("🚨") for r in results):
        print("\n🚨 At least one model would reason where others cannot — the "
              "cross-family comparison is NOT fair until that is resolved.")
        sys.exit(1)
    print("\n✅ All models render without a thinking block — comparison is fair "
          "on this axis.")


if __name__ == "__main__":
    main()
