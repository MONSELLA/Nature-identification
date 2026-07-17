#!/usr/bin/env python3
"""
qualitative_vlm_evaluation.py

Runs a fixed set of open-ended prompts through a VLM (Qwen/Qwen3.5-0.8B) on
the SAME 20-image diagnostic sample already used to compare the closed-set
models (the one built by evaluate_big5.py's select_diagnostic_sample /
persisted to --diagnostic_sample_file). Responses are written into the SAME
persistent --comparison_file JSON the closed-set scripts already write to,
under this model's own --model_id -- so every model's output (closed-set
predictions AND this VLM's free-text answers) lives in one file, indexed by
image filename, exactly like update_comparison_file() in evaluate_big5.py.

WHAT IS THIS SCRIPT FOR? Unlike the main run_vlm_pipeline.py (which produces
scored metrics), this script is purely QUALITATIVE: it asks a VLM a handful of
open-ended questions about a small fixed set of images and just SAVES the raw
text answers for a human to read and compare against the closed-set models'
predictions on the same images. There's no scoring/metric logic here at all —
it's meant for eyeballing "does this model's reasoning actually make sense?"

------------------------------------------------------------------------------
IMPORTANT: Qwen3.5-0.8B is a genuinely multimodal model -- verified, not
guessed. Per the official Hugging Face transformers docs: "Qwen3.5 is Qwen's
natively multimodal foundation model family, trained from scratch on
interleaved text, image, and video tokens." This is a DIFFERENT model family
from the similarly-named text-only Qwen3 LLMs (Qwen3-0.6B, Qwen3-8B, etc.,
loaded via AutoModelForCausalLM) -- easy to confuse given how close the names
are, which is exactly why this script uses AutoModelForImageTextToText, not
AutoModelForCausalLM.

------------------------------------------------------------------------------
LOADING RECIPE -- kept EXACTLY as the user's own already-verified-working code
------------------------------------------------------------------------------
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)

------------------------------------------------------------------------------
INFERENCE FORMAT -- verified against the official Qwen3.5 transformers doc
page (huggingface.co/docs/transformers/model_doc/qwen3_5) and a real
user-reported working example, not guessed:
    messages = [{"role": "user", "content": [
        {"type": "image", "image": <path or URL>},
        {"type": "text", "text": <prompt>},
    ]}]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True,
                                           tokenize=True, return_dict=True,
                                           return_tensors="pt").to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=...)
    # trim the input prefix before decoding (official pattern, batch-safe):
    trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)]
    text = processor.batch_decode(trimmed, skip_special_tokens=True)[0]

The "image" field is passed as a local file PATH STRING (not a pre-loaded PIL
object), matching the officially documented pattern exactly (their example
uses a URL string; a local path string follows the identical convention,
consistent with how Qwen's own VL model family has historically accepted
path/URL/base64 interchangeably in this field).

------------------------------------------------------------------------------
!!! NOT TESTED END-TO-END AGAINST THE REAL MODEL !!!
------------------------------------------------------------------------------
huggingface.co is not reachable from the sandbox this script was written in,
so the message format above is verified against documentation and a working
community example, but NOT executed here. Run a 1-image, 1-prompt smoke test
first (--max_samples 1 --prompts "What is in the background?" --verbose)
before committing to the full 20-image x N-prompt run.

Each of the (default) 4 prompts is asked as an INDEPENDENT single-turn
conversation -- no shared conversation history carries over between prompts
on the same image, since they're logically separate questions, not a dialog.
"""

import os
import sys
import json
import argparse

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

# The fixed set of questions asked about EVERY image in the diagnostic sample
# (unless the user overrides them with --prompts). Each is independent — see
# the module docstring's note about no shared conversation history.
DEFAULT_PROMPTS = [
    "Summarise the background elements in a concise manner.",
    "Summarise the foreground elements in a concise manner.",
    "Identify the primary subject or subjects of this image. State what they are, and if there are multiple, briefly note how they interact. Justify your conclusion based on visual prominence (size, focus, or placement) in 1 to 2 sentences.",
    "Ignoring any foreground subjects or specific objects, what specific place, background environment, or spatial layout does this image depict? State the setting, then justify your conclusion based on background details in 1 to 2 sentences.",
    "Does this image contain any nature-related elements or concepts? Answer 'Yes' or 'No', and justify your decision in 1 to 2 sentences."
]


def parse_args():
    """Parse command-line flags: which fixed sample/comparison files to use,
    which model to run, which prompts to ask, and how many images to limit to
    for cheap smoke-testing."""
    parser = argparse.ArgumentParser(
        description="Qualitative VLM evaluation on the fixed BIG-5 diagnostic sample")
    parser.add_argument("--diagnostic_sample_file", type=str, required=True,
                        help="Path to the fixed diagnostic sample JSON created by evaluate_big5.py "
                             "(--diagnostic_sample_file there). Must already exist -- run one of the "
                             "closed-set scripts at least once first.")
    parser.add_argument("--images_cache_dir", type=str, required=True,
                        help="Directory where the diagnostic sample's images are cached (same "
                             "--images_cache_dir used by evaluate_big5.py).")
    parser.add_argument("--comparison_file", type=str, required=True,
                        help="Path to the SAME persistent cross-model comparison JSON used by "
                             "evaluate_big5.py. This model's responses are added/overwritten under "
                             "--model_id; every other model's entries are preserved untouched.")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--model_id", type=str, default=None,
                        help="Identifier for this model's entry in --comparison_file. Defaults to a "
                             "sanitized version of --model_name_or_path if not given.")
    parser.add_argument("--prompts", type=str, nargs="+", default=DEFAULT_PROMPTS,
                        help="One or more prompts to ask about each image (stored separately, keyed "
                             "by the exact prompt text). Each prompt is an independent single-turn "
                             "query -- no shared history with the other prompts on the same image.")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit to the first N images of the diagnostic sample, for smoke-testing "
                             "the inference format cheaply before committing to a full run.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.model_id is None:
        # Model names often contain "/" (e.g. "Qwen/Qwen3.5-0.8B") which isn't
        # safe to use as a dict key / filename component in some contexts —
        # replace with underscore to get a clean default identifier.
        args.model_id = args.model_name_or_path.replace("/", "_")

    return args


def load_vlm(model_name_or_path):
    """Loading recipe kept EXACTLY as the user's own already-verified-working code."""
    # AutoProcessor bundles together the image preprocessing AND the text
    # tokenizer for this specific model family, auto-detected from the model's
    # own config on the Hub.
    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        # Left-padding matters for DECODER-ONLY generation models: when
        # batching prompts of different lengths, the actual "next token to
        # generate" must be at the very end of every sequence — left-padding
        # keeps that alignment (padding goes at the START, not the end).
        processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,  # half-precision floats: faster, less GPU memory
        device_map="auto",          # let transformers decide which device(s) to place the model on
        trust_remote_code=True,     # this model ships custom code (not just standard weights)
    )
    model.eval()  # disable training-only behavior (e.g. dropout)
    return processor, model


@torch.no_grad()  # no gradient tracking needed — we're only doing inference, not training
def ask_vlm(processor, model, image_path, prompt, max_new_tokens=256):
    """
    One independent single-turn query: image + ONE question. See module
    docstring for the verified message/generation format.
    """
    # The "chat message" format modern instruction-tuned VLMs expect: a list
    # of turns, each with a role and content. Here content is a LIST of typed
    # blocks (an image block plus a text block) rather than a plain string,
    # since this is a multimodal message.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    # `apply_chat_template` converts this structured message list into the
    # actual token ids the model expects (inserting the model's own special
    # formatting tokens around each turn/role), AND loads+preprocesses the
    # referenced image, all in one call. `add_generation_prompt=True` appends
    # the special tokens that signal "now it's the assistant's turn to speak"
    # so the model knows to start generating a response rather than
    # continuing the user's message.
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    # `model.generate()` returns the FULL sequence (the original input prompt
    # tokens PLUS the newly generated ones concatenated together) — we only
    # want the NEW part. This slices off, for each sequence, however many
    # tokens the corresponding input had (`len(in_ids)`), keeping just
    # everything generated after that point.
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]  # [0] because there's only one sequence in this (batch-of-one) call
    return response.strip()


def update_comparison_file_with_vlm(comparison_file, diagnostic_sample, vlm_responses_by_filename, model_id):
    """
    Same merge semantics as evaluate_big5.py's update_comparison_file: load
    any existing file, OVERWRITE only this model_id's entry for each image
    (Python dict assignment naturally "deletes old, saves new" for the same
    key), leave every other model's entries untouched, save back. The
    payload shape differs from the closed-set models' (raw_prediction/
    pred_nature/pred_biotic/pred_material/no_taxonomy_match) since this is
    qualitative output: {"responses": {prompt: response, ...}}.
    """
    if os.path.isfile(comparison_file):
        # This file may already contain results from OTHER models (closed-set
        # or previous VLM runs) — load it so we can add to it rather than
        # clobbering everyone else's data.
        with open(comparison_file, "r") as f:
            comparison = json.load(f)
    else:
        comparison = {}

    for d in diagnostic_sample:
        filename = d["filename"]
        # `.setdefault(filename, {...})` either returns the EXISTING entry
        # for this filename (if some earlier model run already created it) or
        # creates and returns a brand-new one with these base fields — either
        # way, `entry` now safely points at this image's dict to add our
        # model's results into below.
        entry = comparison.setdefault(filename, {
            "language": d["language"], "gt_nature": d["gt_nature"],
            "gt_biotic": d["gt_biotic"], "gt_material": d["gt_material"],
            "predictions": {},
        })
        responses = vlm_responses_by_filename.get(filename)
        if responses is None:
            # This image was in the diagnostic sample but we never actually
            # got a response for it this run (e.g. the cached file was
            # missing) — record that explicitly rather than silently omitting
            # this model's entry for the image.
            entry["predictions"][model_id] = {"missing_from_this_run": True}
        else:
            # Overwrites any PREVIOUS entry for this exact model_id on this
            # image (e.g. from an earlier partial run), while every OTHER
            # model_id's entry in `entry["predictions"]` is left completely
            # untouched.
            entry["predictions"][model_id] = {"responses": responses}

    with open(comparison_file, "w") as f:
        json.dump(comparison, f, indent=2)
    return comparison


def main():
    args = parse_args()
    print(f"🚀 Starting qualitative VLM evaluation ({args.model_name_or_path})")

    if not os.path.isfile(args.diagnostic_sample_file):
        raise FileNotFoundError(
            f"--diagnostic_sample_file '{args.diagnostic_sample_file}' not found. Run one of the "
            f"closed-set scripts (e.g. evaluate_big5.py) at least once first -- it creates this "
            f"fixed sample and this script reuses it rather than selecting its own images."
        )
    with open(args.diagnostic_sample_file, "r") as f:
        diagnostic_sample = json.load(f)
    if args.max_samples is not None:
        diagnostic_sample = diagnostic_sample[:args.max_samples]
    print(f"[INFO] Using {len(diagnostic_sample)} images from {args.diagnostic_sample_file}")
    print(f"[INFO] Prompts ({len(args.prompts)}):")
    for p in args.prompts:
        print(f"    - {p}")

    print(f"[INFO] Loading {args.model_name_or_path} (this can take a while on first download)...")
    processor, model = load_vlm(args.model_name_or_path)

    vlm_responses_by_filename = {}
    n_errors = 0
    for i, d in enumerate(diagnostic_sample):
        filename = d["filename"]
        local_path = os.path.join(args.images_cache_dir, filename)
        if not os.path.isfile(local_path):
            # This image is listed in the diagnostic sample but hasn't been
            # downloaded to the local cache yet — skip it rather than crashing
            # the whole run over one missing file.
            print(f"⚠️  [{i+1}/{len(diagnostic_sample)}] {local_path} not found in cache -- skipping "
                  f"(was it downloaded by evaluate_big5.py yet?)")
            continue

        print(f"[{i+1}/{len(diagnostic_sample)}] {filename}")
        responses = {}
        for prompt in args.prompts:
            try:
                response = ask_vlm(processor, model, local_path, prompt, max_new_tokens=args.max_new_tokens)
            except Exception as e:
                # A single prompt failing (e.g. an OOM on one particularly
                # large image) shouldn't abort the entire run — record the
                # error as the "response" text itself so it's visible in the
                # final output, and keep going with the remaining
                # prompts/images.
                response = f"[ERROR during generation: {e}]"
                n_errors += 1
                print(f"⚠️  Generation error for {filename} / '{prompt}': {e}")
            responses[prompt] = response
            if args.verbose:
                print(f"    Q: {prompt}")
                print(f"    A: {response}")
        vlm_responses_by_filename[filename] = responses

    if n_errors:
        print(f"\n⚠️  {n_errors} generation error(s) occurred -- check the responses above for "
              f"'[ERROR during generation: ...]' entries before trusting the full set.")

    comparison = update_comparison_file_with_vlm(
        args.comparison_file, diagnostic_sample, vlm_responses_by_filename, args.model_id)
    # Count how many DISTINCT model_ids appear across every image's
    # predictions dict, as a quick sanity-check number to print (e.g. "3
    # models so far" confirms this run's results were correctly merged
    # alongside previous models' results rather than replacing them).
    n_models = len(set(m for e in comparison.values() for m in e["predictions"]))
    print(f"\n💾 Updated cross-model comparison file: {args.comparison_file} "
          f"(model_id='{args.model_id}', {len(comparison)} images tracked, {n_models} models so far)")


if __name__ == "__main__":
    main()
