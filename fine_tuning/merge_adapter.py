"""
Merge a trained LoRA/DoRA adapter into the base weights, producing a
standalone checkpoint.

  python fine_tuning/merge_adapter.py \
      --base google/gemma-4-12B-it \
      --adapter runs/lora_gemma12b_balanced/adapter \
      --out /gpfs/projects/.../gemma-4-12B-it-rft-merged

NOT NEEDED FOR THE DEFAULT (PLAIN LoRA) WORKFLOW
=================================================
`run_vlm_pipeline.py --lora_adapter_path <adapter dir>` serves a plain-LoRA
adapter DIRECTLY, on the same resident vLLM engine as the base model,
selectively per call (base weights for captioning, adapted for extraction/
labeling — see src.vlm_pipeline.run_inference's `use_lora`). That is the
intended path for train_lora.py's default output and this script is not part
of it — evaluate the adapter directory straight from job_finetune.sh.

WHEN THIS SCRIPT IS ACTUALLY NEEDED
====================================
  - `--use_dora` was passed to train_lora.py. vLLM's native LoRA serving does
    not support DoRA weights (vllm-project/vllm#10849, open as of this
    writing) — a DoRA adapter must be merged and served as an ordinary
    checkpoint (every call adapted, including captioning; see train_lora.py's
    module docstring for the tradeoff this reintroduces).
  - You want a single self-contained checkpoint for something OUTSIDE this
    pipeline (handing it to a collaborator, a different serving stack) rather
    than an adapter + base-model pair.

Either way, a merged checkpoint is served by the EXISTING pipeline exactly
like any baseline model — `--model_name <merged path>`, no `--lora_adapter_path`
— so the comparison stays on the same code path.

The processor/tokenizer is copied from the base model so the output directory
is a complete, self-contained checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="Base model id the adapter was trained on")
    ap.add_argument("--adapter", required=True, help="Adapter directory from train_lora.py")
    ap.add_argument("--out", required=True, help="Output directory for the merged checkpoint")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoProcessor

    from train_lora import _load_model

    dtype = getattr(torch, args.dtype)
    print(f"Loading base {args.base} …")
    # device_map is intentionally left to the loader: merging is a weight
    # operation, so it works on CPU too, just slowly.
    model = _load_model(args.base, dtype)
    print(f"Applying adapter {args.adapter} …")
    model = PeftModel.from_pretrained(model, args.adapter)
    print("Merging …")
    model = model.merge_and_unload()

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    AutoProcessor.from_pretrained(args.base).save_pretrained(args.out)

    with open(os.path.join(args.out, "rft_provenance.json"), "w", encoding="utf-8") as fh:
        json.dump({"base_model": args.base, "adapter": os.path.abspath(args.adapter),
                   "dtype": args.dtype}, fh, indent=1)
    print(f"Merged checkpoint written to {args.out}")
    print("Evaluate it with the normal pipeline, e.g.:")
    print(f"  python scripts/run_vlm_pipeline.py --stage all --model_family gemma "
          f"--model_name {args.out} --dataset big5_twitter "
          f"--split_file /home/pmonserrat/datasets/big_5/rft/splits/test_images.txt ...")


if __name__ == "__main__":
    main()
