"""
LoRA/DoRA fine-tune of the VLM pipeline's LANGUAGE decoder on a
rejection-sampled BIG-5 training set.

  python fine_tuning/train_lora.py \
      --model google/gemma-4-12B-it \
      --dataset_dir /home/pmonserrat/datasets/big_5/rft/rft_gemma12b \
      --output_dir runs/lora_gemma12b_balanced

PLAIN LoRA IS THE DEFAULT (--use_dora is opt-in), and that is a serving
decision, not a training one. DoRA usually edges out plain LoRA on quality at
the same rank, but vLLM's native LoRA adapter serving does not support DoRA
weights as of this writing — there's an open, unmerged feature request for it
(vllm-project/vllm#10849 / the closed PR #14389) — while it fully supports
plain LoRA, with sub-millisecond per-request hot-swap. That matters here
specifically because THIS project needs to apply the adapter to only SOME
calls in a run (extraction + labeling) while the caption call always runs the
untouched base weights, on the SAME resident engine, in a SINGLE pass over
the dataset (see src.models.vlm_models.VLLMBackedVLM's `use_lora` /
src.vlm_pipeline.run_inference's `use_lora`). Passing a DoRA-trained adapter
to that serving path would not error — it would silently run plain-LoRA math
against DoRA-trained weights, which is wrong, not merely unsupported. Set
--use_dora only if you accept going back to the two-pass-or-merge serving
story documented in merge_adapter.py's docstring.

WHAT IS AND IS NOT TRAINED
==========================
The vision tower and the multimodal projector are FROZEN and get no adapter.
Only the language decoder's linear layers receive LoRA/DoRA weights. That is a
deliberate scope choice, and it is also what makes the visual-embedding cache
valid (see vision_cache.py) — an adapter anywhere in the vision path would
make an image's soft tokens change between epochs, and every cached vector
would silently go stale.

The script ASSERTS this rather than assuming it: after wrapping the model it
walks every trainable parameter and aborts if any of them lives outside the
language decoder.

LABEL MASKING
=============
Loss is computed on the assistant's response tokens only. The prompt is
rendered twice per example — once with `add_generation_prompt=True` (prompt
only) and once with the assistant turn appended — and the first N tokens of
the full sequence are masked out, where N is the prompt's own token count. The
two tokenizations are then checked to agree on that prefix, so a chat template
that does not tokenize prefix-stably fails loudly instead of quietly training
on a misaligned target.

STRUCTURED TARGETS ARE TRAINED AS RAW TEXT
==========================================
Extraction and labeling responses are JSON. They are trained as plain strings —
there is no constrained decoding during training, so the model must learn the
shape as well as the content. At inference time vLLM's structured output still
enforces the schema, so a training-time formatting slip cannot produce an
unparseable prediction; what the model actually learns from these examples is
the CONTENT within a shape it is already being held to.

LOSS AGGREGATION
================
By default the model's own loss is used unchanged (token-level mean over the
batch), so a run is comparable to any other Trainer run. When the dataset
carries per-example `weight` values (--balance loss_weight in
build_rft_dataset.py), the loss switches to a per-example mean followed by a
weighted average, which is the only aggregation where a weight means what it
says. That difference is reported at startup so the two modes are never
confused for one another.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(__file__))

from src.models.prompts import build_system_prompts  # noqa: E402
from rft_common import iter_jsonl  # noqa: E402
from vision_cache import (VisionEmbeddingCache, capture_reference_image_embeddings,  # noqa: E402
                          inject_image_features, load_training_image,
                          resolve_image_token_id, verify_equivalence)


# =============================================================================
# Dataset
# =============================================================================
class RFTDataset(Dataset):
    """The JSONL written by build_rft_dataset.py, with system prompts resolved.

    `system_key` is stored per example instead of the prompt text (the
    definition files are several KB and would otherwise be repeated ~20k
    times); the three strings are read once here and shared by reference, so
    every example with the same key points at the SAME Python string — which
    also keeps the tokenized prefix bit-identical across examples.
    """

    def __init__(self, path: str, system_prompts: Dict[str, str]):
        self.rows: List[Dict[str, Any]] = list(iter_jsonl(path))
        self.system_prompts = system_prompts
        unknown = {r["system_key"] for r in self.rows} - set(system_prompts)
        if unknown:
            raise ValueError(f"{path}: unknown system_key(s) {sorted(unknown)}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        return {
            "system": self.system_prompts[row["system_key"]],
            "prompt": row["prompt"],
            "target": row["target"],
            "image_path": row["image_path"],
            "weight": float(row.get("weight", 1.0)),
        }


# =============================================================================
# Collator
# =============================================================================
@dataclass
class RFTCollator:
    """Builds one padded batch, masking the prompt out of the labels.

    The image is preprocessed for EVERY example, cache hit or miss — the
    processor is what inserts the right number of image soft tokens into
    `input_ids`, and short-circuiting it would mean predicting that count
    ourselves, which is exactly the kind of assumption that breaks silently on
    a model with dynamic image resolution. What the cache saves is the GPU
    vision forward (and the pixel transfer), not the CPU preprocessing; the
    latter runs in dataloader workers and overlaps with the GPU anyway.
    """

    processor: Any
    max_seq_len: int
    max_image_side: Optional[int]
    cache: Optional[VisionEmbeddingCache]

    def _render(self, system: str, prompt: str, target: Optional[str]) -> str:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
        ]
        if target is None:
            return self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        messages.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Builds one batch via a SINGLE batched processor call across every
        item, not one call per item followed by hand-rolled reassembly.

        THIS IS THE ACTUAL FIX for a real crash at --per_device_train_batch_size
        > 1: this processor does not pad every image to one fixed patch grid
        (different images legitimately produce differently-shaped
        pixel_values/aux tensors), so calling it once per image and later
        `torch.cat`-ing the results assumed a uniform shape that is simply
        false here — confirmed by "Sizes of tensors must match except in
        dimension 0" the moment two differently-shaped images landed in one
        batch. Calling the processor ONCE, with the whole batch's texts and
        images together (`text=[...], images=[...]`), is the documented,
        standard way HuggingFace processors are meant to be used for
        multi-image batches — its OWN padding/batching logic resolves the
        shape mismatch correctly, the same contract vLLM's own inference-time
        batching already relies on, since it's the same processor class.
        Nothing here reimplements that logic; it just finally calls it the
        way it's built to be called.
        """
        tokenizer = getattr(self.processor, "tokenizer", self.processor)

        full_texts: List[str] = []
        char_starts: List[int] = []
        images = []
        paths: List[str] = []
        weights: List[float] = []
        cached_list: List[Optional[torch.Tensor]] = []
        for item in features:
            image = load_training_image(item["image_path"], self.max_image_side)
            full_text = self._render(item["system"], item["prompt"], item["target"])
            # Locate the assistant's own response by searching for its OWN
            # literal content within full_text — independent of whatever
            # whitespace the template puts around turn boundaries (an
            # earlier, per-item design compared TWO separately-rendered
            # strings against each other and that was measurably wrong on a
            # real run: a template can legitimately render a turn's trailing
            # whitespace conditionally on whether more turns follow). rfind
            # (not find): if the target text happens to also appear earlier
            # (e.g. quoted inside the prompt), the real assistant turn is
            # still the LAST occurrence.
            char_start = full_text.rfind(item["target"])
            if char_start == -1:
                raise RuntimeError(
                    "Could not find the target text verbatim inside the rendered chat "
                    "template output — the template may be transforming/escaping message "
                    f"content, which this reconstruction does not expect. "
                    f"image={item['image_path']!r}"
                )
            full_texts.append(full_text)
            char_starts.append(char_start)
            images.append(image)
            paths.append(item["image_path"])
            weights.append(item["weight"])
            cached_list.append(self.cache.get(item["image_path"]) if self.cache else None)

        # ONE call for the whole micro-batch. `padding_side` is forced to
        # "right" for the duration of this call (then restored) rather than
        # trusted to whatever the tokenizer's own default happens to be:
        # the per-item stripping below (via attention_mask) assumes the REAL
        # tokens are the FIRST `real_len` positions, which only holds under
        # right-padding — a standard, temporary mutate-then-restore pattern,
        # not a novel trick.
        original_padding_side = getattr(tokenizer, "padding_side", None)
        if original_padding_side is not None:
            tokenizer.padding_side = "right"
        try:
            # NOT `images=images` (a flat list). This processor's own
            # make_nested_list_of_images treats a FLAT list of images as ONE
            # example containing all of them ("a list of images is a single
            # batch") and only a LIST-OF-LISTS as "one sublist per example" —
            # confirmed by reading transformers/image_utils.py directly after
            # a real crash: "Received inconsistently sized batches of images
            # (1) and text (4)" at --per_device_train_batch_size 4, because
            # the flat 4-image list got collapsed into a single nested sample.
            # Every example here has exactly one image, so each gets its own
            # singleton sublist.
            batched = self.processor(text=full_texts, images=[[img] for img in images], padding=True,
                                     return_tensors="pt", return_offsets_mapping=True)
        finally:
            if original_padding_side is not None:
                tokenizer.padding_side = original_padding_side

        offsets = batched.get("offset_mapping")
        if offsets is None:
            # Deliberately NOT falling back to a per-item reconstruction here
            # (an earlier design had one) — reproducing that correctly for an
            # already-batched, already-padded tensor would need re-deriving
            # per-item boundaries a different way, and this project has
            # already spent a full session's worth of unverified guesses
            # about this processor's behavior. Fail loudly and specifically
            # instead of adding another one.
            raise RuntimeError(
                "This processor did not return offset_mapping from a batched call, so "
                "there is no reliable way to place the per-item label mask. Set "
                "--per_device_train_batch_size 1 (a batch of 1 needs no cross-item "
                "reconciliation at all), or extend this collator with a genuine "
                "per-item fallback rather than guessing at one."
            )

        input_ids_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []
        # Per-item "everything the processor returned besides input_ids/
        # attention_mask/offset_mapping" (pixel_values, and — on a
        # tiling-aware processor — whatever else it adds, e.g. image_sizes/
        # num_crops/image_grid_thw) — a SLICE of the batched tensors, not an
        # independent per-image call, so it is by construction the same
        # shape as every other item's slice: the shape-mismatch class of bug
        # this whole rewrite exists to remove cannot recur here.
        image_kwargs_list: List[Optional[Dict[str, torch.Tensor]]] = []

        for i, item_path in enumerate(paths):
            real_len = int(batched["attention_mask"][i].sum().item())
            ids = batched["input_ids"][i][:real_len]  # right-padding forced above
            row = offsets[i][:real_len]
            if isinstance(row, torch.Tensor):
                n_prompt = int((row[:, 1] <= char_starts[i]).sum().item())
            else:
                n_prompt = sum(1 for (_, end) in row if end <= char_starts[i])
            labels = ids.clone()
            labels[:n_prompt] = -100

            if ids.shape[0] > self.max_seq_len:
                # Truncating from the LEFT would cut the system prompt and the
                # image tokens; truncating from the right would cut the answer
                # being trained on. Neither is acceptable, so this is an error
                # — raise --max_seq_len instead.
                raise RuntimeError(
                    f"Example is {ids.shape[0]} tokens, over --max_seq_len {self.max_seq_len} "
                    f"({item_path}). Raise --max_seq_len; truncating would cut "
                    f"either the definitions or the target."
                )

            input_ids_list.append(ids)
            labels_list.append(labels)

            # Keep the image kwargs only when they will actually be needed (no
            # cache, or a miss that must be computed and stored) — a cache hit
            # needs none of it. Non-tensor keys (rare metadata some
            # processors emit) are dropped rather than guessed at: a tensor
            # key concatenates safely below; a non-tensor one has no generic
            # batching rule, and if the model's forward() actually required
            # it, that surfaces as an immediate, loud error from the model
            # itself rather than a silently wrong result.
            if cached_list[i] is None:
                image_kwargs_list.append({
                    k: v[i:i + 1] for k, v in batched.items()
                    if k not in ("input_ids", "attention_mask", "offset_mapping")
                    and isinstance(v, torch.Tensor)
                })
            else:
                image_kwargs_list.append(None)

        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        max_len = max(x.shape[0] for x in input_ids_list)
        input_ids = torch.full((len(input_ids_list), max_len), pad_id, dtype=torch.long)
        labels = torch.full((len(labels_list), max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(input_ids_list), max_len), dtype=torch.long)
        for i, (ids, lab) in enumerate(zip(input_ids_list, labels_list)):
            input_ids[i, : ids.shape[0]] = ids
            labels[i, : lab.shape[0]] = lab
            attention_mask[i, : ids.shape[0]] = 1

        batch: Dict[str, Any] = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "image_paths": paths,
            "example_weights": torch.tensor(weights, dtype=torch.float32),
            "cached_features": cached_list,
        }
        owner = [i for i, kw in enumerate(image_kwargs_list) if kw is not None]
        if owner:
            # Concatenated back together, for the NORMAL (uncached) forward
            # path. This is now PROVABLY shape-safe, not just assumed to be:
            # every image_kwargs_list[i] is a slice of the SAME single
            # batched tensor `batched` produced above, so re-concatenating a
            # subset of its own rows can never hit a shape mismatch — there
            # is nothing left for two DIFFERENT images to disagree about,
            # since they were never processed as two independent calls in
            # the first place.
            keys = image_kwargs_list[owner[0]].keys()
            batch["image_kwargs"] = {
                key: torch.cat([image_kwargs_list[i][key] for i in owner], dim=0)
                for key in keys
            }
            batch["pixel_owner"] = owner
            # Per-owned-item kwargs, kept SEPARATE from the concatenated
            # version above — see RFTTrainer._features_for_batch, which
            # processes cache misses one image at a time for exactly the
            # reason explained on image_kwargs_list above.
            batch["pixel_owner_kwargs"] = [image_kwargs_list[i] for i in owner]
        return batch


# =============================================================================
# GPU memory logging — so batch-size tuning is based on measured headroom,
# not another guess submitted blind to the cluster.
# =============================================================================
def _build_memory_logger_callback():
    """Real peak GPU memory usage, printed every --logging_steps (piggybacks
    on the Trainer's own logging cadence via on_log, so it costs no extra
    synchronization beyond what's already happening there). Peak
    allocated/reserved since the LAST reset — reset after every print, so
    each number reflects the recent window of steps, not a running-forever
    high-water mark that would just report whatever the single worst step
    in the whole run happened to be.

    This is what actually answers "how much more --per_device_train_batch_size
    room is there" — with real numbers from THIS run, instead of reasoning
    about it from a FLOPs estimate or an inference-time measurement that
    doesn't include training's own extra memory (gradients, optimizer
    state, activations).
    """
    from transformers import TrainerCallback

    class GpuMemoryLoggerCallback(TrainerCallback):
        def on_log(self, args, state, control, **kwargs):
            if not torch.cuda.is_available():
                return
            # Loop every VISIBLE device, not just device 0. With
            # device_map="auto" spanning >1 GPU (naive model parallelism —
            # see the --gres=gpu:N note in job_finetune.sh), the base model's
            # layers are split across cards, so a single card's peak memory
            # says nothing about whether the OTHER card is the one closest to
            # OOM; per-device peaks are what actually tells you which card to
            # size the next batch-size step against.
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.max_memory_allocated(i) / 1e9
                reserved = torch.cuda.max_memory_reserved(i) / 1e9
                total = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"[gpu-mem] step {state.global_step} device {i}: peak allocated {allocated:.1f} GB / "
                      f"peak reserved {reserved:.1f} GB / device total {total:.1f} GB")
                torch.cuda.reset_peak_memory_stats(i)

    return GpuMemoryLoggerCallback()


# =============================================================================
# Trainer
# =============================================================================
def build_trainer_class():
    """Defined inside a function so this module imports cleanly (for tests and
    for `--help`) on a machine without transformers installed."""
    from transformers import Trainer

    class RFTTrainer(Trainer):
        vision_cache: Optional[VisionEmbeddingCache] = None
        image_token_id: Optional[int] = None
        # The language-decoder submodule (train_lora.language_model_of's
        # result, captured in main() BEFORE the peft wrap — still a valid,
        # live reference afterward, since get_peft_model() injects adapters
        # into the EXISTING module tree in place rather than rebuilding it).
        # Needed by capture_reference_image_embeddings to know where to
        # install its forward-pre-hook.
        decoder: Optional[torch.nn.Module] = None
        use_example_weights: bool = False

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            paths = inputs.pop("image_paths", [])
            weights = inputs.pop("example_weights", None)
            cached = inputs.pop("cached_features", None)
            owner = inputs.pop("pixel_owner", None)
            owner_kwargs = inputs.pop("pixel_owner_kwargs", None)
            image_kwargs = inputs.pop("image_kwargs", None)

            if self.vision_cache is not None and self.vision_cache.enabled:
                base = model.get_base_model() if hasattr(model, "get_base_model") else model
                # input_ids/attention_mask are read here BEFORE input_ids is
                # popped below — capture_reference_image_embeddings needs the
                # real per-example rows to run its own genuine forward pass.
                features = self._features_for_batch(
                    base, paths, cached, owner, owner_kwargs,
                    inputs["input_ids"], inputs["attention_mask"])
                inputs["inputs_embeds"] = inject_image_features(
                    base, inputs.pop("input_ids"), features, self.image_token_id)
            elif image_kwargs is not None:
                # Normal (uncached) path: hand the model EVERYTHING the
                # processor returned for these images (pixel_values plus any
                # other aux keys), exactly as if the processor had been
                # called with every image in this batch at once. The model's
                # own forward() does the row-attribution work internally —
                # nothing here has to know how.
                inputs.update(image_kwargs)

            labels = inputs.get("labels")
            outputs = model(**inputs)

            if not self.use_example_weights or weights is None:
                loss = outputs.loss
                return (loss, outputs) if return_outputs else loss

            # Per-example mean, then weighted average across the batch — the
            # only aggregation under which a per-example weight is meaningful.
            logits = outputs.logits[..., :-1, :]
            gold = labels[..., 1:]
            per_token = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(), gold.reshape(-1),
                ignore_index=-100, reduction="none").view(gold.shape)
            valid = (gold != -100)
            per_example = (per_token * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
            w = weights.to(per_example.device, per_example.dtype)
            loss = (per_example * w).sum() / w.sum().clamp(min=1e-8)
            return (loss, outputs) if return_outputs else loss

        def _features_for_batch(self, base, paths, cached, owner, owner_kwargs,
                                input_ids_batch, attention_mask_batch):
            """Cached features for every example, computing and storing misses.

            Cache-miss images are computed ONE AT A TIME via
            capture_reference_image_embeddings — a real, genuine forward pass
            per image (aborted right after the decoder receives its
            inputs_embeds, so the actual decoder compute never runs) rather
            than a batched, independently-reimplemented vision-only call.
            Besides sidestepping the tiling/pan-and-scan row-attribution
            ambiguity a batched multi-image call would have (a tiled
            processor can give different images different soft-token
            counts), this is what makes the cached features EXACTLY equal to
            what a normal forward would have produced — there is no
            reimplementation left to disagree with reality (see
            vision_cache.py's module docstring for the real, measured bug
            this replaced). `base` is the receiver here (not the possibly
            peft-wrapped `model`) because it is what `self.decoder` was
            resolved against.
            """
            device = next(base.parameters()).device
            slots: List[Optional[torch.Tensor]] = list(cached or [None] * len(paths))
            if owner:
                for batch_idx, kwargs in zip(owner, owner_kwargs):
                    item_kwargs = {
                        k: (v.to(device=device, dtype=base.dtype) if torch.is_floating_point(v)
                            else v.to(device=device))
                        for k, v in kwargs.items()
                    }
                    row_ids = input_ids_batch[batch_idx:batch_idx + 1].to(device)
                    row_mask = attention_mask_batch[batch_idx:batch_idx + 1].to(device)
                    feat = capture_reference_image_embeddings(
                        base, self.decoder, row_ids, row_mask, self.image_token_id, **item_kwargs)
                    slots[batch_idx] = feat
                    self.vision_cache.put(paths[batch_idx], feat)
            missing = [p for p, s in zip(paths, slots) if s is None]
            if missing:
                raise RuntimeError(f"No image features for {len(missing)} example(s): {missing[:3]}")
            # torch.cat over each slot FLATTENED to (rows, hidden), NOT
            # torch.stack: different images can legitimately produce
            # different soft-token row counts (tiling/pan-and-scan), and
            # inject_image_features already flattens whatever it receives
            # down to (total_rows, hidden) internally regardless — so there
            # is no correctness requirement for a uniform per-example shape
            # here, and torch.stack would wrongly demand one.
            hidden = slots[0].shape[-1]
            flat = [s.to(device=device).reshape(-1, hidden) for s in slots]
            return torch.cat(flat, dim=0)

    return RFTTrainer


# =============================================================================
# Model setup
# =============================================================================
def language_model_of(model):
    """Locate the language decoder submodule, WHEREVER it lives in the tree.

    A multimodal wrapper names its text decoder "language_model", but at a
    DEPTH that varies by architecture — a single-vision-tower VLM
    (Qwen-VL/LLaVA-style) puts it at `model.language_model` (depth 1); an
    OMNI-modal wrapper juggling several encoders nests it one level deeper.
    Confirmed directly on google/gemma-4-12B-it via
    fine_tuning/inspect_model_layers.ipynb: `model.model` is
    `Gemma4UnifiedModel`, whose own children are `language_model`
    (`Gemma4UnifiedTextModel`, the actual decoder), `embed_vision`, AND
    `embed_audio` — three siblings, not one. An earlier version of this
    function checked only `model.language_model` then fell back to
    `model.model` as a fixed top-level guess; on this exact architecture
    that resolved to the WRAPPER (`model.model`), which would have put
    LoRA/DoRA adapters on Linear layers inside `embed_vision`/`embed_audio`
    too — a real bug, caught only by actually inspecting the printed tree
    rather than trusting the guess.

    So: search the WHOLE module tree for a submodule literally named
    "language_model" and take the SHALLOWEST match (closest to the root —
    correct regardless of how deep a given architecture nests it). Returns
    its FULL dotted path as `lm_prefix`, not just a bare attribute name —
    find_target_modules/assert_only_language_trainable match against this
    exact string via `.startswith(lm_prefix + ".")`, so it has to be the
    real path, nested or not. Falls back to `model.model` (a plain causal
    LM with no multimodal wrapper at all) only when no "language_model"
    submodule exists anywhere.
    """
    candidates = [(name, sub) for name, sub in model.named_modules()
                 if name.split(".")[-1] == "language_model" and isinstance(sub, torch.nn.Module)]
    if candidates:
        name, sub = min(candidates, key=lambda item: item[0].count("."))
        return sub, name
    sub = getattr(model, "model", None)
    if isinstance(sub, torch.nn.Module):
        return sub, "model"
    raise RuntimeError(
        f"Could not locate the language decoder inside {type(model).__name__} "
        f"(no submodule named 'language_model' anywhere, and no top-level 'model' "
        f"attribute either). Pass --target_modules explicitly.")


def _linear_like_types() -> tuple:
    """torch.nn.Linear, plus bitsandbytes' quantized Linear replacements when
    available — under --qlora, every decoder Linear is a bnb.nn.Linear4bit
    (or, on an 8-bit config, Linear8bitLt), not a plain torch.nn.Linear, so a
    bare `isinstance(module, torch.nn.Linear)` check would silently find ZERO
    target modules under 4-bit loading rather than failing loudly (this
    module's own `if not names: raise` would catch it, but only after the
    fact) — checked explicitly here instead of relying on bitsandbytes'
    inheritance chain (Linear4bit does subclass nn.Linear in current
    versions, which would make the bare check happen to work, but that's an
    implementation detail of a fast-moving dependency, not a contract worth
    depending on silently).
    """
    types = [torch.nn.Linear]
    try:
        import bitsandbytes as bnb
        types.append(bnb.nn.Linear4bit)
        types.append(bnb.nn.Linear8bitLt)
    except ImportError:
        pass
    return tuple(types)


def find_target_modules(model, lm_prefix: str) -> List[str]:
    """Every Linear inside the language decoder, by FULL module name.

    Full names, not the usual suffix list ("q_proj", …): peft matches a suffix
    against every module in the model, so a suffix shared with the vision
    tower would put adapters there too — which would both widen the fine-tune
    beyond the intended scope and invalidate the embedding cache.
    """
    linear_types = _linear_like_types()
    names = []
    for name, module in model.named_modules():
        if not name.startswith(lm_prefix + "."):
            continue
        if isinstance(module, linear_types) and "lm_head" not in name:
            names.append(name)
    if not names:
        raise RuntimeError(f"No Linear layers found under {lm_prefix!r}.")
    return names


def assert_only_language_trainable(model, lm_prefix: str) -> Dict[str, Any]:
    """Abort if any trainable parameter lives outside the language decoder."""
    trainable, offenders, total = 0, [], 0
    for name, param in model.named_parameters():
        total += param.numel()
        if not param.requires_grad:
            continue
        trainable += param.numel()
        # peft prefixes wrapped modules with "base_model.model."; strip any
        # leading wrapper segments before checking where the parameter lives.
        stripped = name
        for prefix in ("base_model.model.", "base_model."):
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix):]
        if not stripped.startswith(lm_prefix + "."):
            offenders.append(name)
    if offenders:
        raise RuntimeError(
            f"{len(offenders)} trainable parameter(s) are OUTSIDE the language decoder "
            f"({lm_prefix!r}), e.g. {offenders[:5]}. The vision path must stay frozen — "
            f"otherwise the cached image embeddings go stale after the first optimizer "
            f"step and every later epoch trains against vectors the model no longer "
            f"produces."
        )
    return {"trainable_params": trainable, "total_params": total,
            "trainable_pct": round(100.0 * trainable / max(total, 1), 4)}


# =============================================================================
# Entry point
# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/gemma-4-12B-it")
    ap.add_argument("--dataset_dir", required=True, help="Directory from build_rft_dataset.py")
    ap.add_argument("--output_dir", required=True)

    # --- LoRA/DoRA (peft defaults; every one exposed so they can be swept) ---
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--use_dora", action="store_true", default=False,
                    help="Train DoRA instead of plain LoRA. OFF by default — see the module "
                         "docstring for why: vLLM cannot serve a DoRA adapter selectively "
                         "(vllm-project/vllm#10849), which is required for this project's "
                         "base-weights-for-captioning / adapter-for-extraction+labeling "
                         "serving pattern. Only pass this if you specifically intend to "
                         "merge (merge_adapter.py) and serve every call adapted, or to run a "
                         "training-only quality comparison.")
    ap.add_argument("--target_modules", nargs="+", default=None,
                    help="Default: every Linear in the language decoder, by full name.")
    ap.add_argument("--qlora", action="store_true", default=False,
                    help="Load the frozen base weights in 4-bit (bitsandbytes NF4 + double "
                         "quantization) instead of full precision -- the standard QLoRA "
                         "recipe. Only shrinks the BASE weights' memory footprint (~24GB "
                         "bf16 -> ~6GB); the LoRA adapter itself is trained and served "
                         "exactly as without this flag (plain LoRA, --lora_adapter_path, "
                         "never quantized). Use this to free per-GPU headroom for a bigger "
                         "--per_device_train_batch_size when a second GPU isn't available. "
                         "Requires the `bitsandbytes` package. NOT YET VERIFIED end-to-end "
                         "on this project's real cluster hardware -- treat the first run "
                         "with this flag as a genuine test, not a confirmed-working default "
                         "the way the rest of this script's settings are.")

    # --- Optimization ---
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--num_train_epochs", type=float, default=2.0)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=None,
                    help="Default: mirrors --per_device_train_batch_size. NOT the same "
                         "memory cost per example as training, even at an equal batch size -- "
                         "eval runs under torch.no_grad() (Trainer.prediction_step), so it "
                         "carries none of training's backward-pass activation retention, "
                         "gradient-checkpointing recompute buffers, gradients, or optimizer "
                         "state. It can very plausibly run at a HIGHER batch size than "
                         "training on the same card; mirroring train's batch size here is a "
                         "safe default chosen to stop a real crash (HF's own "
                         "per_device_eval_batch_size otherwise defaults to a hardcoded 8, "
                         "independent of whatever --per_device_train_batch_size was actually "
                         "tuned to fit -- confirmed OOM: training stable for 200 steps at "
                         "batch_size=1, then OOM inside evaluate() at the first --eval_steps "
                         "checkpoint), not a claim that this IS the real eval ceiling. Raise "
                         "it explicitly once you have real headroom numbers to tune against.")
    ap.add_argument("--gradient_accumulation_steps", type=int, default=16)
    ap.add_argument("--auto_find_batch_size", action="store_true", default=False,
                    help="Standard HF/accelerate mechanism: if a training step OOMs, "
                         "automatically HALVE --per_device_train_batch_size and retry, "
                         "instead of crashing — the framework-native way to find the real "
                         "memory ceiling on unfamiliar hardware/model combinations, rather "
                         "than guessing a fixed value one cluster submission at a time. "
                         "Off by default so a plain rerun stays exactly reproducible; the "
                         "final batch size actually used is logged in train_config.json "
                         "either way. NOTE: does NOT auto-adjust "
                         "--gradient_accumulation_steps to compensate, so the EFFECTIVE "
                         "batch size can end up smaller than the 1x16/2x8 default of 16 if "
                         "it backs off — acceptable for LoRA (few trainable params, "
                         "insensitive to a modest effective-batch change), but worth knowing.")
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--lr_scheduler_type", default="cosine")
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--max_seq_len", type=int, default=8192)
    ap.add_argument("--gradient_checkpointing", action="store_true", default=True)
    ap.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--dataloader_num_workers", type=int, default=4)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--eval_on_start", action="store_true", default=False,
                    help="Run one full evaluation pass at step 0, before any training step, "
                         "as a fail-fast sanity check -- standard HF Trainer flag, not a "
                         "custom addition. Directly relevant here: --eval_steps checkpoints "
                         "run evaluate() BEFORE save_checkpoint() in Trainer's own internal "
                         "order (see train_lora.py's checkpoint-resume comment further down), "
                         "so an eval-time OOM at, say, step 246 would crash before writing a "
                         "checkpoint there, losing the training already done. --eval_on_start "
                         "surfaces the same failure at step 0 instead, before any real GPU-hour "
                         "is spent on this run, rather than discovering it hours in.")
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--save_total_limit", type=int, default=2)

    # --- Image / cache ---
    ap.add_argument("--max_image_side", type=int, default=1024,
                    help="Must match the inference-time value, or training and serving "
                         "see different pixels. 0 disables resizing.")
    ap.add_argument("--vision_cache_dir", default=None,
                    help="Where cached image embeddings live. Default: <output_dir>/vision_cache")
    ap.add_argument("--no_vision_cache", dest="vision_cache", action="store_false", default=True,
                    help="Disable the vision-embedding cache and its --verify_cache check "
                         "entirely, training through the standard model(pixel_values=..., ...) "
                         "path instead. Currently what job_finetune.sh actually passes — see "
                         "vision_cache.py's module docstring for why (a real, unresolved "
                         "embedding-scaling mismatch on google/gemma-4-12B-it; the cache is a "
                         "speed optimization, not a correctness requirement, so this is the "
                         "pragmatic default until/unless that's revisited).")
    ap.add_argument("--verify_cache", type=int, default=2,
                    help="Batches to check the cached-embedding forward against the normal "
                         "multimodal forward before training. 0 skips (not recommended).")
    ap.add_argument("--verify_tolerance", type=float, default=1e-3)
    ap.add_argument("--image_token_id", type=int, default=None)

    # --- Definitions / tracking ---
    ap.add_argument("--nature_definition_path", default="data/big5_taxonomy/big5_nature_definition.txt")
    ap.add_argument("--biotic_definition_path", default="data/big5_taxonomy/big5_biotic_definition.txt")
    ap.add_argument("--material_definition_path", default="data/big5_taxonomy/big5_material_definition.txt")
    ap.add_argument("--wandb_project", default=None)
    ap.add_argument("--run_name", default=None)
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, TrainingArguments

    torch_dtype = getattr(torch, args.dtype)

    # --- System prompts, shared by reference across examples ---
    caption_system, label_full, label_material = build_system_prompts(
        args.nature_definition_path, args.biotic_definition_path, args.material_definition_path)
    system_prompts = {"nature": caption_system, "full": label_full, "material": label_material}

    train_ds = RFTDataset(os.path.join(args.dataset_dir, "train.jsonl"), system_prompts)
    val_path = os.path.join(args.dataset_dir, "val.jsonl")
    val_ds = RFTDataset(val_path, system_prompts) if os.path.exists(val_path) else None
    use_weights = any(r.get("weight", 1.0) != 1.0 for r in train_ds.rows)
    print(f"train: {len(train_ds)} examples | val: {len(val_ds) if val_ds else 0}")
    print(f"loss aggregation: {'per-example mean, weighted' if use_weights else 'token mean (model default)'}")

    # --- Model ---
    processor = AutoProcessor.from_pretrained(args.model)
    model = _load_model(args.model, torch_dtype, qlora=args.qlora)
    lm, lm_prefix = language_model_of(model)

    if args.qlora:
        # Standard QLoRA call order: prepare the freshly 4-bit-loaded base
        # BEFORE wrapping it with LoRA — casts LayerNorm-like modules back to
        # fp32 for training stability (quantized weights stay 4-bit; this is
        # about the surrounding non-quantized ops) and, gated on
        # use_gradient_checkpointing exactly like this project's own
        # non-qlora branch further below, calls model.enable_input_require_grads()
        # -- required whenever checkpointing runs over an all-frozen base, or
        # no gradient reaches the adapters at all (see that branch's comment).
        # It also calls model.gradient_checkpointing_enable() itself; this is
        # harmless to redo via TrainingArguments(gradient_checkpointing=...)
        # below (idempotent, the standard peft+Trainer QLoRA pattern), not a
        # conflict between the two call sites.
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing)

    # Freeze everything up front; peft re-enables only the adapters it adds.
    # (prepare_model_for_kbit_training above already does this too when
    # --qlora is set -- kept unconditional so both paths share the identical
    # "every param starts frozen before get_peft_model runs" invariant.)
    for param in model.parameters():
        param.requires_grad = False

    targets = args.target_modules or find_target_modules(model, lm_prefix)
    adapter_kind = "DoRA" if args.use_dora else "LoRA"
    print(f"{adapter_kind} targets: {len(targets)} Linear modules under {lm_prefix!r}")
    peft_model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=targets, bias="none", task_type="CAUSAL_LM",
        use_dora=args.use_dora,
    ))
    if args.use_dora:
        print("⚠️  --use_dora: this adapter CANNOT be served selectively via vLLM "
              "(vllm-project/vllm#10849) — plan to merge_adapter.py it and serve every "
              "call adapted, including captioning, rather than passing it to "
              "run_vlm_pipeline.py's --lora_adapter_path.")
    if args.gradient_checkpointing:
        # Required whenever gradient checkpointing runs over a base whose
        # parameters are ALL frozen (which is exactly the LoRA/DoRA setup): the
        # checkpointed segment's inputs then carry requires_grad=False, torch
        # skips recomputation, and NO gradient reaches the adapters. It fails
        # as a warning and a flat loss curve, not an exception. The hook this
        # installs sits on the input-embedding module, so it also covers the
        # cached-embedding path, which calls that same module directly.
        if not args.qlora:
            # --qlora already did this above via prepare_model_for_kbit_training
            # (same underlying call, same gating on gradient_checkpointing) --
            # skipped here so it isn't invoked under two different code paths.
            model.enable_input_require_grads()
        model.config.use_cache = False

    param_stats = assert_only_language_trainable(peft_model, lm_prefix)
    print(f"trainable: {param_stats['trainable_params']:,} / {param_stats['total_params']:,} "
          f"({param_stats['trainable_pct']}%) — vision path verified frozen")

    # `is not None`, not truthy — an explicit `--image_token_id 0` (however
    # unlikely a real tokenizer makes it) must not be silently overridden by
    # the auto-resolved value, the same way 0 is a perfectly valid id anywhere
    # else this project reads one.
    image_token_id = (args.image_token_id if args.image_token_id is not None
                      else resolve_image_token_id(model, processor))

    cache = None
    if args.vision_cache:
        cache_dir = args.vision_cache_dir or os.path.join(args.output_dir, "vision_cache")
        fingerprint = VisionEmbeddingCache.make_fingerprint(
            model=args.model, dtype=args.dtype, max_image_side=args.max_image_side,
            processor=str(getattr(processor, "image_processor", "")),
        )
        cache = VisionEmbeddingCache(cache_dir, fingerprint, enabled=True)
        print(f"vision cache: {cache.root}")

    collator = RFTCollator(processor=processor, max_seq_len=args.max_seq_len,
                           max_image_side=args.max_image_side or None, cache=cache)

    # --- The check that makes the cache trustworthy (see vision_cache.py) ---
    if cache is not None and args.verify_cache > 0:
        probe = RFTCollator(processor=processor, max_seq_len=args.max_seq_len,
                            max_image_side=args.max_image_side or None, cache=None)
        n_probe = min(args.verify_cache, len(train_ds))
        if n_probe < args.verify_cache:
            print(f"--verify_cache {args.verify_cache} exceeds the train set size "
                  f"({len(train_ds)}) — probing all {n_probe} example(s) instead.")
        batches = [_batch_to_device(probe([train_ds[i]]), model.device) for i in range(n_probe)]
        _diagnose_image_token_id(batches[0], image_token_id, getattr(processor, "tokenizer", processor),
                                 model, lm)
        report = verify_equivalence(model, lm, batches, image_token_id, args.verify_tolerance)
        print(f"vision-cache equivalence OK: max |Δlogits| {report['max_abs_logit_diff']:.3g}, "
              f"max |Δloss| {report['max_abs_loss_diff']:.3g} (tol {args.verify_tolerance})")

    if args.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        # See --per_device_eval_batch_size's own help text for why this isn't
        # just left to TrainingArguments' default (a hardcoded 8, independent
        # of whatever train batch size was actually tuned to fit -- the real
        # cause of a real OOM inside evaluate() after 200 stable training
        # steps). Mirroring train's batch size is a safe floor, not
        # necessarily eval's true ceiling -- pass --per_device_eval_batch_size
        # explicitly to raise it once that's worth tuning.
        per_device_eval_batch_size=(args.per_device_eval_batch_size
                                    if args.per_device_eval_batch_size is not None
                                    else args.per_device_train_batch_size),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        auto_find_batch_size=args.auto_find_batch_size,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        gradient_checkpointing=args.gradient_checkpointing,
        # use_reentrant=False: PyTorch's own current recommendation over the
        # historical (and still-default) reentrant implementation. Reentrant
        # checkpointing re-enters autograd via a second forward call and, per
        # PyTorch's docs, carries real extra memory/compute overhead versus
        # non-reentrant (which uses hooks instead) — a genuine, standard
        # memory saving on the recompute path, not a training-behavior
        # change (same activations recomputed, same gradients produced).
        gradient_checkpointing_kwargs=(
            {"use_reentrant": False} if args.gradient_checkpointing else None),
        bf16=(args.dtype == "bfloat16"),
        fp16=(args.dtype == "float16"),
        logging_steps=args.logging_steps,
        eval_strategy="steps" if val_ds else "no",
        eval_steps=args.eval_steps,
        eval_on_start=args.eval_on_start if val_ds else False,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,   # the collator needs image_path/weight through
        seed=args.seed,
        report_to=["wandb"] if args.wandb_project else [],
        run_name=args.run_name,
    )

    RFTTrainer = build_trainer_class()
    trainer = RFTTrainer(model=peft_model, args=training_args, train_dataset=train_ds,
                         eval_dataset=val_ds, data_collator=collator,
                         callbacks=[_build_memory_logger_callback()])
    trainer.vision_cache = cache
    trainer.image_token_id = image_token_id
    trainer.decoder = lm
    trainer.use_example_weights = use_weights

    # Checkpoints ARE written every --save_steps (default 200) under
    # args.output_dir/checkpoint-<step> (model, optimizer, scheduler, RNG,
    # trainer_state.json) — TrainingArguments' save_strategy="steps" above
    # already does this unconditionally, regardless of whether the run
    # finishes cleanly. What was MISSING is resuming from them: trainer.train()
    # with no argument always starts step 0, so a naive rerun after a crash
    # would silently re-train from scratch rather than continue, wasting
    # every hour already spent. get_last_checkpoint (the standard HF idiom,
    # the same one used in transformers' own example scripts) finds the
    # highest-step checkpoint under output_dir, if any, and resumes from it;
    # if none exists yet (a genuinely first run) it returns None and
    # trainer.train(resume_from_checkpoint=None) behaves exactly as before.
    # This means simply re-submitting this same job after a crash (steps 1-2
    # are cheap and deterministic, so redoing them is harmless) now actually
    # continues training instead of restarting it.
    from transformers.trainer_utils import get_last_checkpoint
    resume_checkpoint = get_last_checkpoint(args.output_dir) if os.path.isdir(args.output_dir) else None
    if resume_checkpoint:
        print(f"Found existing checkpoint, resuming from: {resume_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(os.path.join(args.output_dir, "adapter"))
    processor.save_pretrained(os.path.join(args.output_dir, "adapter"))

    with open(os.path.join(args.output_dir, "train_config.json"), "w", encoding="utf-8") as fh:
        json.dump({**vars(args), **param_stats,
                   "image_token_id": image_token_id,
                   "n_train_examples": len(train_ds),
                   "n_val_examples": len(val_ds) if val_ds else 0,
                   "vision_cache": cache.stats() if cache else None}, fh, indent=1)
    print(f"Saved adapter to {os.path.join(args.output_dir, 'adapter')}")


def _diagnose_image_token_id(batch: Dict[str, Any], image_token_id: int, tokenizer,
                             model, decoder) -> None:
    """Print concrete, checkable evidence about whether `image_token_id` and
    the token-substitution assumption this whole cache design relies on (N
    soft-image-token positions embedded directly in input_ids, replaceable
    via a masked_scatter) actually hold for this architecture.

    Runs UNCONDITIONALLY before verify_equivalence, on a REAL training
    example, so even if verify_equivalence still fails afterward, the log
    already has real evidence in it instead of only a bare Δlogits number —
    built specifically because two structurally DIFFERENT feature-computation
    mechanisms (an earlier get_image_features()-based one, and the current
    hook-based capture) produced the IDENTICAL Δlogits/Δloss on a real run,
    which rules out feature computation as the cause and points at
    image_token_id / the injection mechanism instead — see vision_cache.py's
    module docstring.
    """
    input_ids = batch["input_ids"]
    ids = input_ids[0].tolist()
    positions = [i for i, t in enumerate(ids) if t == image_token_id]
    print(f"[diag] image_token_id={image_token_id} appears {len(positions)} time(s) in a "
          f"real example (sequence length {len(ids)}).")
    if not positions:
        print("[diag]   ZERO occurrences — inject_image_features's own count check should "
              "already have raised on this; seeing this print without that error firing "
              "first is itself worth investigating.")
        return
    runs = 1
    for a, b in zip(positions, positions[1:]):
        if b != a + 1:
            runs += 1
    print(f"[diag]   forms {runs} contiguous run(s) — "
          f"{'as expected for one real image span' if runs == 1 else 'NOT one contiguous span, suspicious: a genuine per-patch/per-crop image span is normally one unbroken run, so this id is likely matching something other than the real expanded image tokens'}.")
    ctx = 6
    for p in positions[: min(3, len(positions))]:
        lo, hi = max(0, p - ctx), min(len(ids), p + ctx + 1)
        window = tokenizer.convert_ids_to_tokens(ids[lo:hi])
        print(f"[diag]   around position {p}: {window}")

    # Does the model's OWN captured inputs_embeds have the SAME sequence
    # length as input_ids? If the model represents images via simple token
    # substitution (the assumption inject_image_features/masked_scatter
    # relies on), these MUST match — a mismatch would mean the real
    # architecture adds positions beyond what input_ids encodes, which no
    # amount of image_token_id tuning could fix; the injection design itself
    # would need to change.
    try:
        from vision_cache import capture_reference_image_embeddings
        owner_kwargs = batch.get("pixel_owner_kwargs")
        if owner_kwargs:
            embeds = capture_reference_image_embeddings(
                model, decoder, input_ids, batch["attention_mask"], image_token_id, **owner_kwargs[0])
            print(f"[diag]   captured embeddings shape: {tuple(embeds.shape)} "
                  f"(sliced from a captured inputs_embeds whose own sequence length should "
                  f"equal input_ids' {len(ids)} if token substitution holds).")
    except Exception as e:
        print(f"[diag]   capture probe itself raised {type(e).__name__}: {e}")


def _batch_to_device(batch: Dict[str, Any], device) -> Dict[str, Any]:
    """Move a collated batch to `device`, recursing into the nested
    `image_kwargs` dict and `pixel_owner_kwargs` list of dicts — a plain
    top-level `isinstance(v, torch.Tensor)` filter (the original approach)
    silently drops both, since neither is itself a Tensor, which would leave
    verify_equivalence with no image data to work from at all."""
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif k == "image_kwargs" and isinstance(v, dict):
            out[k] = {ik: (iv.to(device) if isinstance(iv, torch.Tensor) else iv) for ik, iv in v.items()}
        elif k == "pixel_owner_kwargs" and isinstance(v, list):
            out[k] = [{ik: (iv.to(device) if isinstance(iv, torch.Tensor) else iv) for ik, iv in item.items()}
                      for item in v]
    return out


def _load_model(model_id: str, torch_dtype, qlora: bool = False):
    """Load a multimodal checkpoint, preferring the image-text-to-text class.

    --qlora (qlora=True) loads the FROZEN base weights in 4-bit
    (bitsandbytes NF4, double-quantized) instead of full bf16/fp16 — the
    standard QLoRA recipe (Dettmers et al. 2023, "QLoRA: Efficient
    Finetuning of Quantized LLMs"). This only changes how the BASE weights
    are STORED; the LoRA adapter matrices this project trains stay
    full-precision (peft dequantizes the frozen base on the fly for the
    adapter's own matmul), so what actually gets trained — and how the
    resulting adapter is served (plain LoRA via --lora_adapter_path, never
    quantized) — is unaffected. What changes is memory: the ~11.9B-parameter
    decoder drops from ~24GB (bf16) to ~6GB (4-bit), freeing real per-GPU
    headroom for a bigger --per_device_train_batch_size on one 48GB card.
    Requires the `bitsandbytes` package.
    """
    from transformers import AutoModelForCausalLM
    quantization_config = None
    if qlora:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )
    try:
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch_dtype, device_map="auto",
            quantization_config=quantization_config)
    except Exception:
        return AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch_dtype, device_map="auto",
            quantization_config=quantization_config)


if __name__ == "__main__":
    main()
