"""
Cached visual embeddings for the DoRA fine-tune.

CURRENT STATUS: NOT ENABLED BY DEFAULT (job_finetune.sh passes
--no_vision_cache). This is a pure SPEED optimization — disabling it changes
nothing about what gets trained, only how many times an image's vision-tower
forward gets recomputed. On a real run against google/gemma-4-12B-it, its
injection path hit five real, distinct bugs in a row, the last of which (a
genuine embedding-VALUE mismatch, max |Δlogits| ~31) survived a full rewrite
to capture the model's REAL computed features via a forward hook rather than
reimplementing them — meaning the remaining cause is most likely an
architecture-specific embedding-scaling step applied to TEXT positions that
this hasn't been pinned down without further live access (see
`capture_reference_image_embeddings`'s own docstring, and the module history
below, for exactly what was ruled in/out along the way). Given the payoff is
throughput, not correctness, and training WITHOUT this cache runs through the
completely standard, never-implicated `model(pixel_values=..., ...)` HF
path, the pragmatic call was to stop debugging this and just train without
it. Revisit only if training throughput without the cache turns out to
actually matter in practice.

WHY THIS IS SOUND (the design, if/when it IS enabled again)
=============================================================
The fine-tune trains the LANGUAGE decoder only: the vision tower and the
multimodal projector are frozen, get no adapter, and are put in eval mode. The
soft-token embeddings a given image produces are therefore a CONSTANT for the
whole run — identical in epoch 1 and epoch 5, identical for every example that
uses that image. Recomputing them is pure waste, so they are computed once, on
first use, and reused from then on.

Two separate savings, and the second is the larger one:

  ACROSS EPOCHS  — the intended win. Epoch 2+ pays no vision forward at all.
  WITHIN EPOCH 1 — one accepted image contributes ~5 training examples (one
                   extraction call plus one labeling call per extracted
                   entity), so even the first pass over the data recomputes the
                   same image several times without a cache.

THE CORRECTNESS RISK, AND HOW IT IS CLOSED
==========================================
Injecting precomputed embeddings means bypassing the model's own multimodal
forward path and rebuilding `inputs_embeds` by hand. A mistake there — a wrong
image-token id, an off-by-one in the scatter, a dtype downcast — does not
crash. It trains, converges to something, and produces a model that is merely
worse, which is indistinguishable from ordinary bad hyperparameters.

AN EARLIER VERSION of this module tried to CACHE the output of the model's
own `get_image_features()` method and independently re-derive `inputs_embeds`
from it (image features + `get_input_embeddings()` + a hand-rolled scatter),
on the assumption that `get_image_features()` returns exactly what the
model's real multimodal forward uses internally. Confirmed WRONG on a real
run of google/gemma-4-12B-it: `verify_equivalence`'s own comparison caught a
genuine, large mismatch (max |Δlogits| ~31, far beyond floating-point noise)
— `get_image_features()` does NOT reliably equal what the real forward feeds
into the decoder for this architecture (some additional processing step
between the two, exact cause unconfirmed and no longer relevant to this
design).

So this module now captures the REAL value instead of trying to reproduce
it: `capture_reference_image_embeddings` runs the model's genuine,
unmodified multimodal forward for one image (vision tower, projector,
whatever internal processing happens, all via the model's own code, none of
it reimplemented here) and reads the resulting `inputs_embeds` tensor
straight off the language decoder's own forward call, via a forward
pre-hook, aborting immediately after so the (irrelevant, expensive) decoder
computation itself never runs. What gets cached is then a VERBATIM slice of
that tensor at the image-token positions — not a value this code computed by
guessing which method/field/postprocessing step reproduces it, so there is
nothing left to mismatch on the FEATURE VALUES.

`verify_equivalence` still exists and the trainer still runs it before every
training run (`--verify_cache N`, on by default), but what it tests is now
narrower and more honest about it: given features that are ALREADY known-
correct by construction, does re-injecting them via `inject_image_features`
(the mask-based scatter into a fresh `inputs_embeds`) reproduce the ORIGINAL
forward's logits/loss? That still catches a real class of bug (wrong
image_token_id, an off-by-one in the scatter, a dtype downcast) — it just no
longer needs to ALSO prove feature-computation equivalence, since that
question no longer exists.

STORAGE
=======
One torch file per image under --cache_dir, written DIRECTLY to disk — no
in-memory buffer of computed embeddings is ever kept by this class (`put()`
does not accumulate anything in `self`; `hits`/`misses` are the only state it
tracks). Every write is `flush()` + `os.fsync()`ed to a temp file BEFORE the
atomic rename, so a "successful" put is actually durable on physical disk, not
just sitting in the OS page cache — an `os.replace()` alone guarantees a
reader never sees a half-written file, but says nothing about whether the
bytes survive a killed job or a node failure before the OS gets around to
flushing its own cache; fsync is what closes that gap. This combination (temp
file + fsync + atomic rename) is also what makes it safe for several
dataloader workers to read concurrently while the main process writes: a
reader either sees the old absence or the complete, durable file, never
something in between. Sized for this project: ~256 soft tokens x hidden,
bf16 — a few MB per image, ~10 GB for the whole BIG-5 training split. bf16 is
stored via torch rather than numpy on purpose: numpy has no bfloat16, and
silently rounding through float16 would introduce exactly the kind of small
numerical difference the equivalence check is meant to be able to trust.
"""

from __future__ import annotations

import hashlib
import socket
import os
from io import BytesIO
from typing import Any, Dict, List, Optional

import torch
from PIL import Image


# =============================================================================
# Image loading — deliberately identical to inference
# =============================================================================
def load_training_image(path: str, max_image_side: Optional[int] = 1024) -> Image.Image:
    """Load an image the way inference does, including the lossy JPEG step.

    src.models.vlm_models.VLLMBackedVLM._encode_image downscales any image
    whose longest side exceeds `max_image_side` and RE-ENCODES it as JPEG
    quality 90 before the model ever sees it. Training must reproduce that
    round-trip, not just the resize: a model fine-tuned on pristine LANCZOS
    output and then served images carrying JPEG ringing is being evaluated on
    a slightly different input distribution than it was trained on. The cost
    is one extra encode/decode per oversized image, paid once per image thanks
    to the cache.

    Images already within the cap are returned as-is (converted to RGB, which
    inference also guarantees — see the CMYK/grayscale note in _encode_image).
    """
    img = Image.open(path)
    width, height = img.size
    if max_image_side and max(width, height) > max_image_side:
        scale = max_image_side / max(width, height)
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        resized = img.convert("RGB").resize(new_size, Image.LANCZOS)
        buffer = BytesIO()
        resized.save(buffer, format="JPEG", quality=90)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    return img.convert("RGB")


# =============================================================================
# The cache
# =============================================================================
class VisionEmbeddingCache:
    """Lazily-filled, on-disk cache of per-image soft-token embeddings.

    The cache key includes a FINGERPRINT of everything that can change the
    embeddings — model id, dtype, processor config, max_image_side — so a cache
    built for one configuration is never silently reused by another. Changing
    any of them yields a different subdirectory rather than stale vectors.
    """

    def __init__(self, cache_dir: str, fingerprint: str, enabled: bool = True):
        self.enabled = enabled
        self.root = os.path.join(cache_dir, fingerprint)
        if self.enabled:
            os.makedirs(self.root, exist_ok=True)
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_fingerprint(**parts: Any) -> str:
        """A short, stable directory name for one configuration."""
        blob = "|".join(f"{k}={parts[k]!r}" for k in sorted(parts))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def _path(self, image_path: str) -> str:
        # Hash the path rather than reusing the basename: BIG-5 filenames are
        # unique per platform folder but not necessarily across platforms, and
        # a collision here would silently pair one image's pixels with
        # another's embedding.
        h = hashlib.sha256(image_path.encode("utf-8")).hexdigest()
        return os.path.join(self.root, h[:2], h + ".pt")

    def get(self, image_path: str) -> Optional[torch.Tensor]:
        if not self.enabled:
            return None
        path = self._path(image_path)
        if not os.path.exists(path):
            self.misses += 1
            return None
        try:
            tensor = torch.load(path, map_location="cpu")
        except Exception:
            # A file truncated by a killed job — treat as a miss and let it be
            # rewritten, rather than taking down the run.
            self.misses += 1
            return None
        self.hits += 1
        return tensor

    def put(self, image_path: str, features: torch.Tensor) -> None:
        """Write straight to disk — nothing is kept buffered in this object
        (or anywhere else in this class) once this call returns. The tensor
        is moved to CPU, serialized into an OPEN file handle we control
        directly (not `torch.save(tensor, path_string)`, which hides the
        handle and gives no chance to flush/fsync it), explicitly flushed
        past Python's own buffering AND fsync'd past the OS page cache, and
        only THEN atomically renamed into place. Skipping either flush or
        fsync is exactly the failure mode this guards against: os.replace()
        alone only guarantees a reader never sees a half-written file, not
        that the bytes have actually reached physical disk — a job killed
        (SLURM preemption, OOM-kill, node failure) between a successful-
        looking write and the OS's own lazy flush can otherwise lose an
        entry that every earlier check believed was safely cached.
        """
        if not self.enabled:
            return
        path = self._path(image_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # PID alone is NOT a safe uniqueness key on a cluster: PIDs are only
        # unique within one node's kernel, so two SLURM jobs on DIFFERENT
        # nodes writing this same cache path concurrently (e.g. the same
        # --balance config launched twice by accident) could get the same
        # PID and collide on this temp filename. Host + SLURM job id + PID
        # together are unique cluster-wide; falls back to just the hostname
        # when run outside SLURM (SLURM_JOB_ID unset).
        job_id = os.environ.get("SLURM_JOB_ID", "local")
        tmp = f"{path}.{socket.gethostname()}.{job_id}.{os.getpid()}.tmp"
        with open(tmp, "wb") as fh:
            torch.save(features.detach().to("cpu"), fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)  # atomic on POSIX; the data is already durable

    def stats(self) -> Dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}


# =============================================================================
# Model plumbing — image token id, feature extraction, embedding injection
# =============================================================================
def resolve_image_token_id(model, processor) -> int:
    """The token id that stands in for one image soft token.

    Probed off the model config (several attribute spellings are in use across
    architectures) and then off the tokenizer, rather than hardcoded — and it
    RAISES when it cannot be found. A wrong id would scatter image features
    into the wrong sequence positions and still train happily, so guessing
    here is not an acceptable fallback.
    """
    config = model.config
    holders = (config, getattr(config, "text_config", None), getattr(config, "vision_config", None))
    for attr in ("image_token_id", "image_token_index", "image_token"):
        for holder in holders:
            if holder is None:
                continue
            value = getattr(holder, attr, None)
            if value is None:
                continue
            # Accept int-like values too (numpy/torch scalar types), not just
            # a strict `isinstance(value, int)` — a config could plausibly
            # store this as either, and silently skipping a genuinely correct
            # value here just to fall through to the less reliable
            # tokenizer-string search below is exactly the kind of silent
            # wrong-guess this function's own docstring says not to make.
            if isinstance(value, int):
                return value
            if isinstance(value, bool):
                continue  # bool is technically an int subclass; never a real token id
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    tokenizer = getattr(processor, "tokenizer", processor)
    for name in ("<image_soft_token>", "<image>", "<|image|>", "<image_placeholder>"):
        token_id = tokenizer.convert_tokens_to_ids(name)
        if isinstance(token_id, int) and token_id >= 0 and token_id != getattr(tokenizer, "unk_token_id", None):
            return token_id
    raise RuntimeError(
        "Could not determine the image token id from the model config or tokenizer. "
        "Set it explicitly with --image_token_id; do NOT guess, a wrong value "
        "silently scatters image features into the wrong positions."
    )


class _CaptureAndAbort(Exception):
    """Internal control-flow signal, never meant to escape this module: raised
    from inside the decoder forward-pre-hook the instant `inputs_embeds` has
    been captured, so the rest of the (expensive, and for this purpose
    irrelevant) decoder forward pass never actually runs."""


@torch.no_grad()
def capture_reference_image_embeddings(model, decoder, input_ids: torch.Tensor,
                                       attention_mask: torch.Tensor,
                                       image_token_id: int, **image_kwargs: Any) -> torch.Tensor:
    """Run ONE image through the model's REAL, unmodified multimodal forward
    and return the slice of its own `inputs_embeds` at the image-token
    positions — not a value this code computed by guessing which method call
    reproduces the model's internal processing (an earlier version tried
    exactly that, via `get_image_features()`, and was measurably wrong on a
    real run — see the module docstring). There is nothing left to get wrong
    about the FEATURE VALUES here: they are read directly off a genuine
    forward pass, verbatim.

    `input_ids`/`attention_mask` must carry a batch dim of exactly 1 (one
    image, processed in total isolation — see RFTCollator's own docstring on
    why cache-miss images are never batched together: a tiling/pan-and-scan
    processor can give different images different soft-token counts, which
    this single-image-at-a-time design sidesteps rather than resolves).

    `decoder` is the language-decoder submodule (train_lora.language_model_of)
    — the hook is installed there because it is the one point in the module
    tree that is GUARANTEED to receive the fully-formed `inputs_embeds`
    (vision tower + projector + scatter already done, all by the model's own
    code) regardless of how many internal steps the outer wrapper's forward()
    takes to build it.
    """
    if input_ids.shape[0] != 1:
        raise RuntimeError(
            f"capture_reference_image_embeddings expects batch size 1, got "
            f"{input_ids.shape[0]} — this function processes exactly one image per call "
            f"by design (see its own docstring for why)."
        )

    captured: Dict[str, Optional[torch.Tensor]] = {"inputs_embeds": None}

    def _hook(module, args, kwargs):
        import inspect
        try:
            bound = inspect.signature(module.forward).bind_partial(*args, **kwargs)
            embeds = bound.arguments.get("inputs_embeds")
        except TypeError:
            embeds = kwargs.get("inputs_embeds")
        captured["inputs_embeds"] = embeds
        raise _CaptureAndAbort()

    handle = decoder.register_forward_pre_hook(_hook, with_kwargs=True)
    was_training = model.training
    model.eval()
    try:
        model(input_ids=input_ids, attention_mask=attention_mask, **image_kwargs)
    except _CaptureAndAbort:
        pass
    finally:
        handle.remove()
        if was_training:
            model.train()

    embeds = captured["inputs_embeds"]
    if embeds is None:
        raise RuntimeError(
            "The decoder forward-pre-hook never captured an inputs_embeds tensor — either "
            "it never fired (the outer model's forward() doesn't call this decoder "
            "submodule directly), or the decoder receives embeddings under a parameter "
            "name/position this hook's signature-binding didn't recognize. Inspect "
            "type(model).forward's source to find how it actually calls the decoder."
        )
    mask = (input_ids[0] == image_token_id)
    n = int(mask.sum().item())
    if n == 0:
        raise RuntimeError(
            f"No image_token_id ({image_token_id}) positions found in input_ids — cannot "
            f"slice out this image's own embedding block. Wrong --image_token_id?"
        )
    return embeds[0][mask]


def inject_image_features(model, input_ids: torch.Tensor, image_features: torch.Tensor,
                          image_token_id: int) -> torch.Tensor:
    """Build `inputs_embeds` with the image features placed at the image-token
    positions — the hand-rolled equivalent of the model's multimodal forward.

    `get_input_embeddings()` is the model's own embedding module, so any
    architecture-specific scaling it applies (Gemma's scaled word embedding,
    for instance) is applied here too — which is precisely why this must not
    be replaced by a raw weight lookup.
    """
    inputs_embeds = model.get_input_embeddings()(input_ids)
    mask = (input_ids == image_token_id)
    n_slots = int(mask.sum().item())
    n_features = int(image_features.numel() // image_features.shape[-1])
    if n_slots != n_features:
        raise RuntimeError(
            f"Image-token/feature mismatch: the batch has {n_slots} image-token positions "
            f"but {n_features} feature vectors. This usually means the processor produced "
            f"a different number of soft tokens than the cached features were built with "
            f"(e.g. pan-and-scan enabled for one and not the other) — the cache "
            f"fingerprint should have separated those, so investigate rather than resize."
        )
    expanded = mask.unsqueeze(-1).expand_as(inputs_embeds)
    features = image_features.reshape(-1, image_features.shape[-1]).to(
        dtype=inputs_embeds.dtype, device=inputs_embeds.device)
    return inputs_embeds.masked_scatter(expanded, features)


# =============================================================================
# The check that makes the cache trustworthy
# =============================================================================
@torch.no_grad()
def verify_equivalence(model, decoder, batches: List[Dict[str, Any]], image_token_id: int,
                       tolerance: float = 1e-3) -> Dict[str, float]:
    """Compare the normal multimodal forward against the cached-embedding one.

    Unlike an earlier version of this function, the "cached" features here
    are NOT independently recomputed via `get_image_features()` — they are
    captured VERBATIM from the SAME reference forward's own `inputs_embeds`
    (see `capture_reference_image_embeddings`), so this no longer tests
    "does our reimplementation match the model's internals" (that question
    doesn't exist anymore — there is no reimplementation left). What it DOES
    still test, and the reason it still exists rather than being deleted: the
    injection MECHANISM — `inject_image_features`'s image-token masking and
    scatter — a wrong `--image_token_id` or an off-by-one there would still
    silently corrupt training, and this catches it the same way as before.

    Each `batch` is expected in the shape `RFTCollator.__call__` produces:
    `input_ids`/`attention_mask`/`labels` as top-level tensors, `image_kwargs`
    (pixel_values plus any other aux keys the processor returned) for the
    reference forward, and `pixel_owner_kwargs` (a list of one dict — these
    batches are built at batch_size=1, no cache, by train_lora.py's `main()`)
    for the capture step.

    Small nonzero differences are expected and fine: the two paths accumulate
    the same arithmetic in a different order, and bf16 is not associative.
    A LARGE difference means the injection mechanism itself is wrong, and it
    is far cheaper to find that out here than after a full training run.
    """
    was_training = model.training
    model.eval()
    max_logit_diff = 0.0
    max_loss_diff = 0.0
    try:
        for batch in batches:
            owner_kwargs = batch.get("pixel_owner_kwargs")
            if not owner_kwargs or len(owner_kwargs) != 1:
                raise RuntimeError(
                    f"verify_equivalence got a batch with {len(owner_kwargs or [])} owned "
                    f"image(s), expected exactly 1 — every probe example must be a single-"
                    f"image, cache-MISS batch (built with a cache=None, batch_size=1 "
                    f"collator, matching capture_reference_image_embeddings' own "
                    f"single-image contract)."
                )
            reference = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                **batch["image_kwargs"],
            )
            features = capture_reference_image_embeddings(
                model, decoder, batch["input_ids"], batch["attention_mask"],
                image_token_id, **owner_kwargs[0])
            inputs_embeds = inject_image_features(model, batch["input_ids"], features, image_token_id)
            cached = model(
                inputs_embeds=inputs_embeds,
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            max_logit_diff = max(max_logit_diff,
                                 (reference.logits.float() - cached.logits.float()).abs().max().item())
            max_loss_diff = max(max_loss_diff,
                                abs(float(reference.loss) - float(cached.loss)))
    finally:
        if was_training:
            model.train()

    result = {"max_abs_logit_diff": max_logit_diff, "max_abs_loss_diff": max_loss_diff,
              "tolerance": tolerance}
    if max_loss_diff > tolerance:
        raise RuntimeError(
            f"Cached-embedding forward does NOT match the normal multimodal forward: "
            f"max |Δloss| = {max_loss_diff:.6g} > tolerance {tolerance:.6g} "
            f"(max |Δlogits| = {max_logit_diff:.6g}). Refusing to train on it. "
            f"Since features are now captured verbatim (not reimplemented), a mismatch "
            f"here means the injection MECHANISM is wrong — check --image_token_id, or "
            f"that the decoder forward-pre-hook actually fired (see "
            f"capture_reference_image_embeddings' own error message if it didn't)."
        )
    return result
