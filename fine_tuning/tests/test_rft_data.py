"""
Tests for the rejection-sampling data pipeline.

  python -m unittest discover -s fine_tuning/tests

Deliberately torch-free: everything here (acceptance rule, reasoning recovery,
prompt/target reconstruction, splitting) is plain data manipulation and runs on
a laptop, so the part of the fine-tune that decides WHAT the model learns can
be checked without a GPU. The trainer itself is not covered here.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from rft_common import (BuildStats, examples_for_record, group_key, image_verdict,  # noqa: E402
                        post_id_of, split_taxonomy_reasoning)
from src.models.prompts import TaxonomyResponse, MaterialResponse, ObjectExtractionResponse  # noqa: E402


def entity(nature=True, biotic=None, material=None, obj="thing", route="vlm_full"):
    return {"object": obj, "final_nature": nature, "final_biotic": biotic,
            "final_material": material, "label_route": route}


def record(gt_nature=True, gt_biotic=None, gt_material=None, finals=None, **extra):
    rec = {
        "image_path": "/data/big_5/twitter/12345_0.jpg",
        "targets": [{"class_name": "scene", "gt_nature": gt_nature,
                     "gt_biotic": gt_biotic, "gt_material": gt_material}],
        "caption": "A caption.",
        "objects": [], "object_labels": [], "object_finals": finals or [],
        "extraction_reasoning": "Step 1: ...", "extraction_parse_failed": False,
    }
    rec.update(extra)
    return rec


class TestAcceptanceRule(unittest.TestCase):
    def test_non_nature_accepted_only_with_zero_nature_entities(self):
        clean = record(gt_nature=False, finals=[entity(nature=False)])
        self.assertTrue(image_verdict(clean).accepted)

        hallucinated = record(gt_nature=False, finals=[entity(nature=True, biotic=True, material=True)])
        verdict = image_verdict(hallucinated)
        self.assertFalse(verdict.accepted)
        self.assertEqual(verdict.reason, "predicted_nature_on_non_nature_image")

    def test_nature_needs_at_least_one_nature_entity(self):
        rec = record(gt_nature=True, gt_biotic=[True], gt_material=[True],
                     finals=[entity(nature=False)])
        self.assertEqual(image_verdict(rec).reason, "no_nature_entity_on_nature_image")

    def test_strict_requires_one_entity_to_carry_both_axes(self):
        # GT: biotic + material. One entity is biotic/immaterial, another is
        # abiotic/material — between them every axis is covered, but neither
        # is right about the scene on its own.
        rec = record(gt_nature=True, gt_biotic=[True], gt_material=[True], finals=[
            entity(biotic=True, material=False, obj="cartoon dog"),
            entity(biotic=False, material=True, obj="rock"),
        ])
        self.assertFalse(image_verdict(rec, "strict").accepted)
        self.assertTrue(image_verdict(rec, "lenient").accepted)

        rec_one = record(gt_nature=True, gt_biotic=[True], gt_material=[True],
                         finals=[entity(biotic=True, material=True, obj="dog")])
        self.assertTrue(image_verdict(rec_one, "strict").accepted)
        self.assertTrue(image_verdict(rec_one, "lenient").accepted)

    def test_coder_disagreement_accepts_either_direction(self):
        # gt_material == [True, False]: the coders split, and per the project
        # convention the image genuinely counts as both.
        rec = record(gt_nature=True, gt_biotic=[True], gt_material=[True, False],
                     finals=[entity(biotic=True, material=False, obj="painted flower")])
        self.assertTrue(image_verdict(rec, "strict").accepted)
        # Lenient wants BOTH directions present among the entities, so one
        # entity is not enough for it — the two rules genuinely differ here.
        self.assertFalse(image_verdict(rec, "lenient").accepted)

    def test_missing_gt_is_rejected_not_treated_as_non_nature(self):
        verdict = image_verdict(record(gt_nature=None, finals=[]))
        self.assertFalse(verdict.accepted)
        self.assertEqual(verdict.reason, "gt_nature_missing")

    def test_unknown_rule_raises(self):
        with self.assertRaises(ValueError):
            image_verdict(record(), rule="loose")


class TestReasoningRecovery(unittest.TestCase):
    def test_splits_the_non_nature_wording(self):
        combined = ("The target entity is a glass. It fails the criteria for nature. "
                    "Not applicable since the entity is not nature.")
        head, tail = split_taxonomy_reasoning(combined)
        self.assertTrue(head.endswith("fails the criteria for nature."))
        self.assertTrue(tail.startswith("Not applicable"))

    def test_splits_the_nature_wording(self):
        combined = ("The target entity is a tree. This fulfills the inclusion criteria. "
                    "Since nature is 'yes', I evaluate the sub-axes. A tree is flora.")
        head, tail = split_taxonomy_reasoning(combined)
        self.assertTrue(head.endswith("inclusion criteria."))
        self.assertTrue(tail.startswith("Since nature is 'yes'"))

    def test_returns_none_rather_than_guessing_a_boundary(self):
        self.assertIsNone(split_taxonomy_reasoning("One flat sentence with no step marker."))
        self.assertIsNone(split_taxonomy_reasoning(""))
        self.assertIsNone(split_taxonomy_reasoning(None))

    def test_uses_the_last_opener_not_the_first(self):
        # "Not applicable" appearing inside Step 1 must not be mistaken for the
        # boundary when a real Step-2 opener follows it.
        combined = ("The exclusion for packaging is Not applicable here. It is nature. "
                    "Since nature is 'yes', I evaluate the sub-axes.")
        head, tail = split_taxonomy_reasoning(combined)
        self.assertIn("packaging", head)
        self.assertTrue(tail.startswith("Since nature is 'yes'"))


class TestExampleReconstruction(unittest.TestCase):
    def full_label(self, nature="yes", life="biotic", tang="material"):
        return {"reasoning": "It is a tree. Since nature is 'yes', I evaluate the sub-axes.",
                "nature": nature, "biotic": life, "material": tang,
                "parse_failed": False, "vlm_called": True}

    def build(self, **kw):
        rec = record(**kw)
        return examples_for_record(rec, "twitter", "test/model",
                                   ("extraction", "label_full", "label_material"), BuildStats())

    def test_target_key_order_matches_the_schema(self):
        # TaxonomyResponse is an interleaved chain of thought: the reasoning
        # field precedes the verdict it justifies. Emitting the keys in a
        # different order would train the model to answer before thinking.
        examples = self.build(
            objects=["tree"], object_labels=[self.full_label()],
            finals=[entity(obj="tree", biotic=True, material=True)])
        target = next(e for e in examples if e.stage == "label_full").target
        self.assertEqual(list(json.loads(target).keys()), list(TaxonomyResponse.model_fields))

    def test_extraction_target_matches_its_schema(self):
        examples = self.build(objects=[], object_labels=[], finals=[])
        target = next(e for e in examples if e.stage == "extraction").target
        self.assertEqual(list(json.loads(target).keys()), list(ObjectExtractionResponse.model_fields))

    def test_material_target_matches_its_schema_and_prompt_carries_the_mapped_biotic(self):
        label = {"reasoning": "A real sky.", "nature": None, "biotic": None,
                 "material": "material", "parse_failed": False, "vlm_called": True}
        examples = self.build(
            objects=["sky"], object_labels=[label],
            finals=[entity(obj="sky", biotic=False, material=True, route="mapped_nature_material")])
        example = next(e for e in examples if e.stage == "label_material")
        self.assertEqual(list(json.loads(example.target).keys()), list(MaterialResponse.model_fields))
        self.assertIn('"sky"', example.prompt)
        self.assertEqual(example.system_key, "material")

    def test_human_terms_produce_no_labeling_example(self):
        label = {"reasoning": None, "nature": None, "biotic": None, "material": None,
                 "parse_failed": False, "vlm_called": False}
        examples = self.build(objects=["woman"], object_labels=[label],
                              finals=[entity(obj="woman", nature=False, route="human_exclusion")])
        self.assertEqual([e.stage for e in examples], ["extraction"])

    def test_parse_failures_are_dropped(self):
        label = {"reasoning": None, "nature": None, "biotic": None, "material": None,
                 "parse_failed": True, "vlm_called": True}
        stats = BuildStats()
        rec = record(objects=["tree"], object_labels=[label], finals=[entity(obj="tree")],
                     extraction_parse_failed=True)
        examples = examples_for_record(rec, "twitter", "m", ("extraction", "label_full"), stats)
        self.assertEqual(examples, [])
        self.assertEqual(stats.counts.get("extraction_skipped_parse_failed"), 1)
        self.assertEqual(stats.counts.get("label_skipped_parse_failed"), 1)

    def test_inconsistent_nature_yes_but_subaxes_none_is_dropped(self):
        # Real, measured case (16/27213 on the gemma-4-12B-it BIG-5 artifacts):
        # the model says nature="yes" but leaves life_category/tangibility as
        # "none", violating the schema's own instruction. Must not be trained
        # on verbatim.
        label = self.full_label(nature="yes", life="none", tang="none")
        stats = BuildStats()
        rec = record(objects=["tree"], object_labels=[label], finals=[entity(obj="tree")])
        examples = examples_for_record(rec, "twitter", "m", ("label_full",), stats)
        self.assertEqual(examples, [])
        self.assertEqual(stats.counts.get("label_full_skipped_inconsistent_subaxes"), 1)

    def test_inconsistent_nature_no_but_subaxes_set_is_dropped(self):
        # Mirror case: not observed in real data, but guarded defensively.
        label = self.full_label(nature="no", life="biotic", tang="material")
        stats = BuildStats()
        rec = record(objects=["tree"], object_labels=[label], finals=[entity(obj="tree", nature=False)])
        examples = examples_for_record(rec, "twitter", "m", ("label_full",), stats)
        self.assertEqual(examples, [])
        self.assertEqual(stats.counts.get("label_full_skipped_inconsistent_subaxes"), 1)

    def test_consistent_nature_no_with_subaxes_none_is_kept(self):
        # nature="no" with life_category/tangibility genuinely "none" is the
        # NORMAL, correct shape for a non-nature verdict — must not be dropped.
        label = self.full_label(nature="no", life="none", tang="none")
        label["reasoning"] = ("The target entity is a car. It fails the criteria for nature. "
                              "Not applicable since the entity is not nature.")
        examples = self.build(objects=["car"], object_labels=[label],
                              finals=[entity(obj="car", nature=False)])
        self.assertEqual([e.stage for e in examples], ["extraction", "label_full"])

    def test_unsplittable_reasoning_is_dropped_not_fabricated(self):
        label = self.full_label()
        label["reasoning"] = "A flat sentence with no step marker at all."
        stats = BuildStats()
        rec = record(objects=["tree"], object_labels=[label], finals=[entity(obj="tree")])
        examples = examples_for_record(rec, "twitter", "m", ("label_full",), stats)
        self.assertEqual(examples, [])
        self.assertEqual(stats.counts.get("label_full_skipped_unsplittable_reasoning"), 1)

    def test_raw_reasoning_fields_win_over_recovery(self):
        # Artifacts written after the fields were added should be used
        # verbatim, not re-split from the joined string.
        label = self.full_label()
        label["nature_reasoning"] = "RAW STEP ONE."
        label["sub_axes_reasoning"] = "RAW STEP TWO."
        examples = examples_for_record(
            record(objects=["tree"], object_labels=[label], finals=[entity(obj="tree")]),
            "twitter", "m", ("label_full",), BuildStats())
        target = json.loads(examples[0].target)
        self.assertEqual(target["nature_reasoning"], "RAW STEP ONE.")
        self.assertEqual(target["sub_axes_reasoning"], "RAW STEP TWO.")

    def test_extraction_prompt_embeds_this_images_caption(self):
        examples = self.build(objects=[], object_labels=[], finals=[],
                              caption="A very distinctive caption.")
        prompt = next(e for e in examples if e.stage == "extraction").prompt
        self.assertIn("A very distinctive caption.", prompt)


class TestGrouping(unittest.TestCase):
    def test_post_id_strips_the_slot_index(self):
        self.assertEqual(post_id_of("/x/1477823022849568769_0.jpg"), "1477823022849568769")
        self.assertEqual(post_id_of("/x/-3NEKN7YEcCmPzGy_8.png"), "-3NEKN7YEcCmPzGy")

    def test_unexpected_name_falls_back_to_the_whole_stem(self):
        self.assertEqual(post_id_of("/x/oddname.jpg"), "oddname")

    def test_group_is_namespaced_by_platform(self):
        self.assertNotEqual(group_key("/x/1_0.jpg", "twitter"), group_key("/x/1_0.jpg", "weibo"))


class TestMakeSplits(unittest.TestCase):
    """End-to-end over a synthetic artifact — the property that matters is
    that no POST is ever split across two splits."""

    def test_posts_never_span_splits(self):
        repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        with tempfile.TemporaryDirectory() as tmp:
            artifact = os.path.join(tmp, "vlm_responses_synthetic.jsonl")
            with open(artifact, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"record_type": "header", "dataset": "big5_twitter",
                                     "model_name": "synthetic"}) + "\n")
                for post in range(300):
                    for slot in range(1 + post % 4):
                        fh.write(json.dumps({
                            "image_path": f"/img/{post}_{slot}.jpg",
                            "targets": [{"gt_nature": (post + slot) % 2 == 0}],
                        }) + "\n")
            out = os.path.join(tmp, "splits")
            subprocess.run(
                [sys.executable, os.path.join(repo, "fine_tuning/make_splits.py"),
                 "--artifact", artifact, "--out", out],
                check=True, capture_output=True, cwd=repo)

            with open(os.path.join(out, "splits.json"), encoding="utf-8") as fh:
                splits = json.load(fh)
            by_post = {}
            for image, split in splits["image_split"].items():
                by_post.setdefault(post_id_of(image), set()).add(split)
            spanning = {p: s for p, s in by_post.items() if len(s) > 1}
            self.assertEqual(spanning, {}, f"posts split across splits: {list(spanning)[:5]}")

            with open(os.path.join(out, "summary.json"), encoding="utf-8") as fh:
                summary = json.load(fh)
            # Whole-post assignment cannot hit 70/10/20 exactly; a couple of
            # points of slack is the price of not leaking near-duplicates.
            self.assertAlmostEqual(summary["train"]["image_share"], 0.70, delta=0.03)
            self.assertAlmostEqual(summary["test"]["image_share"], 0.20, delta=0.03)


if __name__ == "__main__":
    unittest.main()
