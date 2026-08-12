import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

import balanced_metrics
from balanced_dataset import (
    EpochTemplateSampler,
    ManifestDataset,
    ManifestRecord,
    load_manifest,
)
from balanced_lightning_module import (
    clinical_checkpoint_eligibility,
    clinical_checkpoint_score,
    weighted_causal_loss,
)
from clinical_fixed import extract_assertions


balanced_metrics.extract_assertions = extract_assertions


def record(case, template, reports=1, duplicate=None):
    return ManifestRecord(
        case,
        Path("feature.pt"),
        tuple(Path(f"{case}-{i}.txt") for i in range(reports)),
        (),
        duplicate or case,
        template,
        1.0,
    )


def test_epoch_sampler_caps_templates_and_rotates_annotations():
    records = [record(f"A{i}", "same", 2, "near-same") for i in range(5)] + [
        record("B0", "unique")
    ]
    sampler = EpochTemplateSampler(records, cap=3, seed=42)
    first = sampler.global_indices()
    assert len(first) == 4
    assert (
        max(Counter(records[index].template_group for index, _ in first).values()) == 3
    )
    assert len({index for index, _ in first}) == len(first)
    sampler.set_epoch(1)
    second = sampler.global_indices()
    assert first != second
    first_by_case, second_by_case = dict(first), dict(second)
    shared = {
        index
        for index in set(first_by_case) & set(second_by_case)
        if len(records[index].report_paths) > 1
    }
    assert shared and all(
        first_by_case[index] != second_by_case[index] for index in shared
    )


def test_missing_tooth_is_a_positive_finding():
    findings = extract_assertions("Absence of teeth 3.8 and 4.8.")
    assert {(item.tooth_number, item.polarity) for item in findings} == {
        ("38", "positive"),
        ("48", "positive"),
    }


def test_clinical_metrics_detect_side_negation_unsupported_and_duplicates():
    negated = extract_assertions("No missing teeth are identified.")
    assert {(item.tooth_number, item.polarity) for item in negated} == {
        (None, "negative")
    }
    predictions = [
        "Impacted tooth 48. Osteolytic lesion.",
        "Impacted tooth 48. Osteolytic lesion.",
    ]
    references = [
        ["Impacted tooth 38. No osteolytic lesion."],
        ["Impacted tooth 38. No osteolytic lesion."],
    ]
    metrics, details, cases = balanced_metrics.clinical_metrics(
        predictions, references, ["a", "b"]
    )
    assert metrics["side_accuracy"] == 0.0
    assert metrics["negation_contradiction_rate"] > 0
    assert (
        metrics["unsupported_count"] == 2.0
    )  # The two wrong-side impacted teeth; lesions are contradictions.
    assert metrics["unique_rate"] == 0.5
    assert metrics["max_duplicate_count"] == 2.0
    assert all(len(row["contradictions"]) == 1 for row in cases)


def test_weighted_causal_loss_matches_manual_calculation():
    logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]]])
    labels = torch.tensor([[-100, 0, 1]])
    weights = torch.tensor([[0.0, 1.0, 2.0]])
    weighted, unweighted = weighted_causal_loss(
        logits, labels, weights, torch.tensor([1.0])
    )
    losses = torch.stack(
        (
            F.cross_entropy(logits[:, 0], torch.tensor([0])),
            F.cross_entropy(logits[:, 1], torch.tensor([1])),
        )
    )
    assert torch.allclose(unweighted, losses.mean())
    assert torch.allclose(weighted, (losses[0] + 2 * losses[1]) / 3)


def test_sample_weight_is_not_cancelled_for_batch_size_one():
    logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]]])
    labels = torch.tensor([[-100, 0, 1]])
    token_weights = torch.tensor([[0.0, 1.0, 2.0]])
    unit, _ = weighted_causal_loss(logits, labels, token_weights, torch.tensor([1.0]))
    doubled, _ = weighted_causal_loss(
        logits, labels, token_weights, torch.tensor([2.0])
    )
    assert torch.allclose(doubled, 2 * unit)


def test_sampler_caps_near_templates_and_spreads_annotation_phases():
    records = [
        record(f"case-{index}", f"exact-{index}", reports=3, duplicate="near")
        for index in range(7)
    ]
    sampler = EpochTemplateSampler(records, cap=3, near_cap=3, seed=42)
    first = sampler.global_indices()
    assert len(first) == 3
    assert len({annotation for _index, annotation in first}) > 1
    sampler.set_epoch(1)
    second = sampler.global_indices()
    assert {index for index, _annotation in first} != {
        index for index, _annotation in second
    }


def test_validation_image_shuffle_is_deterministic_derangement(tmp_path):
    records = []
    for index in range(4):
        feature = tmp_path / f"feature-{index}.pt"
        report = tmp_path / f"report-{index}.txt"
        torch.save({"spatial": torch.full((1, 2, 2, 2), float(index))}, feature)
        report.write_text(f"report {index}", encoding="utf-8")
        records.append(
            ManifestRecord(
                str(index), feature, (report,), (), str(index), str(index), 1.0
            )
        )
    first = ManifestDataset(
        records, (1, 1, 1), 1, include_shuffled_image=True, shuffle_seed=42
    )
    second = ManifestDataset(
        records, (1, 1, 1), 1, include_shuffled_image=True, shuffle_seed=42
    )
    assert first.shuffle_sources == second.shuffle_sources
    assert all(
        destination != source for destination, source in first.shuffle_sources.items()
    )
    assert first[0]["shuffle_source_case_id"] != first[0]["case_id"]


def test_clinical_checkpoint_score_penalizes_hallucinations():
    base = {
        "tooth_f1_micro": 0.5,
        "tooth_f1_macro": 0.5,
        "case_specific_recall_micro": 0.5,
        "case_specific_recall_macro": 0.5,
        "side_accuracy": 0.5,
        "unsupported_finding_rate": 0.0,
        "negation_contradiction_rate": 0.0,
    }
    clean = clinical_checkpoint_score(base, {})
    hallucinated = clinical_checkpoint_score(
        {**base, "unsupported_finding_rate": 0.5}, {}
    )
    assert hallucinated < clean
    empty = clinical_checkpoint_score(
        {
            **base,
            "tooth_f1_micro": 0.0,
            "tooth_f1_macro": 0.0,
            "case_specific_recall_micro": 0.0,
            "case_specific_recall_macro": 0.0,
            "side_accuracy": 0.0,
            "empty_generation_rate": 1.0,
        },
        {},
    )
    assert empty < hallucinated < clean


def test_real_manifests_have_no_duplicate_leakage_and_all_annotations():
    root = Path(__file__).parents[1] / "datalist"
    if not (root / "train_manifest.jsonl").exists():
        return
    train, valid = (
        load_manifest(root / "train_manifest.jsonl"),
        load_manifest(root / "valid_manifest.jsonl"),
    )
    assert len(train) + len(valid) == 627
    assert sum(len(row.report_paths) for row in train + valid) == 1004
    assert not (
        {row.duplicate_group for row in train} & {row.duplicate_group for row in valid}
    )
    audit = json.loads((root / "manifest_audit.json").read_text())
    assert audit["unused_reports"] == 0


def test_assertion_scope_binds_findings_to_local_teeth_and_negation():
    findings = extract_assertions(
        "Tooth 46 is present and tooth 48 is impacted. "
        "No osteolytic lesion, but tooth 38 is impacted. Teeth 31 to 33 are missing."
    )
    impacted = {
        (item.tooth_number, item.polarity)
        for item in findings
        if item.concept == "impacted"
    }
    lesions = {
        (item.tooth_number, item.polarity)
        for item in findings
        if item.concept == "lesion"
    }
    missing = {item.tooth_number for item in findings if item.concept == "missing"}
    assert impacted == {("48", "positive"), ("38", "positive")}
    assert lesions == {(None, "negative")}
    assert missing == {"31", "32", "33"}


def test_generation_quality_and_case_specific_f1_metrics():
    repeated = (
        "Presence of a prosthetic crown on tooth 35. "
        "Presence of a prosthetic crown on tooth 35. Tooth 3.9 is missing"
    )
    metrics, details, _cases = balanced_metrics.clinical_metrics(
        [repeated], [["Presence of a prosthetic crown on tooth 35."]], ["case"]
    )
    assert metrics["case_specific_precision_micro"] < 1.0
    assert metrics["case_specific_f1_micro"] < 1.0
    assert metrics["truncated_generation_rate"] == 1.0
    assert metrics["invalid_fdi_case_rate"] == 1.0
    assert metrics["repetition_case_rate"] == 1.0
    assert details["generation_quality"]["invalid_fdi_count"] == 1


def test_checkpoint_hard_gate_rejects_unsafe_generation():
    safe = {
        "unsupported_finding_rate": 0.2,
        "truncated_generation_rate": 0.0,
        "invalid_fdi_case_rate": 0.0,
        "tooth_precision_micro": 0.3,
    }
    eligible, failures = clinical_checkpoint_eligibility(safe, {})
    assert eligible and not failures
    unsafe = {**safe, "unsupported_finding_rate": 0.8, "invalid_fdi_case_rate": 0.1}
    eligible, failures = clinical_checkpoint_eligibility(unsafe, {})
    assert not eligible
    assert len(failures) == 2
