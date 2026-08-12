from __future__ import annotations

import re
import statistics
from collections import Counter, defaultdict

from nltk.stem.porter import PorterStemmer
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import meteor_score

from clinical import (
    FindingAssertion,
    extract_assertions,
    normalize_report,
    shingle_jaccard,
)


class _NoWordNet:
    @staticmethod
    def synsets(_word):
        return []


def _tokens(text):
    return text.lower().strip().split()


def report_metrics(
    predictions: list[str], references: list[str | list[str]]
) -> dict[str, float]:
    if len(predictions) != len(references):
        raise ValueError("prediction/reference length mismatch")
    if not predictions:
        return {"bleu4": 0.0, "meteor": 0.0}
    refs = [
        [_tokens(value) for value in ([item] if isinstance(item, str) else item)]
        for item in references
    ]
    hypotheses = [_tokens(value) for value in predictions]
    bleu = corpus_bleu(refs, hypotheses, smoothing_function=SmoothingFunction().method1)
    stemmer = PorterStemmer()
    meteor = sum(
        meteor_score(ref, hyp, stemmer=stemmer, wordnet=_NoWordNet())
        for ref, hyp in zip(refs, hypotheses)
    ) / len(hypotheses)
    return {"bleu4": float(bleu), "meteor": float(meteor)}


def _assertion_key(item: FindingAssertion, polarity=True):
    values = (item.concept, item.tooth_number, item.anatomy, item.side)
    return values + ((item.polarity,) if polarity else ())


def _tooth_key(item):
    return item.tooth_number, item.concept, item.polarity


def _neutral_side_key(item):
    tooth = item.tooth_number
    neutral_tooth = (
        None
        if tooth is None
        else (("upper" if tooth[0] in "12" else "lower"), tooth[1])
    )
    return item.concept, neutral_tooth, item.anatomy, item.polarity


def _prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


INVALID_DECIMAL_FDI_RE = re.compile(r"(?<!\d)([1-4])\s*\.\s*(\d{1,2})(?!\d)")


def generation_quality_metrics(
    predictions: list[str], references: list[list[str]]
) -> tuple[dict, dict]:
    truncated_cases = invalid_cases = repetition_cases = 0
    invalid_count = repeated_sentences = repeatable_sentences = 0
    max_sentence_repeat = 0
    length_ratios = []
    for prediction, refs in zip(predictions, references):
        stripped = prediction.strip()
        if stripped and not re.search(r"[.!?]\s*$", stripped):
            truncated_cases += 1
        invalid = [
            match.group(0)
            for match in INVALID_DECIMAL_FDI_RE.finditer(prediction)
            if int(match.group(2)) not in range(1, 9)
        ]
        invalid_count += len(invalid)
        invalid_cases += bool(invalid)
        sentences = [
            normalize_report(sentence)
            for sentence in re.split(r"[.!?;\n]+", prediction)
            if len(normalize_report(sentence).split()) >= 4
        ]
        counts = Counter(sentences)
        repeated = sum(count - 1 for count in counts.values() if count > 1)
        repeated_sentences += repeated
        repeatable_sentences += len(sentences)
        repetition_cases += repeated > 0
        max_sentence_repeat = max(max_sentence_repeat, max(counts.values(), default=0))
        reference_length = statistics.mean(map(len, refs)) if refs else 0
        length_ratios.append(len(prediction) / max(reference_length, 1))
    size = len(predictions)
    scalars = {
        "terminal_punctuation_rate": (size - truncated_cases) / size if size else 0.0,
        "truncated_generation_rate": truncated_cases / size if size else 0.0,
        "invalid_fdi_count": float(invalid_count),
        "invalid_fdi_case_rate": invalid_cases / size if size else 0.0,
        "repetition_rate": repeated_sentences / repeatable_sentences
        if repeatable_sentences
        else 0.0,
        "repetition_case_rate": repetition_cases / size if size else 0.0,
        "max_sentence_repeat_count": float(max_sentence_repeat),
        "generation_reference_length_ratio_mean": statistics.mean(length_ratios)
        if length_ratios
        else 0.0,
    }
    details = {
        "truncated_cases": truncated_cases,
        "invalid_fdi_count": invalid_count,
        "invalid_fdi_cases": invalid_cases,
        "repetition_cases": repetition_cases,
        "repeated_sentences": repeated_sentences,
        "repeatable_sentences": repeatable_sentences,
        "length_ratios": length_ratios,
    }
    return scalars, details


def _near_groups(texts: list[str], threshold=0.85) -> list[list[int]]:
    parent = list(range(len(texts)))

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for left in range(len(texts)):
        for right in range(left + 1, len(texts)):
            if shingle_jaccard(texts[left], texts[right]) >= threshold:
                a, b = find(left), find(right)
                if a != b:
                    parent[max(a, b)] = min(a, b)
    groups = defaultdict(list)
    for index in range(len(texts)):
        groups[find(index)].append(index)
    return list(groups.values())


def clinical_metrics(
    predictions: list[str], references: list[list[str]], case_ids: list[str]
) -> tuple[dict, dict, list[dict]]:
    tooth_counts = defaultdict(Counter)
    side = Counter()
    contradiction = Counter()
    unsupported_counts, unsupported_concepts = [], Counter()
    case_specific_rows, concept_specific = [], defaultdict(Counter)
    case_details = []
    total_case_specific_tp = total_case_specific_pred = total_case_specific_ref = 0
    contradiction_cases = unsupported_cases = excluded_no_specific = 0
    positive_finding_candidates = 0
    for prediction, refs, case_id in zip(predictions, references, case_ids):
        predicted = extract_assertions(prediction)
        reference = set().union(*(extract_assertions(text) for text in refs))
        pred_tooth = {_tooth_key(item) for item in predicted if item.tooth_number}
        ref_tooth = {_tooth_key(item) for item in reference if item.tooth_number}
        for tooth in {value[0] for value in pred_tooth | ref_tooth}:
            p = {value for value in pred_tooth if value[0] == tooth}
            r = {value for value in ref_tooth if value[0] == tooth}
            tooth_counts[tooth].update(
                tp=len(p & r), fp=len(p - r), fn=len(r - p), support=len(r)
            )

        reference_by_neutral = defaultdict(set)
        for item in reference:
            if item.side:
                reference_by_neutral[_neutral_side_key(item)].add(item.side)
        for item in predicted:
            if not item.side or _neutral_side_key(item) not in reference_by_neutral:
                continue
            sides = reference_by_neutral[_neutral_side_key(item)]
            side["comparable"] += 1
            if item.side in sides:
                side["correct"] += 1
            elif item.side == "right":
                side["left_to_right"] += 1
            else:
                side["right_to_left"] += 1

        reference_core = defaultdict(set)
        for item in reference:
            reference_core[_assertion_key(item, False)].add(item.polarity)
        conflicts, unsupported = [], []
        for item in predicted:
            core = _assertion_key(item, False)
            if core in reference_core and item.polarity not in reference_core[core]:
                conflicts.append(item)
                contradiction["total"] += 1
                contradiction[
                    f"{item.polarity}_to_{next(iter(reference_core[core]))}"
                ] += 1
            elif item.polarity == "positive" and core not in reference_core:
                unsupported.append(item)
                unsupported_concepts[item.concept] += 1
        comparable = sum(
            _assertion_key(item, False) in reference_core for item in predicted
        )
        contradiction["comparable"] += comparable
        positive_finding_candidates += sum(
            item.polarity == "positive" and item not in conflicts for item in predicted
        )
        if conflicts:
            contradiction_cases += 1
        unsupported_counts.append(len(unsupported))
        if unsupported:
            unsupported_cases += 1

        reference_specific = {
            item for item in reference if item.clause_type == "case-specific"
        }
        predicted_specific = {
            item for item in predicted if item.clause_type == "case-specific"
        }
        predicted_keys = {_assertion_key(item) for item in predicted_specific}
        reference_keys = {_assertion_key(item) for item in reference_specific}
        matched_keys = predicted_keys & reference_keys
        matched = {
            item for item in reference_specific if _assertion_key(item) in matched_keys
        }
        total_case_specific_tp += len(matched_keys)
        total_case_specific_pred += len(predicted_keys)
        total_case_specific_ref += len(reference_keys)
        if reference_specific:
            case_specific_rows.append(
                _prf(
                    len(matched_keys),
                    len(predicted_keys - reference_keys),
                    len(reference_keys - predicted_keys),
                )
            )
        else:
            excluded_no_specific += 1
        for item in predicted_specific:
            concept_specific[item.concept]["predicted"] += 1
        for item in reference_specific:
            concept_specific[item.concept]["reference"] += 1
            if item in matched:
                concept_specific[item.concept]["matched"] += 1
        case_details.append(
            {
                "case_id": case_id,
                "prediction": prediction,
                "references": refs,
                "predicted_assertions": [item.to_dict() for item in sorted(predicted)],
                "reference_assertions": [item.to_dict() for item in sorted(reference)],
                "contradictions": [item.to_dict() for item in conflicts],
                "unsupported": [item.to_dict() for item in unsupported],
            }
        )

    all_counts = sum(tooth_counts.values(), Counter())
    per_tooth = {}
    macro_values = []
    for quadrant in "1234":
        for position in "12345678":
            tooth = quadrant + position
            counts = tooth_counts[tooth]
            values = _prf(counts["tp"], counts["fp"], counts["fn"])
            values["support"] = counts["support"]
            per_tooth[tooth] = values
            if counts["support"]:
                macro_values.append(values)
    micro = _prf(all_counts["tp"], all_counts["fp"], all_counts["fn"])
    macro = {
        key: sum(row[key] for row in macro_values) / len(macro_values)
        if macro_values
        else 0.0
        for key in ("precision", "recall", "f1")
    }
    case_micro = _prf(
        total_case_specific_tp,
        total_case_specific_pred - total_case_specific_tp,
        total_case_specific_ref - total_case_specific_tp,
    )
    case_macro = {
        key: sum(row[key] for row in case_specific_rows) / len(case_specific_rows)
        if case_specific_rows
        else 0.0
        for key in ("precision", "recall", "f1")
    }
    quality, quality_details = generation_quality_metrics(predictions, references)
    normals = [normalize_report(text) for text in predictions]
    exact = defaultdict(list)
    for case_id, normal in zip(case_ids, normals):
        exact[normal].append(case_id)
    near = _near_groups(predictions)
    scalars = {
        **report_metrics(predictions, references),
        "tooth_precision_micro": micro["precision"],
        "tooth_recall_micro": micro["recall"],
        "tooth_f1_micro": micro["f1"],
        "tooth_precision_macro": macro["precision"],
        "tooth_recall_macro": macro["recall"],
        "tooth_f1_macro": macro["f1"],
        "side_accuracy": side["correct"] / side["comparable"]
        if side["comparable"]
        else 0.0,
        "negation_contradiction_rate": contradiction["total"]
        / contradiction["comparable"]
        if contradiction["comparable"]
        else 0.0,
        "negation_contradiction_case_rate": contradiction_cases / len(case_ids)
        if case_ids
        else 0.0,
        "unsupported_count": float(sum(unsupported_counts)),
        "unsupported_case_rate": unsupported_cases / len(case_ids) if case_ids else 0.0,
        "unsupported_finding_rate": sum(unsupported_counts)
        / positive_finding_candidates
        if positive_finding_candidates
        else 0.0,
        "case_specific_precision_micro": case_micro["precision"],
        "case_specific_recall_micro": case_micro["recall"],
        "case_specific_f1_micro": case_micro["f1"],
        "case_specific_precision_macro": case_macro["precision"],
        "case_specific_recall_macro": case_macro["recall"],
        "case_specific_f1_macro": case_macro["f1"],
        **quality,
        "unique_rate": len(exact) / len(case_ids) if case_ids else 0.0,
        "max_duplicate_count": float(max(map(len, exact.values()), default=0)),
        "near_unique_rate": len(near) / len(case_ids) if case_ids else 0.0,
        "max_near_duplicate_count": float(max(map(len, near), default=0)),
        "empty_generation_rate": sum(not text.strip() for text in predictions)
        / len(predictions)
        if predictions
        else 0.0,
    }
    concept_scores = {}
    for concept, counts in concept_specific.items():
        values = _prf(
            counts["matched"],
            counts["predicted"] - counts["matched"],
            counts["reference"] - counts["matched"],
        )
        concept_scores[concept] = {**dict(counts), **values}

    details = {
        "tooth": {"micro": micro, "macro": macro, "per_tooth": per_tooth},
        "side": dict(side),
        "negation": dict(contradiction),
        "unsupported": {
            "per_case": unsupported_counts,
            "mean": statistics.mean(unsupported_counts) if unsupported_counts else 0,
            "median": statistics.median(unsupported_counts)
            if unsupported_counts
            else 0,
            "maximum": max(unsupported_counts, default=0),
            "by_concept": dict(unsupported_concepts),
            "positive_finding_candidates": positive_finding_candidates,
            "definition": "Not supported by the union of reference-report assertions; not image-ground-truth.",
        },
        "case_specific": {
            "excluded_without_reference_assertions": excluded_no_specific,
            "micro": case_micro,
            "macro": case_macro,
            "by_concept": concept_scores,
        },
        "generation_quality": quality_details,
        "duplicates": {
            "exact": [values for values in exact.values() if len(values) > 1],
            "near": [
                [case_ids[index] for index in group] for group in near if len(group) > 1
            ],
        },
    }
    return scalars, details, case_details
