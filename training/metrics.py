from __future__ import annotations

import difflib

from nltk.stem.porter import PorterStemmer
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import meteor_score


class _NoWordNet:
    """METEOR's exact/stem matching without requiring an NLTK corpus download."""

    @staticmethod
    def synsets(_word):
        return []


def tokenize_metric_text(text: str) -> list[str]:
    return text.lower().strip().split()


def sequence_similarity(left: str, right: str) -> float:
    return difflib.SequenceMatcher(None, left.strip(), right.strip()).ratio()


def compute_report_metrics(
    predictions: list[str], references: list[str]
) -> dict[str, float]:
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have equal length")
    if not predictions:
        return {"bleu4": 0.0, "meteor": 0.0}
    hypotheses = [tokenize_metric_text(x) for x in predictions]
    refs = [[tokenize_metric_text(x)] for x in references]
    bleu = corpus_bleu(refs, hypotheses, smoothing_function=SmoothingFunction().method1)
    stemmer = PorterStemmer()
    meteor = sum(
        meteor_score(ref, hyp, stemmer=stemmer, wordnet=_NoWordNet())
        for ref, hyp in zip(refs, hypotheses)
    ) / len(hypotheses)
    return {"bleu4": float(bleu), "meteor": float(meteor)}


def compute_overfit_metrics(
    predictions: list[str],
    references: list[str],
    swapped_predictions: list[str] | None = None,
    swapped_source_predictions: list[str] | None = None,
) -> dict[str, float]:
    """Metrics that distinguish memorization from use of the image prefix."""
    metrics = compute_report_metrics(predictions, references)
    if not predictions:
        return {
            **metrics,
            "exact_match": 0.0,
            "character_similarity": 0.0,
        }
    metrics.update(
        {
            "exact_match": sum(
                prediction.strip() == reference.strip()
                for prediction, reference in zip(predictions, references)
            )
            / len(predictions),
            "character_similarity": sum(
                sequence_similarity(prediction, reference)
                for prediction, reference in zip(predictions, references)
            )
            / len(predictions),
        }
    )
    if swapped_predictions is None:
        return metrics
    if len(swapped_predictions) != len(predictions):
        raise ValueError("swapped predictions must have the same length as predictions")
    swapped_report_metrics = compute_report_metrics(swapped_predictions, references)
    metrics.update(
        {
            "swapped_bleu4": swapped_report_metrics["bleu4"],
            "swapped_meteor": swapped_report_metrics["meteor"],
            "image_swap_change_rate": sum(
                normal.strip() != swapped.strip()
                for normal, swapped in zip(predictions, swapped_predictions)
            )
            / len(predictions),
            "swap_stays_case_similarity": sum(
                sequence_similarity(normal, swapped)
                for normal, swapped in zip(predictions, swapped_predictions)
            )
            / len(predictions),
        }
    )
    if swapped_source_predictions is not None:
        if len(swapped_source_predictions) != len(predictions):
            raise ValueError("swap sources must have the same length as predictions")
        metrics["swap_follows_image_similarity"] = sum(
            sequence_similarity(source, swapped)
            for source, swapped in zip(swapped_source_predictions, swapped_predictions)
        ) / len(predictions)
    return metrics
