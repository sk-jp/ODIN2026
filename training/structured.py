from __future__ import annotations

from collections.abc import Iterable

import torch

from clinical import extract_assertions


FDI_TEETH = tuple(
    f"{quadrant}{position}" for quadrant in "1234" for position in "12345678"
)
TOOTH_CONCEPTS = (
    "missing",
    "impacted",
    "root_canal",
    "implant",
    "prosthesis",
    "caries",
)
CASE_CONCEPTS = (
    "lesion",
    "bone_loss",
    "calculus",
    "extraction_socket",
    "mandibular_canal",
    "sinus",
    "tmj",
)
TOOTH_INDEX = {tooth: index for index, tooth in enumerate(FDI_TEETH)}
TOOTH_CONCEPT_INDEX = {concept: index for index, concept in enumerate(TOOTH_CONCEPTS)}
CASE_CONCEPT_INDEX = {concept: index for index, concept in enumerate(CASE_CONCEPTS)}


def structured_targets(texts: Iterable[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Build union-of-references positive targets for the auxiliary heads."""
    tooth = torch.zeros(len(FDI_TEETH), len(TOOTH_CONCEPTS), dtype=torch.float32)
    case = torch.zeros(len(CASE_CONCEPTS), dtype=torch.float32)
    for text in texts:
        for finding in extract_assertions(text):
            if finding.polarity != "positive" or finding.clause_type != "case-specific":
                continue
            if (
                finding.tooth_number in TOOTH_INDEX
                and finding.concept in TOOTH_CONCEPT_INDEX
            ):
                tooth[
                    TOOTH_INDEX[finding.tooth_number],
                    TOOTH_CONCEPT_INDEX[finding.concept],
                ] = 1.0
            if finding.concept in CASE_CONCEPT_INDEX:
                case[CASE_CONCEPT_INDEX[finding.concept]] = 1.0
    return tooth, case
