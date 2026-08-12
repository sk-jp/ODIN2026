from __future__ import annotations

from typing import Iterable

import clinical as _base
from clinical import *  # noqa: F401,F403


def extract_assertions(text: str) -> set[FindingAssertion]:
    return _base.extract_assertions(text)


def multilabels(texts: Iterable[str]) -> set[str]:
    labels = set()
    for text in texts:
        for item in extract_assertions(text):
            if item.polarity != "positive":
                continue
            if item.concept in {"missing", "impacted", "root_canal", "implant"}:
                labels.add(f"finding:{item.concept}")
            if item.tooth_number:
                labels.add(tooth_group(item.tooth_number))
    return labels


def weighting_labels(texts: Iterable[str]) -> set[str]:
    labels = set()
    for text in texts:
        for item in extract_assertions(text):
            if item.polarity != "positive":
                continue
            if item.concept in {"missing", "impacted", "root_canal", "implant"}:
                labels.add(f"finding:{item.concept}")
            if item.tooth_number:
                labels.add(f"tooth:{item.tooth_number}")
    return labels
