from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable


FDI_RE = re.compile(r"(?<!\d)([1-4])\s*[.]?\s*([1-8])(?!\d)")
FDI_RANGE_RE = re.compile(
    r"(?<!\d)([1-4])\s*[.]?\s*([1-8])\s*"
    r"(?:-|–|—|\bto\b|\bthrough\b)\s*"
    r"(?:(?:([1-4])\s*[.]?\s*)?([1-8]))(?!\d)",
    re.I,
)
WORD_RE = re.compile(r"[a-z0-9]+")
NEGATION_RE = re.compile(
    r"\b(?:no|not|without|absent|absence of|no evidence of|no definite|negative for)\b",
    re.I,
)
COVERAGE_RE = re.compile(
    r"\b(?:scan|acquisition|field of view|volume|available scans?|included|excluded|"
    r"not visible|not assessable|non[- ]evaluable|partially visuali[sz]ed)\b",
    re.I,
)

CONCEPT_PATTERNS: dict[str, re.Pattern[str]] = {
    "missing": re.compile(
        r"\b(?:missing|absen(?:t|ce)|edentulous|not present)\b", re.I
    ),
    "impacted": re.compile(r"\b(?:impact(?:ed|ion)|unerupted|retained tooth)\b", re.I),
    "root_canal": re.compile(
        r"\b(?:root canal|endodontic(?:ally)?(?: treated| treatment)?|root[- ]filled)\b",
        re.I,
    ),
    "implant": re.compile(r"\b(?:implant|implant-supported)\b", re.I),
    "lesion": re.compile(
        r"\b(?:lesion|osteolytic|osteocondensing|radiolucen\w*|radiopaqu\w*|"
        r"periapical\s+(?:pathology|lesion)|osteorarefaction)\b",
        re.I,
    ),
    "bone_loss": re.compile(
        r"\b(?:bone (?:loss|atrophy)|alveolar atrophy|periodontal loss)\b", re.I
    ),
    "calculus": re.compile(
        r"\b(?:calculus|calcification|calcified deposit)\w*\b", re.I
    ),
    "caries": re.compile(r"\b(?:caries|carious|dental decay)\b", re.I),
    "prosthesis": re.compile(r"\b(?:prosthes\w*|restoration|crown|bridge)\b", re.I),
    "extraction_socket": re.compile(
        r"\b(?:extraction|post-extraction|extraction socket)\b", re.I
    ),
    "mandibular_canal": re.compile(r"\bmandibular canal\b", re.I),
    "sinus": re.compile(r"\b(?:maxillary )?sinus(?:es)?\b", re.I),
    "tmj": re.compile(
        r"\b(?:temporomandibular joint|TMJ|mandibular condyle)\w*\b", re.I
    ),
}

ANATOMY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("mandibular_canal", re.compile(r"\bmandibular canal\b", re.I)),
    ("maxillary_sinus", re.compile(r"\bmaxillary sinus(?:es)?\b", re.I)),
    (
        "tmj",
        re.compile(r"\b(?:temporomandibular joint|TMJ|mandibular condyle)\w*\b", re.I),
    ),
    ("mandible", re.compile(r"\bmandib(?:le|ular)\b", re.I)),
    ("maxilla", re.compile(r"\bmaxill(?:a|ary)\b", re.I)),
)
PRESENCE_PATTERN = re.compile(
    r"\b(?:present|presence of|identified|noted|demonstrated|shows?|contains?)\b",
    re.I,
)


@dataclass(frozen=True, order=True)
class FindingAssertion:
    concept: str
    tooth_number: str | None
    anatomy: str | None
    side: str | None
    polarity: str
    clause_type: str

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_report(text: str) -> str:
    return " ".join(WORD_RE.findall(text.lower())) or "<empty>"


def word_shingles(text: str, size: int = 5) -> set[tuple[str, ...]]:
    words = normalize_report(text).split()
    if len(words) < size:
        return {tuple(words)} if words else set()
    return {
        tuple(words[index : index + size]) for index in range(len(words) - size + 1)
    }


def shingle_jaccard(left: str, right: str, size: int = 5) -> float:
    a, b = word_shingles(left, size), word_shingles(right, size)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if a or b else 0.0


def iter_clauses(text: str) -> Iterable[tuple[int, int, str]]:
    start = 0
    for match in re.finditer(r"(?:\r?\n)+|(?<=[.!?;])\s+", text):
        end = match.start()
        if text[start:end].strip():
            yield start, end, text[start:end]
        start = match.end()
    if text[start:].strip():
        yield start, len(text), text[start:]


def is_coverage_clause(clause: str) -> bool:
    return bool(COVERAGE_RE.search(clause)) and bool(
        re.search(
            r"\b(?:included|excluded|visible|assessable|visuali[sz]ed|field of view|volume)\b",
            clause,
            re.I,
        )
    )


def tooth_side(tooth: str) -> str:
    return "right" if tooth[0] in {"1", "4"} else "left"


def tooth_group(tooth: str) -> str:
    jaw = "upper" if tooth[0] in {"1", "2"} else "lower"
    side = tooth_side(tooth)
    position = int(tooth[1])
    kind = "anterior" if position <= 3 else "premolar" if position <= 5 else "molar"
    return f"tooth_group:{jaw}_{side}_{kind}"


def _teeth(clause: str) -> list[str]:
    teeth = {f"{quadrant}{position}" for quadrant, position in FDI_RE.findall(clause)}
    for match in FDI_RANGE_RE.finditer(clause):
        left_quadrant, left_position, right_quadrant, right_position = match.groups()
        right_quadrant = right_quadrant or left_quadrant
        if left_quadrant == right_quadrant:
            begin, end = sorted((int(left_position), int(right_position)))
            teeth.update(
                f"{left_quadrant}{position}" for position in range(begin, end + 1)
            )
    return sorted(teeth)


def _anchor_windows(clause: str) -> list[tuple[str, re.Match[str], str]]:
    anchors: list[tuple[int, int, str | None, re.Match[str]]] = []
    for concept, pattern in CONCEPT_PATTERNS.items():
        for match in pattern.finditer(clause):
            anchors.append((match.start(), match.end(), concept, match))
    for match in PRESENCE_PATTERN.finditer(clause):
        anchors.append((match.start(), match.end(), None, match))
    anchors.sort(key=lambda item: (item[0], item[1], item[2] is None))
    filtered = []
    for anchor in anchors:
        if anchor[2] is None and any(
            other[2] is not None and anchor[0] < other[1] and anchor[1] > other[0]
            for other in anchors
        ):
            continue
        if (
            filtered
            and anchor[2] is not None
            and filtered[-1][2] == anchor[2]
            and anchor[0] - filtered[-1][1] <= 2
        ):
            continue
        if not filtered or anchor[:3] != filtered[-1][:3]:
            filtered.append(anchor)
    centers = [(start + end) / 2 for start, end, _concept, _match in filtered]
    result = []
    for index, (_start, _end, concept, match) in enumerate(filtered):
        if concept is None:
            continue
        left = 0 if index == 0 else int((centers[index - 1] + centers[index]) / 2)
        right = (
            len(clause)
            if index + 1 == len(filtered)
            else int((centers[index] + centers[index + 1]) / 2)
        )
        result.append((concept, match, clause[left:right]))
    return result


def _side(clause: str, tooth: str | None) -> str | None:
    explicit = {side.lower() for side in re.findall(r"\b(left|right)\b", clause, re.I)}
    if len(explicit) == 1:
        return explicit.pop()
    return tooth_side(tooth) if tooth else None


def _anatomy(clause: str) -> str | None:
    for name, pattern in ANATOMY_PATTERNS:
        if pattern.search(clause):
            return name
    return None


def _polarity(concept: str, scope: str, coverage: bool) -> str:
    if coverage:
        return "positive"
    if concept == "missing":
        explicitly_negated = re.search(
            r"\b(?:no|without)\s+(?:evidence of\s+)?(?:missing|absent)\b",
            scope,
            re.I,
        )
        return "negative" if explicitly_negated else "positive"
    return "negative" if NEGATION_RE.search(scope) else "positive"


def extract_assertions(text: str) -> set[FindingAssertion]:
    assertions: set[FindingAssertion] = set()
    for _start, _end, clause in iter_clauses(text):
        for concept, _match, scope in _anchor_windows(clause):
            coverage = is_coverage_clause(scope)
            teeth = _teeth(scope)
            anatomy = _anatomy(scope) or _anatomy(clause)
            polarity = _polarity(concept, scope, coverage)
            targets: list[str | None] = teeth or [None]
            for tooth in targets:
                assertions.add(
                    FindingAssertion(
                        concept=concept,
                        tooth_number=tooth,
                        anatomy=anatomy if tooth is None else None,
                        side=_side(scope, tooth),
                        polarity=polarity,
                        clause_type="coverage" if coverage else "case-specific",
                    )
                )
    return assertions


def multilabels(texts: Iterable[str]) -> set[str]:
    labels: set[str] = set()
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
    labels: set[str] = set()
    for text in texts:
        for item in extract_assertions(text):
            if item.polarity != "positive":
                continue
            if item.concept in {"missing", "impacted", "root_canal", "implant"}:
                labels.add(f"finding:{item.concept}")
            if item.tooth_number:
                labels.add(f"tooth:{item.tooth_number}")
    return labels


def token_clause_weights(
    text: str, offsets: list[tuple[int, int]], coverage: float, finding: float
) -> list[float]:
    clauses = list(iter_clauses(text))
    result: list[float] = []
    for begin, end in offsets:
        weight = 1.0
        if begin != end:
            for clause_begin, clause_end, clause in clauses:
                if begin < clause_end and end > clause_begin:
                    weight = coverage if is_coverage_clause(clause) else finding
                    break
        result.append(weight)
    return result
