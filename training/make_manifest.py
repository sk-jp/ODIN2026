from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from clinical import multilabels, normalize_report, shingle_jaccard, weighting_labels


class UnionFind:
    def __init__(self, values):
        self.parent = {value: value for value in values}

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def discover(data_root: Path, feature_root: Path) -> list[dict]:
    result = []
    for directory in sorted((data_root / "cases").iterdir()):
        if not directory.is_dir():
            continue
        paths = sorted((directory / "reports_en").glob("*.txt"))
        feature = feature_root / directory.name / "features.pt"
        if not paths or not feature.is_file():
            raise FileNotFoundError(f"Incomplete case: {directory.name}")
        texts = [path.read_text(encoding="utf-8").strip() for path in paths]
        if any(not text for text in texts):
            raise ValueError(f"Empty report: {directory.name}")
        result.append(
            {
                "case_id": directory.name,
                "feature_path": str(feature.resolve()),
                "report_paths": [str(path.resolve()) for path in paths],
                "texts": texts,
                "labels": sorted(multilabels(texts)),
                "weighting_labels": sorted(weighting_labels(texts)),
            }
        )
    return result


def group_duplicates(cases: list[dict], threshold: float) -> tuple[dict, dict]:
    ids = [case["case_id"] for case in cases]
    duplicate, exact = UnionFind(ids), UnionFind(ids)
    reports, by_normal = [], defaultdict(list)
    for case in cases:
        for text in case["texts"]:
            normal = normalize_report(text)
            reports.append((case["case_id"], text, normal))
            by_normal[normal].append(case["case_id"])
    for members in by_normal.values():
        members = sorted(set(members))
        for member in members[1:]:
            duplicate.union(members[0], member)
            exact.union(members[0], member)
    for index, (left_id, left, left_normal) in enumerate(reports):
        for right_id, right, right_normal in reports[index + 1 :]:
            if (
                left_id != right_id
                and left_normal != right_normal
                and shingle_jaccard(left, right) >= threshold
            ):
                duplicate.union(left_id, right_id)
    return (
        {key: duplicate.find(key) for key in ids},
        {key: exact.find(key) for key in ids},
    )


def multilabel_split(cases: list[dict], fraction: float, seed: int) -> set[str]:
    groups = defaultdict(list)
    for case in cases:
        groups[case["duplicate_group"]].append(case)
    components, totals = [], Counter()
    for group_id, members in groups.items():
        counts = Counter(label for case in members for label in set(case["labels"]))
        components.append((group_id, members, counts))
        totals.update(counts)
    target_size, targets = (
        round(len(cases) * fraction),
        {key: value * fraction for key, value in totals.items()},
    )
    rng = random.Random(seed)
    tie = {item[0]: rng.random() for item in components}
    selected, counts, size, remaining = set(), Counter(), 0, list(components)
    while remaining and size < target_size:

        def merit(item):
            group_id, members, labels = item
            gain = sum(
                min(value, max(targets[key] - counts[key], 0)) / max(targets[key], 1)
                for key, value in labels.items()
            )
            overshoot = max(size + len(members) - target_size, 0) / max(target_size, 1)
            return (
                gain / len(members) - overshoot,
                -abs(target_size - size - len(members)),
                tie[group_id],
            )

        item = max(remaining, key=merit)
        remaining.remove(item)
        group_id, members, labels = item
        if (
            size
            and size + len(members) > target_size
            and target_size - size < size + len(members) - target_size
        ):
            break
        selected.add(group_id)
        size += len(members)
        counts.update(labels)
    return selected


def effective_weights(
    cases: list[dict], beta: float, low: float, high: float
) -> dict[str, float]:
    frequency = Counter(
        label for case in cases for label in set(case["weighting_labels"])
    )
    class_weight = {
        key: (1 - beta) / (1 - beta**value) for key, value in frequency.items()
    }
    weights = {}
    for case in cases:
        values = [class_weight[label] for label in case["weighting_labels"]]
        weights[case["case_id"]] = sum(values) / len(values) if values else 1.0
    for _ in range(12):
        mean = sum(weights.values()) / len(weights)
        weights = {
            key: min(max(value / mean, low), high) for key, value in weights.items()
        }
    return weights


def write_manifest(path: Path, cases: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for case in sorted(cases, key=lambda row: row["case_id"]):
            handle.write(
                json.dumps(
                    {key: value for key, value in case.items() if key != "texts"},
                    sort_keys=True,
                )
                + "\n"
            )


def build(args) -> dict:
    cases = discover(args.data_root, args.feature_root)
    duplicate, exact = group_duplicates(cases, args.near_threshold)
    for case in cases:
        case["duplicate_group"], case["template_group"] = (
            duplicate[case["case_id"]],
            exact[case["case_id"]],
        )
    valid_groups = multilabel_split(cases, args.valid_fraction, args.seed)
    train = [case for case in cases if case["duplicate_group"] not in valid_groups]
    valid = [case for case in cases if case["duplicate_group"] in valid_groups]
    weights = effective_weights(train, args.beta, args.min_weight, args.max_weight)
    for case in train:
        case["sampling_weight"] = weights[case["case_id"]]
    for case in valid:
        case["sampling_weight"] = 1.0
    if {x["duplicate_group"] for x in train} & {x["duplicate_group"] for x in valid}:
        raise RuntimeError("duplicate leakage")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(args.output_dir / "train_manifest.jsonl", train)
    write_manifest(args.output_dir / "valid_manifest.jsonl", valid)
    audit = {
        "cases": len(cases),
        "reports": sum(len(x["report_paths"]) for x in cases),
        "unused_reports": 0,
        "train_cases": len(train),
        "valid_cases": len(valid),
        "valid_fraction": len(valid) / len(cases),
        "cross_split_duplicate_groups": 0,
        "train_labels": dict(
            sorted(Counter(y for x in train for y in x["labels"]).items())
        ),
        "valid_labels": dict(
            sorted(Counter(y for x in valid for y in x["labels"]).items())
        ),
        "weight_min": min(weights.values()),
        "weight_max": max(weights.values()),
        "weight_mean": sum(weights.values()) / len(weights),
    }
    (args.output_dir / "manifest_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    return audit


def arguments():
    parser = argparse.ArgumentParser(
        description="Create leakage-aware train/validation manifests"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).parent / "datalist"
    )
    parser.add_argument("--valid-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--near-threshold", type=float, default=0.85)
    parser.add_argument("--beta", type=float, default=0.99)
    parser.add_argument("--min-weight", type=float, default=0.25)
    parser.add_argument("--max-weight", type=float, default=4.0)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(arguments()), indent=2))
