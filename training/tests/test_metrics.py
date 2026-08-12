from metrics import compute_report_metrics


def test_metrics_identical_and_empty():
    result = compute_report_metrics(["a b c d"], ["a b c d"])
    assert result["bleu4"] == 1.0
    assert result["meteor"] > 0.99
    assert compute_report_metrics([], []) == {"bleu4": 0.0, "meteor": 0.0}
