from __future__ import annotations

import balanced_metrics
import balanced_lightning_module
from clinical_fixed import extract_assertions

# Patch the shared evaluator with the clinically correct missing-tooth polarity before training starts.
balanced_metrics.extract_assertions = extract_assertions
balanced_lightning_module.clinical_metrics = balanced_metrics.clinical_metrics

from balanced_main import main  # noqa: E402


if __name__ == "__main__":
    main()
