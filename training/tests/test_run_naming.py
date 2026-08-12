from argparse import Namespace

from runtime import run_condition_suffix


def args(lr=None):
    return Namespace(lr=lr)


def test_balanced_run_name_uses_all_effective_learning_rate_conditions():
    cfg = {
        "Optimizer": {
            "learning_rate": 1e-5,
            "projector_learning_rate": 1e-4,
            "lora_learning_rate": 1e-5,
            "structured_learning_rate": 2e-4,
            "projector_only_epochs": 2,
        },
        "StructuredFindings": {"enabled": True},
    }

    assert run_condition_suffix(cfg, args()) == "-PLR0.0001-LLR1e-05-SLR0.0002-STAGE2"


def test_run_name_uses_resolved_values_after_global_and_individual_overrides():
    cfg = {
        "Optimizer": {
            "learning_rate": 3e-5,
            "projector_learning_rate": 3e-4,
            "lora_learning_rate": 2e-5,
            "structured_learning_rate": 4e-4,
            "projector_only_epochs": 3,
        },
        "StructuredFindings": {"enabled": True},
    }

    assert (
        run_condition_suffix(cfg, args(3e-5)) == "-PLR0.0003-LLR2e-05-SLR0.0004-STAGE3"
    )


def test_legacy_single_lr_name_is_preserved():
    cfg = {"Optimizer": {"learning_rate": 3e-5}}

    assert run_condition_suffix(cfg, args(3e-5)) == "-LR3e-05"
