import pytest
import torch

from cosine_annealing_with_warmup_lr import CosineAnnealingWithWarmupLR
from scheduler import build_lr_scheduler


def make_optimizer(lr=0.1):
    return torch.optim.SGD([torch.nn.Parameter(torch.tensor(1.0))], lr=lr)


def test_custom_scheduler_uses_configured_start_and_end_learning_rates():
    optimizer = make_optimizer()
    config = build_lr_scheduler(
        optimizer,
        {
            "name": "CosineAnnealingWithWarmupLR",
            "warmup_ratio": 0.2,
            "start_learning_rate": 0.01,
            "end_learning_rate": 0.001,
        },
        total_steps=10,
        max_epochs=2,
    )
    scheduler = config["scheduler"]

    assert isinstance(scheduler, CosineAnnealingWithWarmupLR)
    assert config["interval"] == "step"
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)

    for _ in range(10):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.001)


def test_custom_scheduler_accepts_scientific_notation_loaded_as_strings():
    optimizer = make_optimizer()
    config = build_lr_scheduler(
        optimizer,
        {
            "name": "CosineAnnealingWithWarmupLR",
            "warmup_ratio": 0.1,
            "start_learning_rate": "1e-7",
            "end_learning_rate": "1e-7",
        },
        total_steps=10,
        max_epochs=2,
    )
    scheduler = config["scheduler"]

    assert scheduler.warmup_start_lr == pytest.approx(1e-7)
    assert scheduler.eta_min == pytest.approx(1e-7)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-7)


def test_custom_scheduler_is_cosine_after_warmup():
    optimizer = make_optimizer(lr=0.1)
    scheduler = build_lr_scheduler(
        optimizer,
        {
            "name": "CosineAnnealingWithWarmupLR",
            "warmup_ratio": 0.1,
            "start_learning_rate": 0.0,
            "end_learning_rate": 0.0,
        },
        total_steps=100,
        max_epochs=10,
    )["scheduler"]

    learning_rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(100):
        optimizer.step()
        scheduler.step()
        learning_rates.append(optimizer.param_groups[0]["lr"])

    assert learning_rates[0] == pytest.approx(0.0)
    assert learning_rates[10] == pytest.approx(0.1)
    # Halfway through the 90-step cosine phase, LR is half the peak LR.
    assert learning_rates[55] == pytest.approx(0.05)
    assert learning_rates[100] == pytest.approx(0.0)


def test_pytorch_scheduler_can_be_selected_by_class_name():
    optimizer = make_optimizer()
    config = build_lr_scheduler(
        optimizer,
        {"name": "StepLR", "step_size": 2, "gamma": 0.1},
        total_steps=10,
        max_epochs=2,
    )

    assert isinstance(config["scheduler"], torch.optim.lr_scheduler.StepLR)
    assert config["interval"] == "epoch"


def test_pytorch_scheduler_with_defaults_needs_only_its_name():
    optimizer = make_optimizer()
    config = build_lr_scheduler(
        optimizer,
        {"name": "ConstantLR"},
        total_steps=10,
        max_epochs=2,
    )

    assert isinstance(config["scheduler"], torch.optim.lr_scheduler.ConstantLR)


def test_unknown_scheduler_name_is_rejected():
    with pytest.raises(ValueError, match="Unknown learning-rate scheduler"):
        build_lr_scheduler(
            make_optimizer(), {"name": "NotAScheduler"}, total_steps=10, max_epochs=2
        )
