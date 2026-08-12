from lightning.pytorch.loggers import CSVLogger

from runtime import build_loggers


def test_local_csv_logger_is_always_enabled(tmp_path):
    cfg = {
        "Logger": {"use_wandb": False, "project": "unused"},
        "Optimizer": {"learning_rate": 1e-5},
        "Scheduler": {},
    }

    loggers = build_loggers(cfg, tmp_path, debug=False)

    assert len(loggers) == 1
    assert isinstance(loggers[0], CSVLogger)
    assert loggers[0].save_dir == str(tmp_path)
    assert loggers[0].name == "logs"
    assert loggers[0]._flush_logs_every_n_steps == 1


def test_wandb_is_added_without_replacing_local_csv(tmp_path, monkeypatch):
    import lightning.pytorch.loggers as logger_module

    class FakeWandbLogger:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.hyperparams = None

        def log_hyperparams(self, values):
            self.hyperparams = values

    monkeypatch.setattr(logger_module, "WandbLogger", FakeWandbLogger)
    cfg = {
        "Logger": {"use_wandb": True, "project": "test-project"},
        "Optimizer": {"learning_rate": 1e-5},
        "Scheduler": {
            "name": "CosineAnnealingWithWarmupLR",
            "start_learning_rate": 1e-7,
            "end_learning_rate": 1e-7,
        },
    }

    loggers = build_loggers(cfg, tmp_path, debug=True)

    assert len(loggers) == 2
    assert isinstance(loggers[0], CSVLogger)
    assert isinstance(loggers[1], FakeWandbLogger)
    assert loggers[1].kwargs == {
        "project": "test-project",
        "name": tmp_path.name,
        "save_dir": tmp_path,
        "offline": True,
    }
    assert loggers[1].hyperparams["learning_rate"] == 1e-5
