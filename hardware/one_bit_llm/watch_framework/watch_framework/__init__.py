"""Watch Framework: 1-Bit CSA 2D Watch Grid PyTorch Engine and QAT Framework.

Usage:
    import watch_framework as wf

    layer = wf.nn.WatchLinear(in_features=128, out_features=64, bias=True)
    conv = wf.nn.WatchConv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)

    wf.save_checkpoint(model, "checkpoint.wfbin")
    wf.load_checkpoint(model, "checkpoint.wfbin")

    trainer = wf.Trainer(model, optimizer, criterion)
    trainer.fit(train_loader, val_loader, epochs=10, warmup_epochs=1)
"""

__version__ = "0.1.0"

from . import nn
from . import io
from . import trainer

from .io.checkpoint import save_checkpoint, load_checkpoint
from .trainer.engine import Trainer

__all__ = [
    "__version__",
    "nn",
    "io",
    "trainer",
    "save_checkpoint",
    "load_checkpoint",
    "Trainer",
]
