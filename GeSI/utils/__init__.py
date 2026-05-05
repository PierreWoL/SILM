import torch
from .data import batch
from .jinja import load_template
from .logging import log_flags, setup_logging
from .textqdm import textpbar, textqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

__all__ = [
    "batch",
    "device",
    "load_template",
    "log_flags",
    "setup_logging",
    "textpbar",
    "textqdm",
]