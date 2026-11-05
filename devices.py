"""Where the models run: an NVIDIA GPU (CUDA), else an Intel GPU (XPU: Arc cards, Core Ultra's built-in Arc
graphics), else the CPU. install.ps1 installs the matching PyTorch build.

An Intel GPU is used only if a small test calculation on it works; if it doesn't (an older chip, a missing driver),
Sable runs on the CPU as before.
"""
import functools

import torch


@functools.cache
def best() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        try:
            x = torch.ones(8, 8, device="xpu", dtype=torch.bfloat16)
            if float((x @ x).sum().cpu()) == 512.0:
                return "xpu"
        except Exception:
            pass
    return "cpu"


def name() -> str:
    d = best()
    if d == "cuda":
        return torch.cuda.get_device_name(0)
    if d == "xpu":
        return torch.xpu.get_device_name(0)
    return "CPU"


def empty_cache():
    if best() == "cuda":
        torch.cuda.empty_cache()
    elif best() == "xpu":
        torch.xpu.empty_cache()
