"""Template-only custom-kernel example used by KernelBench prompt assets.

The original upstream file contained pseudo-code placeholders that were not
valid Python. This version keeps the file illustrative while remaining safe to
import and compile.
"""

import torch
import torch.nn as nn


class _PlaceholderKernel:
    def __call__(self, x, *args, **kwargs):
        return x


custom_kernel_name_1 = _PlaceholderKernel()
custom_kernel_name_2 = _PlaceholderKernel()
custom_kernel_name_3 = _PlaceholderKernel()


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.op1 = custom_kernel_name_3
        self.op2 = nn.Identity()
        self.op34 = custom_kernel_name_1
        self.op56 = custom_kernel_name_2

    def forward(self, x):
        x = self.op1(x)   # placeholder: custom kernel 3
        x = self.op2(x)   # placeholder: standard torch op
        x = self.op34(x)  # placeholder: custom kernel 1
        x = self.op56(x)  # placeholder: custom kernel 2
        return x
