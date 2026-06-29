"""Template-only architecture example used in prompt construction tests.

This file is intentionally valid Python so package validation and compileall do
not fail, while still preserving obvious placeholders for prompt authors.
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Replace these placeholders with real operators for a concrete example.
        self.op1 = nn.Identity()
        self.op2 = nn.Identity()
        self.op3 = nn.Identity()
        self.op4 = nn.Identity()
        self.op5 = nn.Identity()
        self.op6 = nn.Identity()

    def forward(self, x):
        x = self.op1(x)  # placeholder: op1(x, ...)
        x = self.op2(x)  # placeholder: op2(x, ...)
        x = self.op3(x)  # placeholder: op3(x, ...)
        x = self.op4(x)  # placeholder: op4(x, ...)
        x = self.op5(x)  # placeholder: op5(x, ...)
        x = self.op6(x)  # placeholder: op6(x, ...)
        return x


def get_inputs():
    return [torch.randn(1, 128, device="cuda")]


def get_init_inputs():
    return []
