import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_cpp_source = r"""
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>

torch::Tensor conv_transpose3d_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups) {
  c10::optional<torch::Tensor> bias_opt = c10::nullopt;
  if (bias.has_value()) {
    bias_opt = bias.value();
  }
  std::vector<int64_t> dilation = {1, 1, 1};
  return at::conv_transpose3d(
      input,
      weight,
      bias_opt,
      stride,
      padding,
      output_padding,
      groups,
      dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("conv_transpose3d_forward", &conv_transpose3d_forward, "ConvTranspose3d forward");
}
"""

_conv_transpose3d_ext = load_inline(
    name="conv_transpose3d_ext_v1",
    cpp_sources=_cpp_source,
    cuda_sources="",
    functions=None,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1, 1),
        padding: tuple = (0, 0, 0),
        output_padding: tuple = (0, 0, 0),
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )
        self._op = _conv_transpose3d_ext

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._op.conv_transpose3d_forward(
            x,
            self.conv_transpose3d.weight,
            self.conv_transpose3d.bias,
            list(self.conv_transpose3d.stride),
            list(self.conv_transpose3d.padding),
            list(self.conv_transpose3d.output_padding),
            self.conv_transpose3d.groups,
        )