import torch
import os

M = 147456
K = 64
N = 256

seed = int.from_bytes(os.urandom(4), 'little')
torch.manual_seed(seed)

A = torch.randn(M, K, dtype=torch.float32)
B = torch.randn(K, N, dtype=torch.float32)
X = A.view(-1).view(1, 64, 384, 384)
Wt = torch.randn(64, 64, 2, 2, dtype=torch.float32)
Y = torch.nn.functional.conv_transpose2d(X, Wt, bias=None, stride=(2,2), padding=(0,0), output_padding=(0,0))
B.view(-1)[:Wt.numel()] = Wt.view(-1)
C = Y.contiguous().view(M, N)

os.makedirs("data", exist_ok=True)
A.numpy().tofile("data/matA.bin")
B.numpy().tofile("data/matB.bin")
C.numpy().tofile("data/matC_ref.bin")

