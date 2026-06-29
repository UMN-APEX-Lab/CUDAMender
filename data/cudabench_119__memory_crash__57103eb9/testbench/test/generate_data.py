import torch
import os

M = 2**18
K = 4096
N = 2048

seed = 42
torch.manual_seed(seed)

A = torch.randn(M, K, dtype=torch.float32)
B = torch.randn(K, N, dtype=torch.float32)
C = A @ B

A.numpy().tofile("data/matA.bin")
B.numpy().tofile("data/matB.bin")
C.numpy().tofile("data/matC_ref.bin")

