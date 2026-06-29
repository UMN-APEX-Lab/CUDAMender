import torch
import os

os.makedirs("data", exist_ok=True)

B = 16
M = 512
K = 256
N = 128

seed = 42
torch.manual_seed(seed)

A = torch.randn(B, M, K, dtype=torch.float32)
B_mat = torch.randn(B, K, N, dtype=torch.float32)
C = torch.bmm(A, B_mat)

A.numpy().tofile("data/batchA.bin")
B_mat.numpy().tofile("data/batchB.bin")
C.numpy().tofile("data/batchC_ref.bin")

