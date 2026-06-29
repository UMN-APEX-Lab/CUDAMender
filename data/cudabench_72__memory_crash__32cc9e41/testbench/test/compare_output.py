import numpy as np
import os

B = 16
M = 512
K = 256
N = 128

size = B * M * N

def read_binary(file, size):
    return np.fromfile(file, dtype=np.float32, count=size)

out_file = "data/batchC_out.bin"
ref_file = "data/batchC_ref.bin"

if not os.path.exists(out_file) or not os.path.exists(ref_file):
    print("F")
    exit(0)

out = read_binary(out_file, size)
ref = read_binary(ref_file, size)

if np.allclose(out, ref, atol=1e-2):
    print("T")
else:
    print("F")

