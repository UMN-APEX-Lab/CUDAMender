import numpy as np
import os

def read_binary(filename, size):
    return np.fromfile(filename, dtype=np.float32, count=size)

def compare_outputs(output_file, ref_file, size, tolerance=1e-2):
    if not os.path.exists(output_file) or not os.path.exists(ref_file):
        return False
    output = read_binary(output_file, size)
    reference = read_binary(ref_file, size)
    if output.shape != reference.shape:
        return False
    diff = np.abs(output - reference)
    return np.all(diff < tolerance)

if __name__ == "__main__":
    K = 4096
    N = 2048
    M = 262144

    size_C = M * N
    out_file = "./data/matC_out.bin"
    ref_file = "./data/matC_ref.bin"

    if compare_outputs(out_file, ref_file, size_C):
        print("T")
    else:
        print("F")

