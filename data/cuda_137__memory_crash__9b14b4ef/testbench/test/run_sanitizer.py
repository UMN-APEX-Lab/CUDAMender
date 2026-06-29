#!/usr/bin/env python3
import re
import subprocess
import sys
cmd = sys.argv[1:] or ['./test.out']
proc = subprocess.run(['compute-sanitizer','--tool','memcheck','--leak-check','no','--target-processes','all','--error-exitcode','99',*cmd], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
out = (proc.stdout or '') + (proc.stderr or '')
print(out, end='')
low = out.lower()
mem_patterns = ['invalid __global__ read','invalid __global__ write','invalid __shared__ read','invalid __shared__ write','invalid __local__ read','invalid __local__ write','illegal memory access','cudaerrorillegaladdress','an illegal memory access was encountered','misaligned address','out of bounds','invalid address','invalid pc']
if any(p in low for p in mem_patterns):
    print('illegal memory access detected by compute-sanitizer')
    sys.exit(99)
sys.exit(proc.returncode)
