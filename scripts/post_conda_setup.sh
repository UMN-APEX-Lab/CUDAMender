#!/usr/bin/env bash
# After `conda env create -f environment.yml && conda activate cuda-debugger-bench`,
# run this once to register the vendored kernelbench package.
set -e

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SITE_PACKAGES="$(python3 -c 'import site; print(site.getsitepackages()[0])')"

echo "$CODE_ROOT/harness/_vendor" > "$SITE_PACKAGES/applied_kernels_vendored.pth"

echo "Registered vendored kernelbench at $SITE_PACKAGES/applied_kernels_vendored.pth"
echo
echo "Sanity check:"
python3 -c "import kernelbench; print('  kernelbench:', kernelbench.__file__)"
echo
echo "Next: bash scripts/prepare_benchmark.sh"
