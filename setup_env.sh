#!/usr/bin/env bash
# Set up Python venv with pinned requirements + register vendored packages.
#
# After this completes, activate with: `source .venv/bin/activate`
# and re-run prepare_benchmark.sh to materialize symlinks.
set -e

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 -m venv .venv
# shellcheck source=/dev/null
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Register vendored packages (KernelBench upstream MIT-licensed pkg) into the
# venv's site-packages via a .pth file. This makes `import kernelbench` work
# after activation without editing PYTHONPATH manually.
SITE_PACKAGES="$(python3 -c 'import site; print(site.getsitepackages()[0])')"
echo "$CODE_ROOT/harness/_vendor" > "$SITE_PACKAGES/applied_kernels_vendored.pth"

echo
echo "Setup complete."
echo "  - venv:           $CODE_ROOT/.venv"
echo "  - vendored .pth:  $SITE_PACKAGES/applied_kernels_vendored.pth"
echo
echo "Next steps:"
echo "  source .venv/bin/activate"
echo "  bash scripts/prepare_benchmark.sh"
echo "  pytest tests/test_smoke.py"
