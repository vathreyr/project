#!/usr/bin/env bash
# =============================================================================
# setup_psc.sh  --  create a venv and install a V100-compatible vLLM on PSC.
#
# WHY THIS IS CAREFUL (and not a one-line `pip install vllm`):
#   The target GPU is a V100 (Volta, sm_70). Recent vLLM prebuilt wheels
#   (>=0.20.0) DROPPED sm_70 kernels, and FlashAttention/FlashInfer never
#   supported Volta. vLLM 0.18.x is the last line reported to ship sm_70
#   wheels AND is modern enough to have the V1/V2 engine, NGram GPU spec
#   decode, and per-request acceptance metrics. We pin to it and then VERIFY
#   that the installed build can actually touch the V100 before trusting it.
#
# This script does NOT guess PSC's module system for you -- load your CUDA
# toolchain module first (e.g. `module load cuda`), then run this.
#
# Usage:
#   bash hw/setup_psc.sh            # create .venv_hw and install
#   source .venv_hw/bin/activate    # then use it
# =============================================================================
set -euo pipefail

VENV="${VENV:-.venv_hw}"
# PSC Bridges-2 is RHEL8 (glibc 2.28). vLLM wheels switched from `manylinux1`
# (glibc 2.5, installable here) to `manylinux_2_31` (glibc >= 2.31, NOT
# installable here) at v0.12.0. So 0.11.0 is the newest vLLM with a wheel that
# installs on PSC; 0.12+ would fall back to a source build (needs CUDA_HOME,
# compiles for hours). 0.11.0 pins torch==2.8.0, which still ships sm_70 (V100)
# kernels, runs the V1 engine by default, and has n-gram speculative decoding.
# Override with VLLM_VERSION=... only if the self-check below reports no sm_70.
VLLM_VERSION="${VLLM_VERSION:-0.11.0}"
PYTHON="${PYTHON:-python3}"

echo "=== 0. Pre-flight: is a GPU visible at install time? ==="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version \
               --format=csv,noheader || true
else
    echo "WARNING: nvidia-smi not found. If you are on a login node, that is"
    echo "         expected -- but you MUST install and run on a GPU node"
    echo "         (e.g. inside an salloc/srun) or nothing below will work."
fi

echo
echo "=== 1. Create venv: ${VENV} ==="
# vLLM needs Python >= 3.10. PSC's bare `python3` is often 3.6 (too old) --
# fail fast with guidance instead of a confusing 'No matching distribution'.
PYVER="$("${PYTHON}" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
echo "Using ${PYTHON} = Python ${PYVER}"
"${PYTHON}" - <<'PY' || {
import sys
sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)
PY
    echo "ERROR: Python ${PYVER} is too old for vLLM (need >= 3.10)."
    echo "Load a modern Python module first, e.g. on Bridges-2:"
    echo "    module load anaconda3     # then re-run this script"
    echo "or point PYTHON at a newer interpreter:"
    echo "    PYTHON=/path/to/python3.11 bash hw/setup_psc.sh"
    exit 1
}
"${PYTHON}" -m venv "${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install --upgrade pip wheel setuptools

echo
echo "=== 2. Install vLLM ${VLLM_VERSION} (pulls a matching torch) ==="
# Let vLLM pull the torch build it was compiled against; forcing a different
# torch is the usual way a V100 install breaks.
python -m pip install "vllm==${VLLM_VERSION}"

# vLLM 0.11.0 pins only `transformers>=4.55.2` (no upper bound), so a fresh
# install pulls transformers 5.x (+ tokenizers 0.23.3) -- transformers 5.x
# REMOVED all_special_tokens_extended and crashes tokenizer loading. Pin to the
# 4.x line 0.11.0 was built against, AND the matching tokenizers, or
# transformers' own import-time version check aborts otherwise. transformers
# 4.57.6 needs tokenizers>=0.22.0,<=0.23.0 (0.23.0 was never released stable ->
# use 0.22.2) AND huggingface-hub>=0.34.0,<1.0 (the newer stack pulls hub 1.x,
# which that check rejects). Pin all three to a consistent, vLLM-0.11.0-safe set.
python -m pip install "transformers==4.57.6" "tokenizers==0.22.2" \
    "huggingface-hub<1.0"

# Experiment deps (trace replay, plots, async load client, metric scrape).
python -m pip install numpy matplotlib pandas aiohttp datasets requests

echo
echo "=== 3. VERIFY the build can touch this V100 ==="
# This is the real test: import vllm, confirm CUDA is live, print the compute
# capability, and -- crucially -- run one tiny GPU op. If the wheel has no
# sm_70 kernels this will raise "no kernel image is available for execution".
python - <<'PY'
import sys
try:
    import torch
except Exception as e:
    print("FAIL: torch import:", e); sys.exit(2)

print("torch:", torch.__version__, "| cuda build:", torch.version.cuda)
if not torch.cuda.is_available():
    print("FAIL: torch.cuda.is_available() is False on this node."); sys.exit(3)

cc = torch.cuda.get_device_capability(0)
name = torch.cuda.get_device_name(0)
print(f"device: {name} | compute capability: {cc}")
archs = getattr(torch.cuda, "get_arch_list", lambda: [])()
print("torch compiled archs:", archs)
if cc == (7, 0) and archs and not any("70" in a for a in archs):
    print("WARNING: torch was not compiled for sm_70 -- GPU ops will fail.")

# The decisive check: a real kernel launch on the GPU.
try:
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    y = (x @ x).sum().item()
    print("OK: fp16 matmul on GPU succeeded (sum=%.3f)." % y)
except Exception as e:
    print("FAIL: GPU kernel launch failed -> this wheel has no sm_70 kernels.")
    print("     ", e)
    sys.exit(4)

try:
    import vllm
    print("OK: vllm", vllm.__version__, "imported.")
except Exception as e:
    print("FAIL: vllm import:", e); sys.exit(5)
print("\nENV OK: this build can run on the V100.")
PY

echo
echo "=== DONE ==="
echo "If step 3 printed 'ENV OK', activate the venv and run:"
echo "    source ${VENV}/bin/activate"
echo "    python hw/check_env.py"
echo "    python hw/step1_smoke.py"
echo
echo "If step 3 FAILED with 'no sm_70 kernels' / 'no kernel image', try a"
echo "lower rung of the ladder, e.g.:  VLLM_VERSION=0.17.0 bash hw/setup_psc.sh"
echo "and report which version first prints 'ENV OK'."
