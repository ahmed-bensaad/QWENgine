#!/usr/bin/env bash
# Continuous integration without a GPU (GitHub-hosted runners have none): installs the CUDA 12.4 compiler and
# libraries, builds everything for sm_75 and runs the checks that need neither a GPU nor the model files (JSON, chat
# template, metrics, image resize vs Pillow, Python and shell syntax). Used by .github/workflows/ci.yml.
# Reproduce it in a clean container (the repository, with its submodule checked out, is mounted read-only; the build
# stays in the container):
#
#   docker run --rm -v "$PWD":/src:ro ubuntu:22.04 bash /src/tools/ci.sh /src /tmp/build
#
# Arguments: source directory (default: this repository), build directory (default: <source>/build-ci).
set -euo pipefail
src="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
build="${2:-$src/build-ci}"
sudo=$(command -v sudo || true)
export DEBIAN_FRONTEND=noninteractive

$sudo apt-get update -q
$sudo apt-get install -y -q --no-install-recommends wget ca-certificates gnupg g++-12 ninja-build python3 python3-pil
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
     -O /tmp/cuda-keyring.deb
$sudo dpkg -i /tmp/cuda-keyring.deb
# CMake >= 3.24: GitHub's runner image has one; Ubuntu 22.04 ships 3.22, so take Kitware's package otherwise
if ! command -v cmake > /dev/null || ! printf '3.24\n%s\n' "$(cmake --version | head -1 | awk '{print $3}')" | sort -V -C; then
    wget -qO- https://apt.kitware.com/keys/kitware-archive-latest.asc | gpg --dearmor |
        $sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg > /dev/null
    echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main" |
        $sudo tee /etc/apt/sources.list.d/kitware.list > /dev/null
    kitware_cmake=cmake
fi
$sudo apt-get update -q
$sudo apt-get install -y -q --no-install-recommends cuda-nvcc-12-4 cuda-cudart-dev-12-4 libcublas-dev-12-4 cuda-cccl-12-4 \
     ${kitware_cmake:-}
export PATH="/usr/local/cuda-12.4/bin:$PATH"

cmake -S "$src" -B "$build" -G Ninja -DCMAKE_CXX_COMPILER=g++-12 -DCMAKE_CUDA_ARCHITECTURES=75 \
      -DQ38_MODEL_DIR=/nonexistent -DQ38_GGML_DIR=
cmake --build "$build"
ctest --test-dir "$build" --output-on-failure -R '^unit\.(json|chat|metrics|image_resize)$'

# scripts: syntax only (bytecode kept out of the source tree)
PYTHONPYCACHEPREFIX="$build/pycache" python3 -m py_compile "$src"/bench/*.py "$src"/tests/unit/*.py "$src"/tests/integration/*.py "$src"/tools/*.py
for f in "$src"/tools/*.sh "$src"/tests/golden/*.sh; do bash -n "$f"; done
echo "CI: build and GPU-free checks passed"
