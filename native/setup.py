"""Build the native wave extension cqsa_native_v2 (package U2 of the ultimate program).

Kernel sets (CQSA_KERNEL_SET): `common` (default; fp16 + bf16, head dims 64 + 128, causal and
non-causal) and `native_v2_dev` (head dim 64 fp16 only, for fast iteration; defines
CQSA_NATIVE_HDIM64_ONLY so the dispatch never references the hdim128 units). Every forward
(dtype, hdim, causal) and every backward (dtype, hdim) is two translation units: the main one
(CQS off + the census-driven ALIGNED wave kernel) and `_legacy` (v11's CQS code with the wave
offsets, the per-element path whose ptxas run dominates the build). CQSA_NATIVE_V2_MODULE names
the module (default cqsa_native_v2) so a dev build can live next to the shipping one.
"""
import os
from pathlib import Path

from setuptools import setup

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME


THIS_DIR = Path(__file__).resolve().parent


def detect_cuda_arch() -> str:
    """Compute capability of the GPU in this machine, e.g. "80" for A100; sm_80 without one."""
    try:
        major, minor = torch.cuda.get_device_capability(0)
        return f"{major}{minor}"
    except Exception:
        print("WARNING: no CUDA device visible at build time; defaulting to arch 80. "
              "Set FLASH_ATTN_CUDA_ARCHS to match your target GPU.")
        return "80"


def cuda_arch_list() -> list[str]:
    raw = os.getenv("FLASH_ATTN_CUDA_ARCHS", "") or detect_cuda_arch()
    archs = [a.strip() for a in raw.split(";") if a.strip()]
    if not archs:
        raise RuntimeError("FLASH_ATTN_CUDA_ARCHS resolved to an empty list")
    for a in archs:
        if not a.isdigit():
            raise RuntimeError(f"Invalid CUDA arch '{a}'. Use numeric values like 80;90")
    return archs


def nvcc_arch_flags(archs: list[str]) -> list[str]:
    flags: list[str] = []
    for arch in archs:
        flags.extend(["-gencode", f"arch=compute_{arch},code=sm_{arch}"])
    newest = max(archs, key=int)
    flags.extend(["-gencode", f"arch=compute_{newest},code=compute_{newest}"])
    return flags


def cqsa_kernel_set() -> str:
    return os.getenv("CQSA_KERNEL_SET", "common").strip().lower()


def cqsa_sources() -> list[str]:
    base = [
        "csrc/flash_attn/flash_api.cpp",
        "csrc/flash_attn/src/cqsa_kernel.cu",
        "csrc/flash_attn/src/cqs_census.cu",
        "csrc/flash_attn/src/cqs_census_bwd.cu",
        "csrc/flash_attn/src/wave_kernels.cu",
    ]
    kernel_set = cqsa_kernel_set()
    if kernel_set == "common":
        dtypes, hdims = ("fp16", "bf16"), (64, 128)
    elif kernel_set == "native_v2_dev":
        dtypes, hdims = ("fp16",), (64,)
    else:
        raise RuntimeError(f"Unknown CQSA_KERNEL_SET='{kernel_set}' for native_v2. Expected common or native_v2_dev.")
    # The main units first: they are short (minutes) and hold every new line of code, so a
    # compile error surfaces early; the LEGACY units (v11's code, 20-30 min each) follow.
    return base + [
        f"csrc/flash_attn/src/flash_fwd_hdim{hd}_{dt}{causal}{part}_sm80.cu"
        for part in ("", "_legacy") for dt in dtypes for hd in hdims for causal in ("", "_causal")
    ] + [
        f"csrc/flash_attn/src/flash_bwd_hdim{hd}_{dt}{part}_sm80.cu"
        for part in ("", "_legacy") for dt in dtypes for hd in hdims
    ]


def build_extension() -> CUDAExtension:
    if CUDA_HOME is None:
        raise RuntimeError("CUDA_HOME is not set. Activate a CUDA-enabled environment before building CQSA.")
    if torch.version.hip is not None:
        raise RuntimeError("CQSA lite setup currently supports CUDA only (ROCm disabled).")

    cutlass_header = THIS_DIR / "csrc" / "cutlass" / "include" / "cutlass" / "cutlass.h"
    if not cutlass_header.exists():
        raise RuntimeError("csrc/cutlass is missing (a symlink to the repo's csrc/cutlass).")

    kernel_set = cqsa_kernel_set()
    nvcc_threads = os.getenv("NVCC_THREADS", "4")
    arch_flags = nvcc_arch_flags(cuda_arch_list())

    extra_compile_args = {
        "cxx": ["-O3", "-std=c++17"],
        "nvcc": [
            "-O3",
            "-std=c++17",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--use_fast_math",
        ] + ([
            # Build diagnostics: source lines in the SASS, the ptxas resource
            # report per instantiation, and per-phase compile times (ptxas
            # included) appended to this CSV, one row per phase and TU.
            "-lineinfo", "-Xptxas", "-v", "-time", os.environ["CQSA_NVCC_TIME_CSV"],
        ] if os.getenv("CQSA_NVCC_TIME_CSV") else []) + [
            "--threads",
            str(nvcc_threads),
            *arch_flags,
        ],
    }

    # The CQS-on routing (census -> ALIGNED -> LEGACY) exists only in the minimal dispatch.
    define_macros = [("CQSA_MINIMAL_FWD_KERNELS", None)]
    if kernel_set == "common":
        define_macros.append(("CQSA_MINIMAL_BOTH_DTYPES", None))
    else:
        define_macros.append(("CQSA_NATIVE_HDIM64_ONLY", None))

    return CUDAExtension(
        name=os.getenv("CQSA_NATIVE_V2_MODULE", "cqsa_native_v2"),
        sources=cqsa_sources(),
        include_dirs=[
            str(THIS_DIR / "csrc" / "flash_attn"),
            str(THIS_DIR / "csrc" / "flash_attn" / "src"),
            str(THIS_DIR / "csrc" / "cutlass" / "include"),
        ],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )


setup(
    name="cqsa-native-v2",
    version="3.0.0.dev0",
    description="Stream-CQSA native wave kernel: census + ALIGNED run-table forward and backward on the uniform wave layout",
    ext_modules=[build_extension()],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
    python_requires=">=3.9",
    install_requires=["torch"],
)
