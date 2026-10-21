"""
Build of the v13 classic CUDA module (ultimate program, package U1): v12f's forward
(census -> ALIGNED run-table kernel -> LEGACY) and v12b's backward (census per column
block -> ALIGNED -> LEGACY) in ONE extension, cqsa_cuda_v13, with v11's full per-element
path kept in both LEGACY directions (neither CQSA_NO_ELEMENT_PATH nor CQSA_BWD_NO_CQS:
those were the one-direction experiment builds, and this module serves every pattern).

    CQSA_KERNEL_SET=a100_fp16_hdim64_128 python setup.py build_ext --inplace   # fast iteration
    CQSA_KERNEL_SET=common python setup.py build_ext --inplace                 # fp16 + bf16 (the shipping set)

The sources are csrc/flash_attn here; cutlass and the version string come from the
repository root (kernels/v13/../..), so the directory carries no stub package.
"""
import os
import re
from pathlib import Path

from setuptools import setup

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME


THIS_DIR = Path(__file__).resolve().parent
REPO_DIR = THIS_DIR.parent.parent
MODULE_NAME = "cqsa_cuda_v13"


def get_version() -> str:
    init_py = (REPO_DIR / "stream_cqsa" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', init_py, re.MULTILINE)
    if not match:
        raise RuntimeError("Unable to find __version__ in stream_cqsa/__init__.py")
    return match.group(1)


def cuda_arch_list() -> list[str]:
    raw = os.getenv("FLASH_ATTN_CUDA_ARCHS", "")
    if not raw:
        try:
            major, minor = torch.cuda.get_device_capability(0)
            raw = f"{major}{minor}"
        except Exception:
            print("WARNING: no CUDA device visible at build time; defaulting to arch 80.")
            raw = "80"
    archs = [a.strip() for a in raw.split(";") if a.strip()]
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
    # Every (dtype, hdim, causal) forward and every (dtype, hdim) backward is two
    # translation units: the main unit (CQS off + the census-driven ALIGNED kernel)
    # and the *_legacy unit (v11's CQS-on kernel with the per-element path, whose
    # ptxas run dominates the build). LEGACY units first: ninja starts jobs in
    # this order and they are the long ones, so the wall time is one LEGACY unit's.
    base = [
        "csrc/flash_attn/flash_api.cpp",
        "csrc/flash_attn/src/cqsa_kernel.cu",
        "csrc/flash_attn/src/cqs_census.cu",
        "csrc/flash_attn/src/cqs_census_bwd.cu",
    ]

    def units(dtypes):
        legacy = [f"csrc/flash_attn/src/flash_fwd_hdim{hd}_{dt}{causal}_legacy_sm80.cu"
                  for dt in dtypes for hd in (64, 128) for causal in ("", "_causal")]
        legacy += [f"csrc/flash_attn/src/flash_bwd_hdim{hd}_{dt}_legacy_sm80.cu" for dt in dtypes for hd in (64, 128)]
        main = [f"csrc/flash_attn/src/flash_fwd_hdim{hd}_{dt}{causal}_sm80.cu"
                for dt in dtypes for hd in (64, 128) for causal in ("", "_causal")]
        main += [f"csrc/flash_attn/src/flash_bwd_hdim{hd}_{dt}_sm80.cu" for dt in dtypes for hd in (64, 128)]
        return legacy + main

    kernel_set = cqsa_kernel_set()
    if kernel_set == "common":
        # The shipping set: fp16 + bf16, head dims 64 and 128, causal and non-causal.
        return base + units(("fp16", "bf16"))
    if kernel_set == "a100_fp16_hdim64_128":
        return base + units(("fp16",))
    raise RuntimeError(f"Unknown CQSA_KERNEL_SET='{kernel_set}' for v13. Expected one of: common, a100_fp16_hdim64_128.")


def build_extension() -> CUDAExtension:
    if CUDA_HOME is None:
        raise RuntimeError("CUDA_HOME is not set. Activate a CUDA-enabled environment before building CQSA.")
    for flag in ("CQSA_BWD_NO_CQS", "CQSA_NO_ELEMENT_PATH"):
        if os.getenv(flag):
            raise RuntimeError(f"v13: {flag} is a one-direction experiment flag (v11x / v12f / v12b); "
                               "the v13 module keeps v11's full mask path in both LEGACY directions")
    cutlass_header = REPO_DIR / "csrc" / "cutlass" / "include" / "cutlass" / "cutlass.h"
    if not cutlass_header.exists():
        raise RuntimeError(f"cutlass is missing at {cutlass_header.parent.parent.parent}")

    kernel_set = cqsa_kernel_set()
    nvcc_threads = os.getenv("NVCC_THREADS", "4")
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
            # Build diagnostics: source lines in the SASS, the ptxas resource report per
            # instantiation, and per-phase compile times appended to this CSV. The path is
            # part of every nvcc command line, so build.sh keeps it constant across tags.
            "-lineinfo", "-Xptxas", "-v", "-time", os.environ["CQSA_NVCC_TIME_CSV"],
        ] if os.getenv("CQSA_NVCC_TIME_CSV") else []) + [
            "--threads", str(nvcc_threads),
            *nvcc_arch_flags(cuda_arch_list()),
        ],
    }
    # The MINIMAL dispatch (head dims 64 / 128 only) is the one that carries the
    # census -> ALIGNED -> LEGACY routing; flash_api.cpp refuses to build without it.
    define_macros = [("CQSA_MINIMAL_FWD_KERNELS", None)]
    if kernel_set == "common":
        define_macros.append(("CQSA_MINIMAL_BOTH_DTYPES", None))

    return CUDAExtension(
        name=MODULE_NAME,
        sources=cqsa_sources(),
        include_dirs=[
            str(THIS_DIR / "csrc" / "flash_attn"),
            str(THIS_DIR / "csrc" / "flash_attn" / "src"),
            str(REPO_DIR / "csrc" / "cutlass" / "include"),
        ],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )


setup(
    name="stream-cqsa-v13-kernel",
    version=get_version(),
    description="Stream-CQSA v13 classic CUDA module (census + ALIGNED + LEGACY, forward and backward)",
    packages=[],
    ext_modules=[build_extension()],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
    python_requires=">=3.9",
)
