import glob
import os
import re
from pathlib import Path

from setuptools import find_packages, setup

# CQSA_SKIP_EXT=1 builds the pure-Python package (the Triton kernels need no
# compiler); the CUDA extensions are then not built and torch is not needed at
# build time. This is what the pure wheel on the release page is.
SKIP_EXT = os.getenv("CQSA_SKIP_EXT", "").strip().lower() in ("1", "true", "yes")
if not SKIP_EXT:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME


THIS_DIR = Path(__file__).resolve().parent
# CQSA_ONLY_NATIVE=1 builds the COMPANION wheel `stream-cqsa-native`: just the
# `cqsa_native` extension (the multi-subproblem wave kernel), no Python package.
# Installed next to `stream-cqsa`, it is picked up automatically
# (stream_cqsa.native_wave imports `cqsa_native` if present). Kept separate so
# that each wheel fits a hosted CI runner's time limit; a source build of the
# main package can still fold it in with CQSA_BUILD_NATIVE=1.
ONLY_NATIVE = os.getenv("CQSA_ONLY_NATIVE", "").strip().lower() in ("1", "true", "yes")
PACKAGE_NAME = "stream-cqsa-native" if ONLY_NATIVE else "stream-cqsa"


def get_version() -> str:
    init_py = (THIS_DIR / "stream_cqsa" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', init_py, re.MULTILINE)
    if not match:
        raise RuntimeError("Unable to find __version__ in stream_cqsa/__init__.py")
    return match.group(1)


def detect_cuda_arch() -> str:
    """
    Compute capability of the GPU in this machine, e.g. "80" for A100.

    Source installs are the supported path: the user builds for the GPU they
    actually have, so the default is to detect it rather than to guess. Set
    FLASH_ATTN_CUDA_ARCHS explicitly to cross-compile or to cover several
    generations, e.g. "80;86;89;90".
    """
    try:
        major, minor = torch.cuda.get_device_capability(0)
        return f"{major}{minor}"
    except Exception:
        # No visible GPU at build time (CI, container, login node). sm_80 is a
        # reasonable floor: FlashAttention's SM80 kernels need Ampere or newer.
        print("WARNING: no CUDA device visible at build time; defaulting to "
              "arch 80. Set FLASH_ATTN_CUDA_ARCHS to match your target GPU.")
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
    # `common` is the shipping default: fp16 + bf16, head dims 64 and 128, causal and
    # non-causal (most transformers). `a100_fp16_hdim64_128` drops bf16 for a shorter
    # development build. Other head dims have no census / ALIGNED kernels (kernels/v13).
    return os.getenv("CQSA_KERNEL_SET", "common").strip().lower()


def cqsa_sources(root: str = "csrc") -> list[str]:
    base = [
        f"{root}/flash_attn/flash_api.cpp",
        f"{root}/flash_attn/src/cqsa_kernel.cu",
    ]
    if root.startswith("native"):
        # native/ is the wave kernel v2 (the same sources as kernels/native_v2): the classic
        # module's census + ALIGNED + LEGACY design on the uniform wave layout, W subproblems
        # per launch, plus the wave merge / scatter-add kernels; the same translation units as
        # csrc/ (head dims 64 and 128, causal and non-causal, every unit split main / legacy)
        base.append(f"{root}/flash_attn/src/wave_kernels.cu")

    # csrc/ is the v13 classic module (the same sources as kernels/v13): a census kernel
    # classifies every tile of a call on the device and routes it to the ALIGNED run-table
    # kernel (upstream FlashAttention-2's loop over the live tiles) or to v11's CQS kernel
    # (LEGACY, the per-element mask path, for mixed tiles), in both directions. Every
    # (dtype, hdim, causal) forward and every (dtype, hdim) backward is two translation
    # units, the main one and the *_legacy one whose ptxas run dominates the build; the
    # dispatch covers head dims 64 and 128 (flash_api.cpp's CQSA_MINIMAL_FWD_KERNELS branch),
    # causal and non-causal, so one module serves both. LEGACY units first: ninja starts
    # jobs in this order and they are the long ones.
    base += [f"{root}/flash_attn/src/cqs_census.cu", f"{root}/flash_attn/src/cqs_census_bwd.cu"]
    kernel_set = cqsa_kernel_set()
    dtypes = {"common": ("fp16", "bf16"), "a100_fp16_hdim64_128": ("fp16",)}.get(kernel_set)
    if dtypes is None:
        raise RuntimeError(
            f"Unknown CQSA_KERNEL_SET='{kernel_set}'. Expected common (fp16 + bf16, the shipping set) or "
            "a100_fp16_hdim64_128 (fp16 only); both cover head dims 64 and 128, causal and non-causal."
        )
    legacy = [f"{root}/flash_attn/src/flash_fwd_hdim{hd}_{dt}{causal}_legacy_sm80.cu"
              for dt in dtypes for hd in (64, 128) for causal in ("", "_causal")]
    legacy += [f"{root}/flash_attn/src/flash_bwd_hdim{hd}_{dt}_legacy_sm80.cu" for dt in dtypes for hd in (64, 128)]
    main = [f"{root}/flash_attn/src/flash_fwd_hdim{hd}_{dt}{causal}_sm80.cu"
            for dt in dtypes for hd in (64, 128) for causal in ("", "_causal")]
    main += [f"{root}/flash_attn/src/flash_bwd_hdim{hd}_{dt}_sm80.cu" for dt in dtypes for hd in (64, 128)]
    return base + legacy + main


def build_extension(name: str = "cqsa_cuda", root: str = "csrc"):
    if CUDA_HOME is None:
        raise RuntimeError("CUDA_HOME is not set. Activate a CUDA-enabled environment before building CQSA.")
    if torch.version.hip is not None:
        raise RuntimeError("CQSA lite setup currently supports CUDA only (ROCm disabled).")

    cutlass_header = THIS_DIR / "csrc" / "cutlass" / "include" / "cutlass" / "cutlass.h"
    if not cutlass_header.exists():
        raise RuntimeError(
            "csrc/cutlass is missing. Copy it from flash-attention or clone with submodules before build."
        )

    kernel_set = cqsa_kernel_set()
    sources = cqsa_sources(root)
    if not sources:
        raise RuntimeError("No CUDA sources found for CQSA build")
    if root == "csrc" and os.getenv("CQSA_BWD_NO_CQS"):
        raise RuntimeError("CQSA_BWD_NO_CQS is a measurement flag of the native tree's v11 backward; the classic "
                           "module's LEGACY backward keeps the full CQS path")

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
            # Measurement build: compiles CQS out of the backward kernel to
            # recover the stock-FlashAttention baseline. Not correct; A/B only.
            "-DCQSA_BWD_NO_CQS",
        ] if os.getenv("CQSA_BWD_NO_CQS") else []) + [
            "--threads",
            str(nvcc_threads),
            *arch_flags,
        ],
    }

    define_macros = []
    # the classic module's census -> ALIGNED -> LEGACY routing lives only in the MINIMAL dispatch
    # (head dims 64 / 128); flash_api.cpp refuses to build without it
    if root == "csrc" or kernel_set != "full":
        define_macros.append(("CQSA_MINIMAL_FWD_KERNELS", None))
    if kernel_set == "common":
        define_macros.append(("CQSA_MINIMAL_BOTH_DTYPES", None))

    return CUDAExtension(
        name=name,
        sources=sources,
        include_dirs=[
            str(THIS_DIR / root / "flash_attn"),
            str(THIS_DIR / root / "flash_attn" / "src"),
            str(THIS_DIR / "csrc" / "cutlass" / "include"),   # cutlass is vendored once, shared by both
        ],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )


if ONLY_NATIVE and not SKIP_EXT:
    setup(
        name=PACKAGE_NAME,
        version=get_version(),
        description="Stream-CQSA native wave kernel (cqsa_native): companion extension for stream-cqsa",
        packages=[],
        ext_modules=[build_extension("cqsa_native", "native/csrc")],
        cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
        python_requires=">=3.9",
        install_requires=["torch"],
    )
    raise SystemExit(0)

def _readme() -> str:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md")
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


_URL = "https://github.com/yiming-b/Stream-CQSA-v3"

setup(
    name=PACKAGE_NAME,
    version=get_version(),
    description="Stream-CQSA: exact out-of-memory recovery for attention. One function with SDPA's signature: the normal "
                "kernel when the call fits, an exact cyclic-quorum-set decomposition when it does not; training and GQA included.",
    long_description=_readme(),
    long_description_content_type="text/markdown",
    author="Yiming Bian",
    license="BSD-3-Clause",
    url=_URL,
    project_urls={
        "Source": _URL,
        "Releases (CUDA wheels)": _URL + "/releases",
        "Changelog": _URL + "/blob/main/CHANGELOG.md",
        "Paper (v1)": "https://arxiv.org/abs/2604.20819",
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Environment :: GPU :: NVIDIA CUDA",
    ],
    packages=find_packages(include=["stream_cqsa", "stream_cqsa.*"]),
    # Extensions (see docs/kernel_technical_note.md, docs/ultimate/03_kernels_report.md):
    #   cqsa_cuda     csrc/     the v13 classic module: census + ALIGNED run-table kernels with
    #                           v11's code as the LEGACY fallback, forward and backward, causal and
    #                           non-causal in one module (CQS-off path == FlashAttention-2). v2's
    #                           second extension (cqsa_cuda_nc, v9's non-causal forward) is retired
    #                           and its tree (csrc_nc/) removed in 3.0.0.
    #   cqsa_native   native/   the multi-subproblem wave kernel v2 (optional: CQSA_BUILD_NATIVE=1;
    #                           head dims 64 and 128, both dtypes); stream_cqsa.native_wave uses it
    ext_modules=[] if SKIP_EXT else (
        [build_extension("cqsa_cuda", "csrc")]
        + ([build_extension("cqsa_native", "native/csrc")]
           if os.getenv("CQSA_BUILD_NATIVE", "").strip().lower() in ("1", "true", "yes") else [])),
    cmdclass={} if SKIP_EXT else {"build_ext": BuildExtension.with_options(use_ninja=True)},
    python_requires=">=3.9",
    # Triton is the no-build kernel path. PyTorch's Linux wheels already depend on it; on Windows the
    # official package has no wheels, so the community build `triton-windows` is used instead.
    install_requires=["torch", "numpy",
                      'triton>=3.0; sys_platform == "linux"',
                      'triton-windows>=3.0; sys_platform == "win32"'],
    extras_require={
        "triton": ["triton"],                      # the no-build kernels (explicit, any platform)
        "progress": ["tqdm"],                      # verbose=True progress bars (a plain bar is used without it)
        "bench": ["matplotlib", "tqdm"],
        "test": ["pytest"],
    },
    entry_points={"console_scripts": ["stream-cqsa-doctor=stream_cqsa.doctor:_main"]},
)
