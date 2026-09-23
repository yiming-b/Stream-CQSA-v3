"""
Automatic configuration of Stream-CQSA from a hardware description.

    from stream_cqsa.autoconfig import detect_hardware, plan, autotune, auto_attention

    hw  = detect_hardware()                        # every CUDA device + host RAM
    hw  = hardware_from_dict({"cuda:0": "40GiB", "cuda:1": "40GiB", "host": "200GiB"})
    cfg = plan(N=2**20, B=1, H=8, D=64, dtype=torch.float16, causal=True, hardware=hw)
    out = auto_attention(q, k, v, causal=True, hardware=hw)   # plan + run

The two knobs the user otherwise has to set are the decomposition depth
(``itr``: 0 = monolithic call, 1 = 7 subproblems, 2 = 49, ...) and where the
accumulator lives (``acc``: gpu or cpu), plus whether Q/K/V stay on the host,
how many subproblems run concurrently, and over how many devices. The planner
enumerates every combination, predicts peak device memory with the engine's
own estimators and wall time with a small cost model, discards what does not
fit the budget, and applies one rule:

    use as much of the memory budget as helps, unless a configuration is both
    faster and smaller -- i.e. take the Pareto frontier of (time, memory) among
    feasible configurations and choose its fastest point.

For a small N with an ample budget that always yields the monolithic call:
Stream-CQSA at itr=1 does 9/7 of the monolithic pair work (of which 22% is then
skipped as fully masked tiles) plus gather/merge, and measured end to end it is
1.4-1.6x the monolithic time at equal residency -- so "all 7 subproblems in
parallel with a device accumulator" is chosen only when the monolithic call
does not fit, which is exactly what the measurements say.

The cost model is deliberately simple (pair throughput + per-token host stage
costs + a per-depth factor + distributed balance/merge terms) and its constants
come from the A100 measurements in next/LOG.md. ``calibrate(hardware)`` refits
them on the machine at hand by timing a monolithic call and a few Stream-CQSA
configurations at a small N (a minute), and ``autotune`` replaces the model by
direct measurement of the feasible candidates when the shape is small enough
to afford it.
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Sequence

import torch

from .stable_stream import (estimate_monolithic_bytes, estimate_peak_bytes, estimate_peak_bytes_bwd, per_el_floor, effective_free_bytes)
from .scheduler import device_rate, equal_task_share

GIB = float(1 << 30)

# Perfect (Singer) difference sets: c = l^2 - l + 1 chunks, |interest_set| = l.
# A subproblem gathers l of the c chunks; the pair work per depth is (l^2/c)^itr
# of the monolithic call, the masked fraction inside a subproblem is (l-1)/l^2,
# the gathered tokens per depth are l^itr * N and there are c^itr tasks.
# The engine's int64 group bits allow itr*(l-1) <= 63.
QUORUM_SETS: dict[int, tuple[int, ...]] = {
    3: (0, 1), 7: (0, 1, 3), 13: (0, 1, 3, 9), 21: (0, 1, 4, 14, 16),
    31: (0, 1, 3, 8, 12, 18), 57: (0, 1, 3, 13, 32, 36, 43, 52),
    73: (0, 1, 3, 7, 15, 31, 36, 54, 63),
    # Singer set for q = 9 (from the trace of GF(q^3) over GF(q)); verified perfect
    91: (0, 20, 28, 37, 60, 70, 73, 84, 85, 89),
    # q = 11 (l = 12); verified perfect
    133: (0, 1, 3, 12, 20, 34, 38, 81, 88, 94, 104, 109),
}


def is_perfect_difference_set(c: int, s: Sequence[int]) -> bool:
    d = sorted((a - b) % c for a in s for b in s if a != b)
    return d == list(range(1, c))


# ---------------------------------------------------------------------------
# hardware description
# ---------------------------------------------------------------------------

def _parse_bytes(x) -> int:
    if isinstance(x, (int, float)):
        return int(x)
    m = re.fullmatch(r"\s*([0-9.]+)\s*([kmgt]i?b?)?\s*", str(x), re.I)
    if not m:
        raise ValueError(f"cannot parse memory size {x!r}")
    val = float(m.group(1)); unit = (m.group(2) or "b").lower().rstrip("b").rstrip("i")
    return int(val * {"": 1, "k": 2**10, "m": 2**20, "g": 2**30, "t": 2**40}[unit])


@dataclass
class DeviceSpec:
    name: str                 # torch device string, e.g. 'cuda:0'
    model: str = "unknown"
    budget_bytes: int = 0     # memory this call may use on the device
    total_bytes: int = 0
    n_sm: int = 108
    identical_group: int = 0  # devices with the same model share a group
    # live 128x128-tile throughput of the forward kernel (tiles/s, one head, D=64, fp16): a cached
    # calibration of this GPU model (scheduler.calibrate_rate) or the model's seed (scheduler.KNOWN_DEVICE_RATES);
    # 0 = unknown (treated as equal to the other devices)
    tile_rate: float = 0.0
    rate_source: str = ""


@dataclass
class HardwareSpec:
    devices: list[DeviceSpec] = field(default_factory=list)
    host_budget_bytes: int = 0        # host RAM this call may use (pinned Q/K/V, accumulator)
    n_cpu: int = 8
    interconnect_gbs: float = 25.0    # per-direction device<->device bandwidth used by the merge (NVLink ~200, PCIe ~25)

    def homogeneous(self) -> bool:
        return len({(d.model, d.tile_rate) for d in self.devices}) <= 1

    def rel_rates(self, devices: Sequence[DeviceSpec] | None = None) -> tuple:
        """Rates of ``devices`` (default: all) relative to the first one (1.0 where unknown)."""
        devs = list(devices if devices is not None else self.devices)
        if not devs or not devs[0].tile_rate or any(not d.tile_rate for d in devs):
            return tuple(1.0 for _ in devs)
        return tuple(d.tile_rate / devs[0].tile_rate for d in devs)

    def summary(self) -> str:
        devs = ", ".join(f"{d.name}={d.model} {d.budget_bytes / GIB:.1f}/{d.total_bytes / GIB:.1f} GiB"
                         + (f" {d.tile_rate:.2e} tiles/s" if d.tile_rate else "") for d in self.devices)
        return f"devices[{devs}] host {self.host_budget_bytes / GIB:.0f} GiB, {self.n_cpu} cpus, link {self.interconnect_gbs:.0f} GB/s"


def _host_free_bytes() -> int:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 64 * (1 << 30)


def detect_hardware(*, budget_fraction: float = 0.9, host_fraction: float = 0.8,
                    devices: Sequence[int] | None = None) -> HardwareSpec:
    """Describe the visible CUDA devices (free memory x budget_fraction) and the host."""
    hw = HardwareSpec(host_budget_bytes=int(_host_free_bytes() * host_fraction),
                      n_cpu=len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 8))
    if not torch.cuda.is_available():
        return hw
    idx = list(devices) if devices is not None else list(range(torch.cuda.device_count()))
    models = {}
    for i in idx:
        p = torch.cuda.get_device_properties(i)
        free, total = effective_free_bytes(i), p.total_memory     # honours a set_per_process_memory_fraction cap
        grp = models.setdefault(p.name, len(models))
        dr = device_rate(p.name)
        hw.devices.append(DeviceSpec(name=f"cuda:{i}", model=p.name, budget_bytes=int(free * budget_fraction),
                                     total_bytes=int(total), n_sm=p.multi_processor_count, identical_group=grp,
                                     tile_rate=dr.tile_rate, rate_source=dr.source))
    if len(hw.devices) > 1:
        # the slowest link of the ring over the planned devices (per-pair classes from `nvidia-smi topo -m`)
        topo = _topo_text()
        mat = parse_topo(topo) if topo else []
        hw.interconnect_gbs = ring_link_gbs(mat, [int(d.name.split(":")[1]) for d in hw.devices]) if mat else 25.0
    return hw


def _topo_text() -> str:
    try:
        import subprocess
        return subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return ""


def _nvlink_present() -> bool:
    return bool(re.search(r"\bNV\d+\b", _topo_text()))


# Per-direction bandwidth seeds (GB/s) of the `nvidia-smi topo -m` link classes between two GPUs, for the merge's
# collectives. NV#: # bonded NVLinks (A100 NVLink3: 25 GB/s per link per direction). PCIe 4.0 x16 through one
# switch (PIX) ~24, several switches (PXB) ~22, the host bridge (PHB) ~20, across PCIe host bridges within a NUMA
# node (NODE) ~18, across the SMP interconnect (SYS) ~12. Seeds, not measurements: the D1 job records the merge
# time and bytes per row, from which the effective bandwidth of a node follows.
LINK_CLASS_GBS = {"PIX": 24.0, "PXB": 22.0, "PHB": 20.0, "NODE": 18.0, "SYS": 12.0}


def parse_topo(text: str) -> list[list[str]]:
    """The GPU x GPU link-class matrix of ``nvidia-smi topo -m`` output (``[]`` if none)."""
    rows = []
    for line in text.splitlines():
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)          # the header carries terminal escapes
        if re.match(r"GPU\d+\s", line):                      # a row (the header line starts with whitespace)
            rows.append(line.split()[1:])
    n = len(rows)
    return [r[:n] for r in rows] if n and all(len(r) >= n for r in rows) else []


def link_gbs(cls: str) -> float:
    m = re.fullmatch(r"NV(\d+)", cls)
    if m:
        return 25.0 * int(m.group(1))
    return LINK_CLASS_GBS.get(cls, 12.0 if cls != "X" else float("inf"))


def ring_link_gbs(matrix: list[list[str]], devices: Sequence[int] | None = None) -> float:
    """Bandwidth of the slowest link on the ring devices[0] -> devices[1] -> ... -> devices[0] (NCCL's ring
    runs at its slowest hop)."""
    idx = list(devices) if devices is not None else list(range(len(matrix)))
    idx = [i for i in idx if i < len(matrix)]
    if len(idx) < 2:
        return float("inf")
    return min(link_gbs(matrix[a][b]) for a, b in zip(idx, idx[1:] + idx[:1]))


def hardware_from_dict(spec: dict[str, Any], *, interconnect_gbs: float | None = None) -> HardwareSpec:
    """
    {"cuda:0": "40GiB", "cuda:1": 40e9, "host": "256GiB", "cpus": 32, "link_gbs": 200}
    Values are memory *budgets* (what the call may use). Devices default to the
    detected model/SM count when the process can see them. A device value may be a dict
    ``{"mem": "80GiB", "model": "NVIDIA A100-SXM4-80GB", "rate": 3.35e7}``: ``rate`` is the live-tile
    rate (tiles/s, ``scheduler.calibrate_rate``); without it the model's cached calibration or seed is
    used (a heterogeneous description is planned on every device with its own rate).
    """
    hw = HardwareSpec()
    models = {}
    for key, val in spec.items():
        k = str(key).lower()
        if k in ("host", "cpu", "ram"):
            hw.host_budget_bytes = _parse_bytes(val)
        elif k in ("cpus", "n_cpu", "ncpu"):
            hw.n_cpu = int(val)
        elif k in ("link_gbs", "interconnect_gbs"):
            hw.interconnect_gbs = float(val)
        else:
            m = re.fullmatch(r"(?:cuda:)?(\d+)", k)
            if not m:
                raise ValueError(f"unknown hardware key {key!r} (use 'cuda:N', 'host', 'cpus', 'link_gbs')")
            i = int(m.group(1))
            # value: a size, or {"mem": size, "model": "A100-SXM4-80GB"}; devices are
            # assumed identical unless a model string says otherwise.
            model, n_sm, rate = "same", 108, 0.0
            if isinstance(val, dict):
                model = str(val.get("model", "same")); mem = _parse_bytes(val.get("mem", val.get("budget", 0)))
                rate = float(val.get("rate", val.get("tile_rate", 0.0)) or 0.0)
            else:
                mem = _parse_bytes(val)
            total = mem
            if torch.cuda.is_available() and i < torch.cuda.device_count():
                p = torch.cuda.get_device_properties(i); n_sm, total = p.multi_processor_count, p.total_memory
            grp = models.setdefault(model, len(models))
            src = "given"
            if not rate:
                # "same" devices are identical by assumption: all take the rate of the first visible GPU's model
                # (or the default seed), never a mix of a visible device's rate and the default
                same = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"
                dr = device_rate(model if model != "same" else same)
                rate, src = dr.tile_rate, dr.source
            hw.devices.append(DeviceSpec(name=f"cuda:{i}", model=model, budget_bytes=mem,
                                         total_bytes=total, n_sm=n_sm, identical_group=grp,
                                         tile_rate=rate, rate_source=src))
    if not hw.host_budget_bytes:
        hw.host_budget_bytes = int(_host_free_bytes() * 0.8)
    if interconnect_gbs is not None:
        hw.interconnect_gbs = float(interconnect_gbs)
    return hw


# ---------------------------------------------------------------------------
# cost model
# ---------------------------------------------------------------------------

@dataclass
class CostModel:
    """
    Wall-time model. Defaults measured on A100-SXM4-80GB with the next/ engine
    (v11 kernel), fp16, H=8 D=64 (next/logs/e2e_compare_v11.json,
    pipeline_opts.json, dist_test_*.json). `calibrate` refits them.
    """
    pair_rate: float = 7.4e11        # monolithic FA-2: score pairs per second over all heads (causal 1M, H=8: 8*N^2/2 / 5.96 s)
    kernel_ratio: float = 1.40       # Stream-CQSA kernel time per live pair / FA-2 time per pair (v11 at L=56K)
    masked_frac: float = 0.22        # kept for calibrate(); the planner uses (l-1)/l^2 per quorum set
    depth_factor: float = 1.0        # residual per-depth factor (per-task overhead now explicit below)
    task_overhead_s: float = 3.0e-3  # fixed cost per subproblem (launch, scheduling, small-L kernel inefficiency); fitted from c=7..73 at N=131K
    # Wave engine: one launch per wave of several subproblems. Per subproblem only its grid tail remains
    # (wave_task_s); the launch gap is paid once per wave (wave_launch_s). The two sum to task_overhead_s, so a
    # W=1 wave costs what a classic subproblem costs (measured equal within 0.6%, bit-identical; layout_ab), and
    # a wave of 49 saves 48 launch gaps (139 vs 155 ms at itr=2, kernel_rows).
    wave_launch_s: float = 1.5e-3
    wave_task_s: float = 1.5e-3
    inplace_ratio: float = 1.045     # kernel time reading Q/K/V in place through the block map / gathered copies (+3-6%, layout_ab)
    bwd_ratio: float = 3.0           # backward (fwd+bwd of a subproblem) / forward time per pair
    # Host-stage costs per gathered token of H*D = 512 fp16 elements (the
    # measured shape); scaled by (B*H*D)/512 for other shapes.
    gather_s_per_tok: float = 0.5 / 3.1e6     # host gather (shared_chunks) per gathered token
    gather_dev_s_per_tok: float = 0.06 / 3.1e6  # device-side gather per token (inputs resident)
    merge_cpu_s_per_tok: float = 2.25 / 3.1e6  # host accumulator merge per gathered token, 8 threads
    merge_gpu_s_per_tok: float = 0.10 / 3.1e6  # device accumulator merge per gathered token
    d2h_s_per_tok: float = 0.25 / 3.1e6        # output d2h per gathered token (host accumulator)
    overlap: float = 0.85            # fraction of the host stages hidden under compute (pipelined engine, n_par=1)
    overlap_npar2: float = 0.92      # with two subproblems in flight
    dist_merge_bytes_per_tok: int = 16 * 8 * 64  # fp32 out + weights, per token (H*D*4 + ...), all_reduce volume
    dist_imbalance: float = 1.35     # per-rank straggling + host-stage contention (4 GPUs sharing one host; measured 62 s vs 40 s ideal at 4M)
    dist_fixed_s: float = 0.4        # per-call cost of the collective path (barriers, NCCL warm-up, task-list build)
    # The assignment model of the multi-device share (scheduler.equal_task_share): "none" = whole subproblems
    # by LPT on the ranks' rates (round-robin's ceil(n/world)/n on equal ranks), "children" = the driver's
    # default lpt + child-refinement splits. "none" until dist_imbalance is refitted: the 1.35 residual was
    # fitted on round-robin rows, where the model was already optimistic at 2 GPUs (26.2 s predicted vs 34.6 s
    # measured at 2M, c=7), and with the split share it would flip the measured 2-GPU verdict (monolithic wins
    # at c=7; tests/test_devkit_autoconfig.py). The D1 job (dist_sched_*.json) is the data for the refit.
    dist_split: str = "none"

    def mono_time(self, N: int, B: int, H: int, causal: bool, direction: str = "fwd") -> float:
        pairs = B * H * N * N * (0.5 if causal else 1.0)
        return pairs / self.pair_rate * (self.bwd_ratio if direction == "bwd" else 1.0)

    def cqsa_time(self, N: int, B: int, H: int, D: int, itr: int, causal: bool, acc: str,
                  stream_from_host: bool, n_par: int, world: int = 1, hw: HardwareSpec | None = None,
                  c: int = 7, l: int = 3, direction: str = "fwd", engine: str = "classic",
                  wave_tokens: int = 0, gather: bool = True, rates: Sequence[float] | None = None) -> float:
        """``engine="wave"``: waves of ``wave_tokens`` packed tokens (at least one subproblem each) replace the
        per-subproblem launches; ``gather=False`` reads Q/K/V in place at ``inplace_ratio`` the kernel time.
        ``rates``: for ``world > 1``, the ranks' rates relative to the device this model describes (default
        all 1.0); the share of the slowest rank comes from the driver's scheduler (``equal_task_share``)."""
        L = N * (l / c) ** itr
        n_tasks = c ** itr
        tokens = n_tasks * L                       # gathered tokens over all subproblems
        masked = (l - 1) / (l * l)                 # non-owner diagonal blocks of a subproblem
        pairs = n_tasks * B * H * L * L * (0.5 if causal else 1.0) * (1 - masked)
        kernel = pairs / self.pair_rate * self.kernel_ratio * (self.depth_factor ** (itr - 1))
        if engine == "wave":
            n_waves = math.ceil(tokens / max(float(wave_tokens), L))
            compute = kernel * (1.0 if gather else self.inplace_ratio) + n_waves * self.wave_launch_s + n_tasks * self.wave_task_s
        else:
            compute = kernel + n_tasks * self.task_overhead_s
        if direction == "bwd":
            compute *= self.bwd_ratio
        ts = tokens * (B * H * D) / 512.0          # token-equivalents of the calibrated shape
        gather_s = ts * (self.gather_s_per_tok if stream_from_host else self.gather_dev_s_per_tok)
        merge = ts * (self.merge_cpu_s_per_tok if acc == "cpu" else self.merge_gpu_s_per_tok)
        d2h = ts * self.d2h_s_per_tok if acc == "cpu" else 0.0
        host = gather_s + merge + d2h
        # the wave's host fold runs on a worker under the next wave (P0.3), like a second subproblem in flight
        ov = self.overlap_npar2 if (n_par >= 2 or (engine == "wave" and acc == "cpu")) else self.overlap
        t = compute + host * (1 - ov) + max(0.0, host * ov - compute)   # hidden part cannot exceed compute
        if world > 1:
            # the slowest rank's share of the single-device time: n equal subproblems assigned by the driver's
            # scheduler to ranks of relative rates `rates` (1.0 = the device this model describes)
            rel = tuple(float(r) for r in (rates if rates is not None else (1.0,) * world))[:world]
            rel = rel + (1.0,) * (world - len(rel))
            per_rank = equal_task_share(n_tasks, rel, split=self.dist_split, c=c, l=l)
            comm = 2 * N * B * H * D * 4 / (max(1.0, (hw.interconnect_gbs if hw else 25.0)) * 1e9) * math.log2(world)
            t = t * per_rank * self.dist_imbalance + comm + self.dist_fixed_s
        return t

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)

    @classmethod
    def from_json(cls, s: str) -> "CostModel":
        d = json.loads(s)
        cm = cls()
        for k_, v_ in d.items():
            if hasattr(cm, k_):
                setattr(cm, k_, v_)
        return cm


# ---------------------------------------------------------------------------
# calibration cache: one cost model per GPU model, on disk
# ---------------------------------------------------------------------------
def _cache_dir() -> str:
    return os.environ.get("CQSA_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "stream_cqsa")


def _gpu_key(device="cuda") -> str:
    try:
        name = torch.cuda.get_device_name(torch.device(device))
    except Exception:
        name = "cpu"
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_").lower()


def _cost_model_cache_path(device="cuda") -> str:
    return os.path.join(_cache_dir(), f"cost_model_{_gpu_key(device)}.json")


def save_cost_model(cm: "CostModel", device="cuda") -> str:
    path = _cost_model_cache_path(device)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(cm.to_json())
    return path


def load_cost_model(device="cuda") -> "CostModel | None":
    """The cost model calibrated for this GPU model, if ``calibrate(save=True)`` ran here before."""
    path = _cost_model_cache_path(device)
    try:
        with open(path) as f:
            return CostModel.from_json(f.read())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# candidates and the plan
# ---------------------------------------------------------------------------

@dataclass
class Plan:
    mode: str                      # 'mono' | 'cqsa' | 'cqsa_dist'
    itr: int = 0
    c: int = 7
    interest_set: tuple = (0, 1, 3)
    direction: str = "fwd"
    acc: str = "gpu"
    stream_from_host: bool = False
    n_par: int = 1
    world: int = 1
    devices: list[str] = field(default_factory=list)
    est_time_s: float = 0.0
    est_peak_gib: float = 0.0      # per device
    est_host_gib: float = 0.0
    reason: str = ""
    candidates: list[dict] = field(default_factory=list)
    evict_inputs: bool = False     # Q/K/V are on the device now and are moved to host memory for the computation
    engine: str = "classic"        # 'wave' (several subproblems per launch) or 'classic' (the pinned per-task pipeline)
    wave_tokens: int = 0           # wave engine: packed-token budget per wave (W)
    gather: bool = True            # wave engine: gathered copies of Q/K/V per wave (True) or read in place (False)
    device_rates: list = field(default_factory=list)   # live-tile rates of `devices` (tiles/s; 0 = unknown)

    def name(self) -> str:
        if self.mode == "mono":
            return "monolithic"
        s = f"cqsa c={self.c} itr={self.itr} acc={self.acc}"
        s += (f" wave W={self.wave_tokens:,} {'gathered' if self.gather else 'in place'}" if self.engine == "wave"
              else f" n_par={self.n_par}")
        s += ((" Q/K/V evicted to host" if self.evict_inputs else " host-resident Q/K/V") if self.stream_from_host else "")
        return s + (f" on {self.world} devices" if self.world > 1 else "")

    def engine_kwargs(self) -> dict:
        return dict(itr=int(self.itr), c=int(self.c), interest_set=tuple(self.interest_set),
                    low_memory=(self.acc == "cpu"), accumulate_on_gpu=(self.acc == "gpu"),
                    stream_from_host=bool(self.stream_from_host), max_parallel=int(self.n_par),
                    shared_chunks=bool(self.stream_from_host and self.itr == 1))


def _host_bytes(N, B, H, D, itemsize, stream_from_host, acc):
    b = 0
    if stream_from_host:
        b += 3 * N * B * H * D * itemsize          # pinned Q/K/V
    if acc == "cpu":
        b += N * B * H * D * 4 + N * B * H * 8     # fp32 accumulator (+ m, l)
    return b


def plan(*, N: int, B: int = 1, H: int = 8, D: int = 64, dtype=torch.float16, causal: bool = True,
         hardware: HardwareSpec | None = None, model: CostModel | None = None,
         max_itr: int = 3, n_pars: Sequence[int] = (1, 2, 4), allow_distributed: bool = True,
         quorum_sets: dict[int, Sequence[int]] | None = None, direction: str = "fwd",
         safety: float = 0.85, tol: float = 0.02, out_bytes_per_el: int = 4,
         resident_input_bytes: int = 0, budget_excludes_inputs: bool = True, evict_inputs: bool = False,
         engines: Sequence[str] | None = None, allow_mono: bool = True, inputs_on_host: bool = False,
         min_itr: int = 1) -> Plan:
    """
    Choose the configuration for one attention call on `hardware` (detected if None).
    ``out_bytes_per_el``: device bytes per element the delivered forward output costs when the
    accumulator is on the host (4 = classic engine default, 2 = fp16 to a device caller, 0 = host caller).

    ``engines``: the executors this call may use, ``("wave", "classic")`` by default when the wave
    extension is importable (``attention()`` passes what its ``kernel=`` allows and the build supports).
    The wave engine is the executor of every device-resident, device-accumulator call when it is
    available: a W=1 wave is the classic engine's launch per subproblem (equal within 0.6%,
    bit-identical), so the classic is not enumerated there. Its wave budget W is an axis: the
    default budget halved down to one subproblem per wave (``wave_token_ladder``), gathered copies of
    Q/K/V (3-6% faster) at every rung before the in-place rungs, so under a tight budget the wave
    shrinks before the depth grows and in place is chosen only when the copies do not fit. Host-resident
    inputs are enumerated on both engines and the memory model decides (the wave streams a chunk pool
    and folds on a worker; the classic pipelines pinned tasks in less device memory).
    ``allow_mono=False`` plans among the decompositions only (a call whose depth the caller pinned);
    ``min_itr``/``max_itr`` bound the depths enumerated (``min_itr == max_itr`` plans a pinned depth).

    ``inputs_on_host``: the caller's Q/K/V are in host memory. Only host-streaming decompositions are
    executable then (a device-resident candidate would need the inputs on the device, and the engines
    take the residency from the tensors: the wave backward would mix host inputs with device outputs,
    the classic engine would run its CPU reference path), so the others are not enumerated; the
    monolithic call copies the inputs to the device and stays a candidate. A host-resident backward is
    therefore the classic pipeline's (the wave backward keeps its fp32 gradients on the device).

    `quorum_sets` ({c: interest_set}, default QUORUM_SETS) is the fourth axis:
    every (c, itr) pair is a candidate. Measured at N=131K (A100-40GB): for the
    same subproblem size a larger c at a lower depth beats a smaller c at a
    higher depth (c=31 itr=1: 468 ms, 1.40 GiB vs c=7 itr=2: 531 ms, 1.40 GiB;
    c=73 itr=1: 519 ms vs c=13 itr=2: 891 ms), because the pair work is
    (l^2/c)^itr and per-task overhead is c^itr; the model has both terms.
    `direction="bwd"` plans the backward (its own memory estimate and cost) --
    the forward and backward depths are chosen independently.

    Inputs already on the device (the end-to-end model case): ``resident_input_bytes`` is what
    Q/K/V occupy there now. ``budget_excludes_inputs`` says whether the budget was measured with
    them in place (free memory: True) or describes the whole allowance (a hardware dict: False).
    A device-resident candidate counts Q/K/V in its estimate, so when the budget excludes them
    they are credited back. A host-streaming candidate needs them off the device: with
    ``evict_inputs`` their bytes are freed for the computation (the candidate is marked
    ``evict``); without it they stay and are charged on top of the estimate.
    """
    from .native_wave import native_available, estimate_wave_peak_bytes, wave_token_ladder, in_place_possible
    hw = hardware or detect_hardware()
    cm = model or load_cost_model() or CostModel()
    itemsize = torch.empty((), dtype=dtype).element_size()
    if not hw.devices:
        raise RuntimeError("no CUDA device in the hardware description")
    if engines is None:
        engines = ("wave", "classic") if native_available() else ("classic",)
    engines = tuple(engines)
    if not engines or any(e not in ("wave", "classic") for e in engines):
        raise ValueError(f"engines must be a non-empty subset of ('wave', 'classic') (got {engines!r})")
    wave_ok = "wave" in engines
    # Devices fastest first (rate, then budget; stable, so equal devices keep their order). A world-w
    # candidate runs on the w fastest devices: the driver's scheduler gives each rank work in proportion to
    # its rate (so a slower device still helps), and every device must hold the candidate's peak (the
    # budget of a world is the smallest of its devices').
    devs = sorted(hw.devices, key=lambda d: (-(d.tile_rate or 0.0), -d.budget_bytes))
    R = int(resident_input_bytes)
    credit = R if budget_excludes_inputs else 0               # estimates below count Q/K/V; the measured budget did not
    budgets = {w: min(d.budget_bytes for d in devs[:w]) * safety + credit for w in range(1, len(devs) + 1)}
    rel_rates = {w: hw.rel_rates(devs[:w]) for w in budgets}
    dev_budget = budgets[1]
    cands: list[dict] = []

    bwd = direction == "bwd"
    mono = estimate_monolithic_bytes(N, B=B, H=H, D=D, itemsize=itemsize) * (2.2 if bwd else 1.0)
    cands.append(dict(mode="mono", itr=0, c=7, interest_set=(0, 1, 3), acc="gpu", stream_from_host=False, n_par=1, world=1,
                      time=cm.mono_time(N, B, H, causal, direction), peak=mono / GIB, host=0.0,
                      ok=bool(allow_mono and mono <= dev_budget)))
    worlds = [1] + ([w for w in range(2, len(devs) + 1)] if allow_distributed else [])
    qsets = dict(quorum_sets) if quorum_sets is not None else dict(QUORUM_SETS)
    for c, iset in qsets.items():
        l = len(iset)
        if not is_perfect_difference_set(c, iset):
            raise ValueError(f"(c={c}, interest_set={tuple(iset)}) is not a perfect difference set")
        for w in worlds:
            for itr in range(max(1, int(min_itr)), max_itr + 1):
                n_tasks = c ** itr
                if itr * (l - 1) > 63 or (w > 1 and n_tasks < w) or n_tasks > 4096:
                    continue
                for acc in ("gpu", "cpu"):
                    for host in (False, True):
                        if (acc == "cpu" and not host) or (inputs_on_host and not host):
                            continue
                        hostb = _host_bytes(N, B, H, D, itemsize, host, acc) * (2.0 if bwd else 1.0)
                        evict = bool(host and R and evict_inputs)
                        stay = R if (host and R and not evict) else 0      # streaming next to the unevicted device copy
                        base = dict(mode="cqsa" if w == 1 else "cqsa_dist", itr=itr, c=c, interest_set=tuple(iset),
                                    acc=acc, stream_from_host=host, evict=evict, world=w, host=hostb / GIB)
                        bud = budgets[w]
                        # the wave engine: single device; every (acc, host) forward, device-resident backward
                        # (the wave backward keeps its fp32 gradients on the device)
                        wave_here = wave_ok and w == 1 and (not bwd or (not host and acc == "gpu"))
                        if wave_here:
                            # Gathered rungs (3-6% faster) down the whole ladder first; the in-place rungs only when no
                            # gathered rung fits, so the wave shrinks before the copies are given up (at small N one
                            # extra launch costs about what the in-place tax costs, and the tie would otherwise go to
                            # the smaller in-place wave). The chunk pool is always read in place, the backward gathers.
                            if host:
                                modes = (False,)
                            elif bwd or not in_place_possible(N, causal):
                                modes = (True,)
                            else:
                                modes = (True, False)
                            for gather in modes:
                                if not gather and any(x["ok"] for x in cands if x is not None and x.get("engine") == "wave"
                                                      and x["itr"] == itr and x["c"] == c and x["acc"] == acc
                                                      and x["stream_from_host"] == host and x["gather"]):
                                    break
                                for W in wave_token_ladder(N, itr, c, l):
                                    peak = estimate_wave_peak_bytes(N, W, B=B, H=H, D=D, itemsize=itemsize, itr=itr, c=c, l=l,
                                                                    stream_from_host=host, accumulate_on_gpu=(acc == "gpu"),
                                                                    gather=gather, direction=direction, out_bytes_per_el=int(out_bytes_per_el))
                                    t = cm.cqsa_time(N, B, H, D, itr, causal, acc, host, 1, world=w, hw=hw, c=c, l=l, direction=direction,
                                                     engine="wave", wave_tokens=W, gather=gather)
                                    # acc=cpu: the async fold's two pinned slots, each a wave's touched tokens (<= W, <= N)
                                    host_w = hostb + (2 * min(int(W), N) * (H * D * 4 + 8 * H + 8) if acc == "cpu" else 0)
                                    cands.append(dict(base, engine="wave", wave_tokens=int(W), gather=bool(gather), n_par=1,
                                                      time=t, peak=(peak + stay) / GIB, host=host_w / GIB,
                                                      ok=(peak + stay <= bud and host_w <= hw.host_budget_bytes)))
                        if wave_here and not host and acc == "gpu":
                            continue                                       # the wave covers this residency (W=1 == classic)
                        if "classic" not in engines:
                            continue
                        for n_par in n_pars:
                            if n_par > n_tasks:
                                continue
                            est = estimate_peak_bytes_bwd if bwd else estimate_peak_bytes
                            extra = {} if bwd else {"out_bytes_per_el": int(out_bytes_per_el)}
                            peak = est(N, itr, B=B, H=H, D=D, itemsize=itemsize, n_par=n_par, c=c, l=l,
                                       stream_from_host=host, accumulate_on_gpu=(acc == "gpu"), **extra)
                            t = cm.cqsa_time(N, B, H, D, itr, causal, acc, host, n_par, world=w, hw=hw, c=c, l=l, direction=direction,
                                             rates=rel_rates[w])
                            cands.append(dict(base, engine="classic", wave_tokens=0, gather=True, n_par=n_par, time=t,
                                              peak=(peak + stay) / GIB,
                                              ok=(peak + stay <= bud and hostb <= hw.host_budget_bytes)))
    feas = [c for c in cands if c["ok"]]
    if not feas:
        floor = per_el_floor(itemsize, True, False) * N * B * H * D
        # the most frugal configuration that was enumerated (so the subproblem-count cap holds): everything
        # offloaded, one subproblem in flight, smallest predicted peak. An engine restriction can leave no
        # offloaded candidate (the wave backward is device-resident only): then the most frugal of what the
        # restriction allows, which runs and escalates or reports the out-of-memory itself.
        single = [c for c in cands if c["mode"] == "cqsa"]
        off = [c for c in single if c["stream_from_host"] and c["acc"] == "cpu" and c["n_par"] == 1] or single
        if not off:
            raise ValueError(f"no decomposition of N={N} can run with engines={engines}"
                             + (" on host-resident inputs" if inputs_on_host else "")
                             + (f" at itr {min_itr}..{max_itr}" if min_itr > 1 else "")
                             + (": the wave backward is device-resident only; use kernel='auto' or 'cuda'"
                                if bwd and engines == ("wave",) else ""))
        fb = min(off, key=lambda c: (c["peak"], c["time"]))
        p = Plan(mode="cqsa", itr=fb["itr"], c=fb["c"], interest_set=tuple(fb["interest_set"]), direction=direction,
                 acc=fb["acc"], stream_from_host=fb["stream_from_host"], n_par=fb["n_par"], world=1,
                 evict_inputs=bool(fb.get("evict", False)),
                 engine=fb["engine"], wave_tokens=fb["wave_tokens"], gather=fb["gather"],
                 est_time_s=fb["time"], est_peak_gib=fb["peak"], est_host_gib=fb["host"],
                 devices=[d.name for d in devs[:1]], candidates=cands,
                 reason=(f"nothing fits: device budget {dev_budget / GIB:.1f} GiB, host budget {hw.host_budget_bytes / GIB:.0f} GiB; "
                         f"the O(N.H.D) device floor with everything offloaded is {floor / GIB:.1f} GiB. "
                         f"Falling back to the most frugal configuration {_cname(fb)} "
                         f"(~{fb['peak']:.1f} GiB); the engine escalates on its own if that still does not fit."))
        return p
    pick, front = frontier_pick(feas, tol)
    p = Plan(mode=pick["mode"], itr=pick["itr"], c=pick["c"], interest_set=tuple(pick["interest_set"]), direction=direction,
             acc=pick["acc"], stream_from_host=pick["stream_from_host"],
             n_par=pick["n_par"], world=pick["world"], devices=[d.name for d in devs[:pick["world"]]],
             device_rates=[d.tile_rate for d in devs[:pick["world"]]],
             est_time_s=pick["time"], est_peak_gib=pick["peak"], est_host_gib=pick["host"], candidates=cands,
             evict_inputs=bool(pick.get("evict", False)), engine=pick.get("engine", "classic"),
             wave_tokens=int(pick.get("wave_tokens", 0)), gather=bool(pick.get("gather", True)))
    mono_c = cands[0]
    if pick["mode"] == "mono":
        p.reason = (f"monolithic fits ({mono_c['peak']:.2f} GiB <= {dev_budget / GIB:.1f} GiB budget) and is the fastest "
                    f"feasible configuration ({mono_c['time']:.2f} s predicted)")
    else:
        alt = sorted(front, key=lambda c: c["time"])[:3]
        why = (f"monolithic fits ({mono_c['peak']:.1f} GiB) at ~{mono_c['time']:.1f} s on one device, but" if mono_c["ok"]
               else f"monolithic needs {mono_c['peak']:.1f} GiB > {dev_budget / GIB:.1f} GiB budget;")
        p.reason = (f"{why} fastest feasible on the "
                    f"(time, memory) frontier: {p.name()} ~{p.est_time_s:.1f} s at {p.est_peak_gib:.1f} GiB/device"
                    + (f", host {p.est_host_gib:.0f} GiB" if p.est_host_gib else "")
                    + "; frontier: " + "; ".join(f"{_cname(c)} {c['time']:.1f}s/{c['peak']:.1f}GiB" for c in alt))
    return p


def frontier_pick(feas: Sequence[dict], tol: float = 0.02) -> tuple[dict, list[dict]]:
    """The planner's rule on feasible candidates: the (time, peak) Pareto frontier, its fastest point, and among
    the points within ``tol`` of that time the smallest peak. Returns (pick, frontier)."""
    front = [c for c in feas if not any((o["time"] < c["time"]) and (o["peak"] < c["peak"]) for o in feas)]
    best_t = min(c["time"] for c in front)
    near = [c for c in front if c["time"] <= best_t * (1 + tol)]
    return min(near, key=lambda c: c["peak"]), front


def _cname(c: dict) -> str:
    if c["mode"] == "mono":
        return "mono"
    tail = (f"/wave{c['wave_tokens'] >> 10}K{'' if c.get('gather', True) else '/inplace'}" if c.get("engine") == "wave"
            else f"/np{c['n_par']}")
    return f"c{c.get('c', 7)}/itr{c['itr']}/{c['acc']}{'/host' if c['stream_from_host'] else ''}{tail}" + (f"x{c['world']}" if c["world"] > 1 else "")


# ---------------------------------------------------------------------------
# calibration and empirical autotune
# ---------------------------------------------------------------------------

def calibrate(hardware: HardwareSpec | None = None, *, N: int = 131072, B: int = 1, H: int = 8, D: int = 64,
              dtype=torch.float16, causal: bool = True, device="cuda", verbose: bool = True,
              save: bool = True, rate: bool = True) -> CostModel:
    """
    Refit the cost model on this machine: time the monolithic kernel and three
    Stream-CQSA configurations (itr=1 acc=gpu; itr=1 acc=cpu host-resident;
    itr=2 acc=gpu) at a small N and solve for pair_rate, kernel_ratio,
    host-stage cost and depth_factor. About a minute on an A100 at N=131072.
    With ``save`` (and ``rate``) the device's live-tile rate for the multi-device
    scheduler (``scheduler.calibrate_rate``, ~1 s) is cached for this GPU model too.
    """
    from .devkit import Config, run_config, measure
    from .stable_stream import TraceRecorder, stream_cqsa_forward
    cm = CostModel()
    dev = torch.device(device)
    g = torch.Generator(device="cpu").manual_seed(0)
    q, k, v = (torch.randn(B, H, N, D, generator=g, dtype=torch.float32).to(dtype).pin_memory() for _ in range(3))
    scale = float(D) ** -0.5
    _, pm = measure(lambda: run_config(q, k, v, Config(mode="mono"), causal=causal, device=dev), device=dev, reps=2)
    pairs = B * H * N * N * (0.5 if causal else 1.0)
    cm.pair_rate = pairs / pm["s"]
    # itr=1, device accumulator, device-resident inputs: compute-dominated
    tr = TraceRecorder(enabled=True, device=dev)
    qd, kd, vd = (t.to(dev) for t in (q, k, v))
    stream_cqsa_forward(qd, kd, vd, itr=1, causal=causal, allow_escalation=False, max_parallel=1)       # warm
    tr = TraceRecorder(enabled=True, device=dev)
    _, info = stream_cqsa_forward(qd, kd, vd, itr=1, causal=causal, allow_escalation=False, max_parallel=1, trace=tr)
    st = info.get("stage_totals_ms", {})
    L = N * 3 / 7; tokens = 7 * L * (B * H * D) / 512.0
    live_pairs = 7 * B * H * L * L * (0.5 if causal else 1.0) * (1 - (cm.masked_frac if causal else cm.masked_frac * 0.5))
    if st.get("compute"):
        cm.kernel_ratio = (st["compute"] / 1e3) / (live_pairs / cm.pair_rate)
    if st.get("gather"):
        cm.gather_dev_s_per_tok = st["gather"] / 1e3 / tokens
    if st.get("merge"):
        cm.merge_gpu_s_per_tok = st["merge"] / 1e3 / tokens
    del qd, kd, vd
    # itr=1 host accumulator, host-resident inputs
    tr = TraceRecorder(enabled=True, device=dev)
    _, info = stream_cqsa_forward(q, k, v, itr=1, causal=causal, allow_escalation=False, max_parallel=1,
                                  stream_from_host=True, low_memory=True, shared_chunks=True, trace=tr)
    st = info.get("stage_totals_ms", {})
    if st.get("gather"):
        cm.gather_s_per_tok = st["gather"] / 1e3 / tokens
    if st.get("merge"):
        cm.merge_cpu_s_per_tok = st["merge"] / 1e3 / tokens
    if st.get("d2h"):
        cm.d2h_s_per_tok = st["d2h"] / 1e3 / tokens
    # depth factor from itr=2 vs itr=1 (device accumulator)
    qd, kd, vd = (t.to(dev) for t in (q, k, v))
    _, p1 = measure(lambda: stream_cqsa_forward(qd, kd, vd, itr=1, causal=causal, allow_escalation=False, max_parallel=1)[0], device=dev, reps=2)
    _, p2 = measure(lambda: stream_cqsa_forward(qd, kd, vd, itr=2, causal=causal, allow_escalation=False, max_parallel=1)[0], device=dev, reps=2)
    _, p31 = measure(lambda: stream_cqsa_forward(qd, kd, vd, itr=1, causal=causal, allow_escalation=False, max_parallel=1,
                                                 c=31, interest_set=QUORUM_SETS[31])[0], device=dev, reps=2)
    # c=31 and c=7 at itr=1 do the same pair work to within 10% (36/31 vs 9/7, masked 5/36 vs 2/9);
    # the difference is 24 extra tasks' overhead plus 3N more gathered tokens.
    cm.task_overhead_s = 0.0
    base7 = cm.cqsa_time(N, B, H, D, 1, causal, "gpu", False, 1, c=7, l=3)
    base31 = cm.cqsa_time(N, B, H, D, 1, causal, "gpu", False, 1, c=31, l=6)
    cm.task_overhead_s = max(0.0, ((p31["s"] - p1["s"]) - (base31 - base7)) / 24.0)
    cm.depth_factor = 1.0
    pred2 = cm.cqsa_time(N, B, H, D, 2, causal, "gpu", False, 1, c=7, l=3)
    pred1 = cm.cqsa_time(N, B, H, D, 1, causal, "gpu", False, 1, c=7, l=3)
    cm.depth_factor = max(1.0, (p2["s"] / p1["s"]) / (pred2 / max(pred1, 1e-9)))
    if verbose:
        print(f"calibrate: mono {pm['s']:.3f}s -> pair_rate {cm.pair_rate:.2e}/s; kernel_ratio {cm.kernel_ratio:.2f}; "
              f"gather {cm.gather_s_per_tok*1e9:.1f} ns/tok (host) {cm.gather_dev_s_per_tok*1e9:.1f} (dev); "
              f"merge {cm.merge_cpu_s_per_tok*1e9:.1f} ns/tok (cpu) {cm.merge_gpu_s_per_tok*1e9:.1f} (gpu); "
              f"d2h {cm.d2h_s_per_tok*1e9:.1f} ns/tok; task_overhead {cm.task_overhead_s*1e3:.2f} ms (c=31 vs 7); itr2/itr1 {p2['s']/p1['s']:.2f} -> depth_factor {cm.depth_factor:.2f}")
    if save:
        # persisted per GPU model: plan() picks it up automatically from now on
        path = save_cost_model(cm, device)
        if verbose:
            print(f"calibrate: saved to {path} (plan() uses it automatically on this GPU model)")
        if rate:
            # the multi-device scheduler's live-tile rate of this GPU model (DeviceSpec.tile_rate, the rank table)
            from .scheduler import calibrate_rate, save_device_rate
            dr = calibrate_rate(dev)
            rpath = save_device_rate(dr)
            if verbose:
                print(f"calibrate: live-tile rate {dr.tile_rate:.3e} tiles/s saved to {rpath}")
    return cm


def autotune(*, N: int, B: int = 1, H: int = 8, D: int = 64, dtype=torch.float16, causal: bool = True,
             hardware: HardwareSpec | None = None, max_candidates: int = 6, device="cuda", verbose: bool = True) -> Plan:
    """
    Plan, then MEASURE the top feasible single-device candidates on the real
    shape and pick by the same rule from measurements. Use when one call is
    cheap enough to run a few times (say under a minute) and will be repeated.
    """
    from .devkit import Config, quick_bench
    p = plan(N=N, B=B, H=H, D=D, dtype=dtype, causal=causal, hardware=hardware, engines=("classic",))   # quick_bench runs the classic configs
    feas = [c for c in p.candidates if c["ok"] and c["world"] == 1]
    feas.sort(key=lambda c: c["time"])
    cfgs = []
    for c in feas[:max_candidates]:
        cfgs.append(Config(mode="mono") if c["mode"] == "mono" else
                    Config(mode="cqsa", itr=c["itr"], c=c["c"], interest_set=tuple(c["interest_set"]), acc=c["acc"],
                           stream_from_host=c["stream_from_host"], n_par=c["n_par"]))
    hw = hardware or detect_hardware()
    budget = min(d.budget_bytes for d in hw.devices) / GIB
    res = quick_bench(N=N, B=B, H=H, D=D, dtype=dtype, causal=causal, configs=cfgs, budget_gib=budget,
                      device=device, verbose=verbose, acc_rows=64, reps=1)
    pick = res["pick"]
    if pick is None:
        return p
    cfg = pick["cfg"]
    return Plan(mode=cfg["mode"], itr=cfg["itr"], c=cfg.get("c", 7), interest_set=tuple(cfg.get("interest_set", (0, 1, 3))),
                acc=cfg["acc"], stream_from_host=cfg["stream_from_host"],
                n_par=cfg["n_par"], world=1, devices=[hw.devices[0].name], est_time_s=pick["s"],
                est_peak_gib=pick["peak_gib"], candidates=p.candidates,
                reason=f"measured: {pick['config']} {pick['s']:.3f} s at {pick['peak_gib']:.2f} GiB (rule applied to measurements)")


# ---------------------------------------------------------------------------
# one-call entry point
# ---------------------------------------------------------------------------

def auto_attention(q, k, v, *, causal: bool = False, scale: float | None = None,
                   hardware: HardwareSpec | dict | None = None, model: CostModel | None = None,
                   plan_only: bool = False, verbose: bool = False, **overrides):
    """
    Plan for this call's shape and hardware, then run it. Returns (out, plan).
    Distributed plans run only inside an initialised torch.distributed group of
    the planned size; otherwise the single-device frontier point is used.
    `overrides` are forwarded to the engine (e.g. allow_escalation=False).
    """
    import warnings
    warnings.warn("stream_cqsa.autoconfig.auto_attention is a legacy entry point; use stream_cqsa.attention(q, k, v, "
                  "is_causal=..., hardware=..., return_plan=True)", DeprecationWarning, stacklevel=2)
    from .stable_stream import stream_cqsa_forward
    if isinstance(hardware, dict):
        hardware = hardware_from_dict(hardware)
    B, H, N, D = q.shape
    p = plan(N=N, B=B, H=H, D=D, dtype=q.dtype, causal=causal, hardware=hardware, model=model, engines=("classic",))
    if p.mode == "cqsa_dist":
        import torch.distributed as dist
        if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() == p.world):
            single = [c for c in p.candidates if c["ok"] and c["world"] == 1]
            if single:
                best = min(single, key=lambda c: c["time"])
                p.reason += f" | not in a {p.world}-rank process group: using the single-device point {_cname(best)}"
                p.mode, p.itr, p.c, p.interest_set, p.acc, p.stream_from_host, p.n_par, p.world = \
                    best["mode"], best["itr"], best["c"], tuple(best["interest_set"]), best["acc"], best["stream_from_host"], best["n_par"], 1
    if verbose:
        print(f"auto_attention: {p.name()} -- {p.reason}")
    if plan_only:
        return None, p
    if p.mode == "mono":
        from .interface import flash_attn_func
        dev = torch.device(p.devices[0])
        qq, kk, vv = (t.to(dev, non_blocking=True) for t in (q, k, v))
        out = flash_attn_func(qq.transpose(1, 2), kk.transpose(1, 2), vv.transpose(1, 2),
                              softmax_scale=scale, causal=causal).transpose(1, 2)
        return out, p
    kw = p.engine_kwargs(); kw.update(overrides)
    if p.stream_from_host:
        qq, kk, vv = (t if t.device.type == "cpu" else t.cpu() for t in (q, k, v))
    else:
        dev = torch.device(p.devices[0]); qq, kk, vv = (t.to(dev, non_blocking=True) for t in (q, k, v))
    if p.mode == "cqsa_dist":
        from .distributed import dist_stream_cqsa_forward
        out, _ = dist_stream_cqsa_forward(qq, kk, vv, causal=causal, scale=scale, **kw)
    else:
        kw.setdefault("verbose", verbose)
        out, _ = stream_cqsa_forward(qq, kk, vv, causal=causal, scale=scale, **kw)
    return out, p
