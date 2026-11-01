"""
The one entry point: ``stream_cqsa.attention`` with the signature of
``torch.nn.functional.scaled_dot_product_attention``.

    out = stream_cqsa.attention(q, k, v, is_causal=True)

* below the memory boundary it IS the monolithic call (SDPA / FlashAttention);
* above it, the planner picks the decomposition (depth, quorum set, accumulator
  placement, host streaming, concurrency) from the free memory and the call
  runs exactly, on the best kernel available (native wave kernel, CUDA
  extension, or the Triton kernels when nothing is compiled);
* gradients flow when the inputs require them;
* ``verbose=True`` (or ``CQSA_VERBOSE=1``) shows what is happening.

``patch_sdpa()`` swaps this in for ``F.scaled_dot_product_attention`` so an
existing model runs unchanged; ``estimate(N, ...)`` is the dry run that says
what a call would cost before it is made.
"""
from __future__ import annotations

import contextlib
import os
import warnings
from typing import Any, Optional

import torch
import torch.nn.functional as F

from .progress import verbose_enabled

__all__ = ["attention", "estimate", "patch_sdpa", "unpatch_sdpa", "patched_sdpa", "kernels_available"]


def kernels_available() -> dict[str, bool]:
    """Which attention kernels this environment can run."""
    from . import interface as I
    out = {"cuda_extension": I.cqsa_cuda is not None, "cuda_extension_noncausal": I.cqsa_cuda_noncausal is not None,
           # a v13-class module (census + ALIGNED forward and backward in one extension; its CUDA backward is the default)
           "cuda_extension_unified": I.CQSA_CUDA_UNIFIED}
    try:
        from .native_wave import native_available
        out["native_wave"] = native_available()
    except Exception:
        out["native_wave"] = False
    try:
        import triton  # noqa: F401
        from . import triton_kernel  # noqa: F401
        out["triton"] = True
    except Exception:
        out["triton"] = False
    try:
        import flash_attn  # noqa: F401
        out["flash_attn"] = True
    except Exception:
        out["flash_attn"] = False
    return out


def _check_inputs(q, k, v, attn_mask, dropout_p, enable_gqa):
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(f"stream_cqsa.attention expects q/k/v as [B, H, N, D]; got {tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}")
    if attn_mask is not None:
        raise NotImplementedError("stream_cqsa.attention supports the plain and the causal (is_causal=True) cases; an explicit "
                                  "attn_mask is not supported. For FlexAttention-style masks use stream_cqsa.adapters.flex_inner.")
    if dropout_p:
        raise NotImplementedError("stream_cqsa.attention does not support dropout (dropout_p must be 0).")
    if k.shape != v.shape:
        raise ValueError(f"k and v must have the same shape [B, H_kv, N, D] (got {tuple(k.shape)} and {tuple(v.shape)})")
    if (q.shape[0], q.shape[2], q.shape[3]) != (k.shape[0], k.shape[2], k.shape[3]):
        raise ValueError("q and k/v must agree in [B, ., N, D] (cross-attention with N_q != N_k is not supported); "
                         f"got {tuple(q.shape)} and {tuple(k.shape)}")
    if k.shape[1] != q.shape[1] and (k.shape[1] < 1 or q.shape[1] % k.shape[1] != 0):
        raise ValueError(f"grouped-query attention needs the K/V heads to divide the query heads (got {q.shape[1]} and {k.shape[1]})")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"stream_cqsa.attention needs fp16 or bf16 inputs (got {q.dtype}); cast with .half() or .bfloat16(). "
                        "Accumulation is fp32 internally regardless.")
    if not (q.dtype == k.dtype == v.dtype):
        raise TypeError("q, k and v must share a dtype")
    if not (q.device == k.device == v.device):
        raise ValueError("q, k and v must be on the same device")


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask=None, dropout_p: float = 0.0,
              is_causal: bool = False, scale: Optional[float] = None, enable_gqa: bool = False, *,
              verbose: Optional[bool] = None, hardware=None, kernel: str = "auto", plan_only: bool = False,
              return_plan: bool = False, return_info: bool = False, **overrides) -> torch.Tensor:
    """
    Exact attention that always fits. Drop-in for ``F.scaled_dot_product_attention``
    (same positional signature; masks and dropout are rejected with a message). Grouped-query /
    multi-query attention: ``k``/``v`` with ``H_kv`` heads dividing ``q``'s ``H`` (query head ``h``
    attends K/V head ``h // (H // H_kv)``), with or without ``enable_gqa=True``; the CUDA kernels
    (classic and wave, both directions) handle it natively, the Triton kernels reject it.

    q/k/v ``[B, H, N, D]`` fp16/bf16, on a device or in host memory. Returns the
    output in the inputs' dtype and device. Differentiable when the inputs require
    gradients. ``kernel``: "auto" (the planner routes by the memory model: the monolithic call
    whenever it fits -- it is the fastest configuration and needs less device memory than any
    device-resident decomposition, so with nothing pinned a device-resident wave runs only after the
    monolithic call ran out of memory after all or under a pinned ``itr=``/``c=``; otherwise the wave
    engine for device-resident device-accumulator calls whenever its wave fits, with the wave budget
    W and gathered-vs-in-place reads chosen from the budget; host-resident inputs are streamed, by
    whichever engine the memory model prefers for a forward and by the classic pinned pipeline for
    a backward and for host-accumulator backwards), "wave" (wave engine, CUDA kernel when built else
    Triton), "wave-cuda", "wave-triton", "cuda", "triton" (classic engine). The wave backward is
    device-resident only: a host-resident backward runs on the classic engine whatever ``kernel`` says.
    ``overrides`` (itr=, c=, interest_set=, max_parallel=, max_wave_tokens=, gather=, ...) pin engine settings.

    Where the accumulator lives is the planner's choice by the (time, memory) frontier: among the feasible
    configurations predicted within ``tol`` (default 0.15, ``CQSA_PLAN_TOL``) of the fastest one, the one with the
    smallest device peak runs. On an A100 that means the host accumulator (``acc=cpu``: the forward's fp32 output
    and the backward's fp32 gradients accumulate in host memory, 40-70% less device memory) from about 1M tokens
    up, where it costs 5-15% of the wall time, and the device accumulator below, where it would cost more.
    ``tol=0`` takes the fastest configuration whatever it holds on the device; ``accumulate_on_gpu=True`` /
    ``low_memory=False`` pin the device accumulator, ``accumulate_on_gpu=False`` / ``low_memory=True`` the host one.
    ``plan_only=True`` returns the plan without computing anything; ``return_info=True`` reports the
    engine, kernel, wave budget and gather mode that ran.

    Inside a model (Q/K/V already on the device, little room left): ``evict_inputs="auto"``
    (default) releases Q/K/V to pinned host memory for the computation when nothing fits next
    to them -- the caller's tensors are rebound, so the device memory is really freed -- and
    streams from there. The output is placed on the device first; then ``restore_inputs="auto"``
    brings Q/K/V back if room remains (otherwise they stay valid host tensors). ``True`` /
    ``False`` force or forbid either step. Gradients still arrive on the device. Memory is only
    freed if nothing else references the storage (e.g. ``del`` a fused QKV tensor first);
    ``return_info=True`` reports ``inputs_evicted``, ``evict_released_gib``, ``inputs_restored``.
    """
    _INSIDE[0] += 1
    try:
        return _attention(q, k, v, attn_mask, dropout_p, is_causal, scale, enable_gqa, verbose=verbose, hardware=hardware,
                          kernel=kernel, plan_only=plan_only, return_plan=return_plan, return_info=return_info, **overrides)
    finally:
        _INSIDE[0] -= 1


def _config_line(*, stage: str, mode: str, itr=None, c=None, n_sub=None, qkv: str, acc: str, engine: str, kernel: str,
                 n_par=None, wave: Optional[str] = None, out: str) -> str:
    """One uniform line that states every automatic choice (printed with verbose=True)."""
    dec = "monolithic (one kernel call, no decomposition)" if mode == "mono" else f"decomposed: itr={itr}, c={c}, {n_sub} subproblems"
    parts = [f"Stream-CQSA {stage}: {dec}", f"Q/K/V on {qkv}", f"accumulator on {acc}", f"engine {engine}, kernel {kernel}"]
    if wave:
        parts.append(wave)
    if n_par:
        parts.append(f"{n_par} in flight")
    parts.append(f"output to {out}")
    return " | ".join(parts)


def place_inputs(*tensors: torch.Tensor, device="cuda", reserve_fraction: float = 0.15, pin: bool = True,
                 verbose: Optional[bool] = None):
    """
    Guardrail for a crowded device: move host tensors to ``device`` only if they fit, and keep
    the rest in (pinned) host memory, where ``attention`` streams them from. A tensor is moved
    when the free device memory minus ``reserve_fraction`` of the total still holds it; an
    out-of-memory during the move is caught and the tensor stays on the host. Device tensors
    are returned unchanged. Returns the tensors in the order given.

        q, k, v = stream_cqsa.place_inputs(q, k, v)          # each on the GPU if it fits, else host (pinned)
        out = stream_cqsa.attention(q, k, v, is_causal=True)
    """
    vb = verbose_enabled(verbose)
    dev = torch.device(device)
    out = []
    for i, t in enumerate(tensors):
        if t.is_cuda or dev.type != "cuda" or not torch.cuda.is_available():
            out.append(t); continue
        free, total = torch.cuda.mem_get_info(dev)
        need = t.numel() * t.element_size()
        moved = None
        if need <= free - reserve_fraction * total:
            try:
                moved = t.to(dev)
            except Exception as exc:                                     # noqa: BLE001
                if not _is_oom(exc):
                    raise
                torch.cuda.empty_cache()
        if moved is None:
            if pin and not t.is_pinned():
                try:
                    t = t.pin_memory()
                except Exception:                                        # noqa: BLE001
                    pass
            if vb:
                print(f"Stream-CQSA place_inputs: tensor {i} ({need / 2**30:.2f} GiB) stays in host memory "
                      f"({free / 2**30:.2f} GiB free on {dev}); attention() will stream it", flush=True)
            out.append(t)
        else:
            if vb:
                print(f"Stream-CQSA place_inputs: tensor {i} ({need / 2**30:.2f} GiB) -> {dev}", flush=True)
            out.append(moved)
    return tuple(out) if len(out) != 1 else out[0]


def _pinned_copy(t: torch.Tensor) -> torch.Tensor:
    """Host copy of a [B, H, N, D] device tensor, stored token-major ([B, N, H, D] in memory):
    the layout the streaming engine gathers from, so it makes no second host copy."""
    B, H, N, D = t.shape
    try:
        h = torch.empty((B, N, H, D), dtype=t.dtype, device="cpu", pin_memory=True)
    except Exception:                                                # noqa: BLE001  (pinned memory exhausted / unavailable)
        h = torch.empty((B, N, H, D), dtype=t.dtype, device="cpu")
    h = h.transpose(1, 2)
    h.copy_(t.detach())
    return h


class _ToHost(torch.autograd.Function):
    """Differentiable move to pinned host memory; the gradient returns to the source device."""
    @staticmethod
    def forward(ctx, t):
        ctx.dev = t.device
        return _pinned_copy(t)

    @staticmethod
    def backward(ctx, g):
        return g.to(ctx.dev)


def _evict_inputs(tensors, grad: bool, vb: bool):
    """Move the caller's device tensors to pinned host memory and RELEASE the device copies.

    A plain ``.cpu()`` copies and leaves the device allocation alive behind the caller's own
    reference. Here each tensor object is rebound onto the host copy (``t.data = host``), so the
    device storage is freed unless something else still references it (a fused QKV projection
    the slices are views of, an activation another op saved). What was actually released is
    measured, not assumed. Returns (host_tensors, state); with ``grad`` the host tensors are
    differentiable copies and the gradients come back on the device."""
    dev = tensors[0].device
    torch.cuda.synchronize(dev)
    before = torch.cuda.memory_allocated(dev)
    seen, expected = set(), 0
    for t in tensors:
        st = t.untyped_storage()
        if st.data_ptr() not in seen:
            seen.add(st.data_ptr()); expected += st.nbytes()
    del st                                     # a storage handle keeps the device allocation alive
    strides = [t.stride() for t in tensors]    # restored tensors get the caller's layout back
    hosts = []
    for t in tensors:
        h = _ToHost.apply(t) if (grad and t.requires_grad) else _pinned_copy(t)
        t.data = h.detach()                    # shares the host copy: no second host allocation
        hosts.append(h)
    torch.cuda.synchronize(dev)
    torch.cuda.empty_cache()
    freed = before - torch.cuda.memory_allocated(dev)
    state = {"device": dev, "tensors": list(tensors), "strides": strides, "freed": int(freed), "expected": int(expected),
             "restored": None}
    if vb:
        print(f"Stream-CQSA: evicted Q/K/V to pinned host memory, released {freed / 2**30:.2f} GiB on {dev}", flush=True)
    if freed < 0.5 * expected:
        warnings.warn(f"Stream-CQSA: evicting Q/K/V released only {freed / 2**30:.2f} of {expected / 2**30:.2f} GiB on {dev}: "
                      "something else still references their storage (a fused QKV tensor they are views of, or an activation "
                      "saved by another op). Drop that reference before the call (e.g. `del qkv`) to free the memory.",
                      RuntimeWarning, stacklevel=4)
    return hosts, state


def _restore_inputs(state, mode, vb: bool, reserve_fraction: float = 0.10, budget_bytes: Optional[int] = None) -> bool:
    """After the output is in place: bring Q/K/V back to the device if there is room (mode "auto"),
    always try (True) or never (False). The output has priority, so this runs last. Tensors that
    do not fit stay in host memory (``t.device`` tells the caller where each one is).

    Room is judged on the memory this process may use, not the physical device: the free memory
    under a set_per_process_memory_fraction cap, less ``reserve_fraction`` of the capped capacity,
    and within ``budget_bytes`` (what a ``hardware=`` budget leaves after the output) when given.
    A reserve of the physical total never restored anything under a small cap or budget."""
    from .stable_stream import effective_free_bytes, effective_total_bytes
    dev = state["device"]
    if mode is False:
        state["restored"] = False
        return False
    ok = True
    left = budget_bytes
    for t, stride in zip(state["tensors"], state["strides"]):
        if t.device == dev:
            continue
        need = t.numel() * t.element_size()
        room = effective_free_bytes(dev) - reserve_fraction * effective_total_bytes(dev)
        if mode == "auto" and (need > room or (left is not None and need > left)):
            ok = False
            continue
        try:
            d = torch.empty_strided(t.shape, stride, dtype=t.dtype, device=dev)
            d.copy_(t.data, non_blocking=True)
            torch.cuda.synchronize(dev)
            t.data = d
            if left is not None:
                left -= need
        except Exception as exc:                                     # noqa: BLE001
            if not _is_oom(exc):
                raise
            torch.cuda.empty_cache(); ok = False
    state["restored"] = ok
    if vb:
        print("Stream-CQSA: Q/K/V restored to " + str(dev) if ok else
              f"Stream-CQSA: Q/K/V left in host memory (no room on {dev} next to the output); they remain valid tensors", flush=True)
    return ok


def _attention(q, k, v, attn_mask, dropout_p, is_causal, scale, enable_gqa, *, verbose, hardware, kernel, plan_only,
               return_plan, return_info=False, **overrides):
    """``return_info=True`` appends the engine's info dict (engine, kernel, itr, subproblem count,
    OOM retries, ...) to the return value: (out[, plan], info)."""
    def _ret(out, p, info):
        r = (out, p) if return_plan else out
        return ((r + (info,)) if return_plan else (out, info)) if return_info else r
    from .autoconfig import plan as _plan, hardware_from_dict, detect_hardware
    _check_inputs(q, k, v, attn_mask, dropout_p, enable_gqa)
    B, H, N, D = q.shape
    Hk = int(k.shape[1])                                  # K/V heads (grouped-query attention: a divisor of H)
    if scale is None:
        scale = float(D) ** -0.5
    on_cuda = q.device.type == "cuda"
    out_device = q.device
    hw = hardware_from_dict(hardware) if isinstance(hardware, dict) else (hardware or detect_hardware())
    direction = "bwd" if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad) else "fwd"
    engines = _engines_for(kernel, q.dtype, D, direction, on_host=not on_cuda)
    out_el = q.element_size() if on_cuda else 0          # attention() delivers fp16 to a device caller, nothing to a host caller
    # Inputs already on the device (the end-to-end model case). evict_inputs: "auto" moves them to
    # host memory for the computation when nothing device-resident fits, True always, False never.
    # restore_inputs: after the output is on the device, "auto" brings them back if there is room.
    evict = overrides.pop("evict_inputs", "auto")
    restore = overrides.pop("restore_inputs", "auto")
    if evict not in ("auto", True, False) or restore not in ("auto", True, False):
        raise ValueError("evict_inputs and restore_inputs take 'auto', True or False")
    vb = verbose_enabled(verbose)
    ev = {"state": None}                                  # eviction state, shared with _run (survives an exception there)
    resident = (sum(t.numel() * t.element_size() for t in (q, k, v)) if on_cuda else 0)
    # inputs_on_host: only host-streaming decompositions can run on host tensors (the planner skips the rest)
    pkw = dict(resident_input_bytes=resident, budget_excludes_inputs=hardware is None,
               evict_inputs=bool(evict) and on_cuda, inputs_on_host=not on_cuda, Hk=Hk)
    if "tol" in overrides:                                # the planner's frontier tolerance (see autoconfig.PLAN_TOL_DEFAULT)
        pkw["tol"] = float(overrides.pop("tol"))
    if evict is True and on_cuda and not plan_only:
        (q, k, v), ev["state"] = _evict_inputs((q, k, v), direction == "bwd", vb)
        on_cuda = False
        pkw = dict(resident_input_bytes=0, inputs_on_host=True, Hk=Hk, **({"tol": pkw["tol"]} if "tol" in pkw else {}))
        engines = _engines_for(kernel, q.dtype, D, direction, on_host=True)
        if hardware is None:
            hw = detect_hardware()
    pkw["engines"] = engines
    p = _plan(N=N, B=B, H=H, D=D, dtype=q.dtype, causal=bool(is_causal), hardware=hw, direction=direction,
              allow_distributed=False, out_bytes_per_el=out_el, **pkw)
    pinned = {k_: overrides[k_] for k_ in ("itr", "c", "interest_set") if k_ in overrides and overrides[k_] != "auto"}
    if overrides and (p.mode == "mono" or pinned):
        # The caller pinned a decomposition (itr=, c=, ...), or any override under a plan that would have
        # been monolithic: the engine, wave budget and gather mode still come from the memory model, among
        # the decompositions of the pinned quorum set (c=7 unless given) at the pinned depth, so plan_only,
        # the reason line and W describe the configuration that runs.
        from .autoconfig import QUORUM_SETS
        c_pin = int(overrides.get("c", 7))
        iset = tuple(overrides.get("interest_set", QUORUM_SETS.get(c_pin, (0, 1, 3))))
        depth = {"min_itr": int(pinned["itr"]), "max_itr": int(pinned["itr"])} if "itr" in pinned else {}
        p = _plan(N=N, B=B, H=H, D=D, dtype=q.dtype, causal=bool(is_causal), hardware=hw, direction=direction,
                  allow_distributed=False, out_bytes_per_el=out_el, allow_mono=False, quorum_sets={c_pin: iset},
                  **depth, **pkw)
    if on_cuda and p.mode == "cqsa" and p.stream_from_host:
        _prefer_device_resident(p)
    if plan_only:
        return p
    out = None
    try:
        out, info = _run(q, k, v, p, is_causal=is_causal, scale=scale, direction=direction, on_cuda=on_cuda, out_el=out_el,
                         overrides=overrides, kernel=kernel, verbose=verbose, vb=vb, evict=evict, ev=ev,
                         out_device=out_device, pkw=pkw)
    finally:
        evicted = ev["state"]
        if evicted is not None and evicted["restored"] is None:
            # a hardware= budget describes the whole allowance: Q/K/V return within what the output leaves of it
            budget = None if hardware is None else next(
                (d.budget_bytes for d in hw.devices if torch.device(d.name) == evicted["device"]), None)
            if budget is not None and out is not None and out.is_cuda:
                budget -= out.numel() * out.element_size()
            try:
                _restore_inputs(evicted, restore, vb, budget_bytes=budget)   # the output is delivered first; Q/K/V only if room remains
            except Exception:                                        # noqa: BLE001
                pass
    if evicted is not None:
        info = dict(info, inputs_evicted=True, inputs_restored=bool(evicted["restored"]),
                    evict_released_gib=round(evicted["freed"] / 2**30, 3))
    else:
        info = dict(info, inputs_evicted=False)
    return _ret(out, p, info)


def _prefer_device_resident(p) -> bool:
    """The inputs are on the device but the plan streams them from the host: take instead the planner's own
    pick (the frontier rule) among the candidates that leave them where they are, if any fits. The plan is
    updated in place; False when nothing device-resident fits (the caller evicts or streams next to them)."""
    from .autoconfig import frontier_pick
    dev_ok = [c for c in p.candidates if c["ok"] and c["mode"] == "cqsa" and not c["stream_from_host"]]
    if not dev_ok:
        return False
    c0, _ = frontier_pick(dev_ok, getattr(p, "tol", None))
    p.itr, p.c, p.interest_set, p.acc, p.stream_from_host, p.n_par = \
        int(c0["itr"]), int(c0["c"]), tuple(c0["interest_set"]), c0["acc"], False, int(c0["n_par"])
    p.engine, p.wave_tokens, p.gather = c0.get("engine", "classic"), int(c0.get("wave_tokens", 0)), bool(c0.get("gather", True))
    p.est_time_s, p.est_peak_gib, p.est_host_gib, p.evict_inputs = c0["time"], c0["peak"], c0["host"], False
    p.reason += f" | Q/K/V are on the device: keeping them there with {p.name()}"
    return True


def _run(q, k, v, p, *, is_causal, scale, direction, on_cuda, out_el, overrides, kernel, verbose, vb, evict, ev,
         out_device, pkw):
    """Execute the plan; returns (output on the caller's device, info)."""
    from .autoconfig import plan as _plan, detect_hardware
    B, H, N, D = q.shape

    # ---- monolithic: the call fits, so it IS the normal kernel ------------------
    if p.mode == "mono" and not overrides:
        if vb:
            print(f"Stream-CQSA: N={N} fits ({p.est_peak_gib:.1f} GiB)", flush=True)
            print(_config_line(stage="plan", mode="mono", qkv=("device" if on_cuda else "host, copied to the device"),
                               acc="device", engine="monolithic", kernel="sdpa", out=f"{out_device} {str(q.dtype).replace('torch.', '')}"), flush=True)
        qd, kd, vd = ((t if on_cuda else t.to("cuda", non_blocking=True)) for t in (q, k, v))
        try:
            out = _TRUE_SDPA(qd, kd, vd, is_causal=bool(is_causal), scale=scale,
                             **({"enable_gqa": True} if kd.shape[1] != qd.shape[1] else {}))
            out = out if out.device == out_device else out.to(out_device)     # evicted inputs: the caller's device
            return out, {"engine": "monolithic", "kernel": "sdpa", "itr": 0, "n_subproblems": 1}
        except Exception as exc:                                     # noqa: BLE001
            if not _is_oom(exc):
                raise
            if vb:
                print("Stream-CQSA: the monolithic call ran out of memory after all; decomposing", flush=True)
            del qd, kd, vd
            torch.cuda.empty_cache()
            # the planner may still think it fits: plan among the decompositions only, so the engine,
            # wave budget and gather mode come from the memory model as for a predicted OOM
            p = _plan(N=N, B=B, H=H, D=D, dtype=q.dtype, causal=bool(is_causal), hardware=detect_hardware(),
                      direction=direction, allow_distributed=False, out_bytes_per_el=out_el, allow_mono=False,
                      **dict(pkw, budget_excludes_inputs=True))
            if on_cuda and p.stream_from_host:
                _prefer_device_resident(p)

    # ---- decomposed ------------------------------------------------------------
    kw = p.engine_kwargs()
    if on_cuda and kw.get("stream_from_host"):
        # The inputs are on the device and nothing device-resident fits (_prefer_device_resident found
        # no candidate): stream from the host, with a pinned host copy -- released from the device
        # when evicting, otherwise next to the caller's device tensors.
        if evict:
            # Nothing fits next to Q/K/V: release them to host memory for the computation. The
            # output goes to the device first afterwards; Q/K/V return only if room remains.
            (q, k, v), ev["state"] = _evict_inputs((q, k, v), direction == "bwd", vb)
            on_cuda = False
            if not kw.get("stream_from_host"):
                kw.update(stream_from_host=True)
        elif direction == "bwd":
            # Copying the inputs to the host would sever the autograd graph. Keep them on the
            # device, park the accumulator in host memory, and let the engine escalate the
            # depth on its own if the device is still short.
            if vb:
                print("Stream-CQSA: no device-resident configuration fits the estimate; keeping Q/K/V on the device with a "
                      "host accumulator and escalating on demand", flush=True)
            kw.update(stream_from_host=False, low_memory=True, accumulate_on_gpu=False, shared_chunks=False)
        else:
            if vb:
                print("Stream-CQSA: no device-resident configuration fits; copying Q/K/V to pinned host memory and streaming", flush=True)
            q, k, v = (_pinned_copy(t) for t in (q, k, v))                # token-major: the engine streams it as is
            on_cuda = False
    kw.update(overrides)
    bwd_itr = kw.pop("bwd_itr", "auto")          # depth of the backward: "auto" (planned on its own), "fwd", or an int
    if not kw.get("stream_from_host"):
        kw["shared_chunks"] = False
    acc_gpu = not (kw.get("low_memory") or kw.get("accumulate_on_gpu") is False)
    # the plan's engine stands only for the residency it was planned for: an override that moves the
    # accumulator or the inputs elsewhere is an unplanned combination, which the classic pipeline runs
    planned = p.engine if (acc_gpu == (p.acc == "gpu") and bool(kw.get("stream_from_host")) == bool(p.stream_from_host)) else "classic"
    use_wave = _pick_wave(kernel, q, direction, engine=planned)
    wave_tokens = int(kw.pop("max_wave_tokens", p.wave_tokens) or 0) or None
    gather = kw.pop("gather", p.gather)
    if vb:
        print(f"Stream-CQSA: {p.reason}", flush=True)
        _acc = "host (CPU)" if not acc_gpu else "device (GPU)"
        _qkv = ("device -> evicted to host, streamed" if ev["state"] else "host, streamed") if kw.get("stream_from_host") else "device"
        _itr = kw.get("itr", "auto"); _c = int(kw.get("c", 7))
        _nsub = (_c ** int(_itr)) if isinstance(_itr, int) else "planned per call"
        if use_wave:
            from .native_wave import resolve_wave_kernel
            _eng, _ker = "wave", resolve_wave_kernel(WAVE_KERNELS.get(kernel, "auto"), q.dtype, int(q.shape[-1]))
        else:
            _eng, _ker = "classic", ("triton" if (kernel in ("triton", "wave-triton") or os.environ.get("CQSA_FORWARD", "").lower() == "triton") else "cuda")
        print(_config_line(stage="plan", mode="cqsa", itr=_itr, c=_c, n_sub=_nsub, qkv=_qkv, acc=_acc, engine=_eng, kernel=_ker,
                           wave=(_wave_desc(wave_tokens, gather, kw.get("stream_from_host")) if use_wave else None),
                           n_par=(None if use_wave else kw.get("max_parallel")),
                           out=f"{out_device} {str(q.dtype).replace('torch.', '')}"), flush=True)
    if use_wave:
        from .native_wave import wave_attention, wave_forward, wave_token_floor, in_place_possible, default_wave_tokens
        wkw = dict(itr=int(kw.get("itr", 1)), c=int(kw.get("c", 7)), interest_set=tuple(kw.get("interest_set", (0, 1, 3))))
        # host-resident: chunk-pool slots (default: what the free memory allows); the engine's knob, not the planner's
        wpool = {"pool_slots": int(kw["pool_slots"])} if kw.get("pool_slots") is not None else {}
        wkern = WAVE_KERNELS.get(kernel, "auto")
        floor = wave_token_floor(N, H=H, D=D, itemsize=q.element_size(), **wkw)
        retries = 0
        while use_wave:
            try:
                if direction == "bwd":
                    # the backward runs in loss.backward(), outside this ladder: it halves its own wave budget
                    # on an OOM and, under kernel="auto", falls back to the classic streamed backward; its
                    # record lands in info["backward"] once it has run
                    binfo: dict = {}
                    out = wave_attention(q, k, v, causal=bool(is_causal), scale=scale, verbose=verbose, max_wave_tokens=wave_tokens,
                                         gather=gather, bwd_itr=(bwd_itr if isinstance(bwd_itr, int) else None),
                                         classic_fallback=kernel not in WAVE_KERNELS, bwd_info=binfo, **wkw, **wpool)
                    info = {"engine": "wave", "kernel": wkern, "max_wave_tokens": wave_tokens,
                            "gather": binfo["forward_gather"], "backward": binfo, **wkw}
                else:
                    out, info = wave_forward(q, k, v, causal=bool(is_causal), scale=scale, verbose=verbose, max_wave_tokens=wave_tokens,
                                             gather=gather, accumulate_on_gpu=acc_gpu, kernel=wkern, **wkw, **wpool)
                    info = {k_: v_ for k_, v_ in info.items() if k_ != "lse"}; info["engine"] = "wave"
                info["wave_oom_retries"] = retries
                break
            except Exception as exc:                                     # noqa: BLE001
                if not _is_oom(exc):
                    raise
                torch.cuda.empty_cache()
                retries += 1
                # OOM inside a wave: halve the wave budget first (the per-wave term is the only one the budget
                # moves), then drop the gathered copies, and only then leave the wave engine, which keeps its
                # accumulator on the device, for the classic pipeline (it escalates the depth on its own)
                cur = wave_tokens if wave_tokens is not None else default_wave_tokens(N)   # wave_forward's own default
                if cur > floor:
                    wave_tokens = max(floor, cur // 2)
                    if vb:
                        print(f"Stream-CQSA: the wave engine ran out of memory; halving the wave budget to {wave_tokens:,} tokens", flush=True)
                elif gather and q.is_cuda and in_place_possible(N, bool(is_causal)) and direction == "fwd":
                    gather = False
                    if vb:
                        print("Stream-CQSA: the wave engine ran out of memory at one subproblem per wave; reading Q/K/V in place", flush=True)
                elif kernel in WAVE_KERNELS:
                    raise
                else:
                    if vb:
                        print("Stream-CQSA: the wave engine ran out of memory; retrying on the streaming engine", flush=True)
                    use_wave = False
    if not use_wave:
        from .native_autograd import stream_cqsa_attn, kernel_env
        from .stable_stream import stream_cqsa_forward
        host = not on_cuda
        # kernel="triton": the classic engine reads its kernel from the environment, set for this call only
        # (the autograd function carries it to the backward); the process default is untouched
        tri = "triton" if kernel in ("triton", "wave-triton") else None
        with kernel_env(tri, tri):
            if direction == "bwd":
                out = stream_cqsa_attn(q, k, v, causal=bool(is_causal), scale=scale, verbose=verbose,
                                       itr=kw.get("itr", "auto"), c=kw.get("c", 7), interest_set=kw.get("interest_set", (0, 1, 3)),
                                       stream_from_host=bool(kw.get("stream_from_host", host)),
                                       accumulate_on_gpu=bool(kw.get("accumulate_on_gpu", True)),
                                       max_parallel=kw.get("max_parallel"), bwd_itr=bwd_itr)
                info = {"engine": "classic", "kernel": "triton" if os.environ.get("CQSA_FORWARD", "").lower() == "triton" else "cuda",
                        "itr": kw.get("itr", "auto"), "c": kw.get("c", 7), "stream_from_host": bool(kw.get("stream_from_host", host)),
                        "accumulate_on_gpu": bool(kw.get("accumulate_on_gpu", True))}
            else:
                kw.setdefault("stream_from_host", host)
                # leave the fp32 result where the accumulator is; _deliver casts and moves it
                out, info = stream_cqsa_forward(q, k, v, causal=bool(is_causal), scale=scale, verbose=verbose, out_device="acc", **kw)
                info = {k_: v_ for k_, v_ in info.items() if k_ not in ("lse",) and not hasattr(v_, "device")}
                info["engine"] = "classic"; info["kernel"] = "triton" if os.environ.get("CQSA_FORWARD", "").lower() == "triton" else "cuda"
    if vb:
        ran_wave = info.get("engine") == "wave"
        print(_config_line(stage="ran", mode="cqsa", itr=info.get("itr", kw.get("itr")), c=info.get("c", kw.get("c", 7)),
                           n_sub=info.get("n_subproblems"),
                           qkv=(("device -> evicted to host, streamed" if ev["state"] else "host, streamed") if kw.get("stream_from_host") else "device"),
                           acc=("host (CPU)" if not acc_gpu else "device (GPU)"),
                           engine=info.get("engine", "?"), kernel=info.get("kernel", "?"),
                           n_par=(None if ran_wave else info.get("n_parallel", kw.get("max_parallel"))),
                           wave=((_wave_desc(info.get("max_wave_tokens"), info.get("gather"), kw.get("stream_from_host"))
                                  + (f", {info['n_waves']} waves of {info['wave_sizes']}" if info.get("n_waves") else "")) if ran_wave else None),
                           out=f"{out_device} {str(q.dtype).replace('torch.', '')}")
              + (f" | OOM retries {info['oom_retries']}" if info.get("oom_retries") else "")
              + (f" | wave OOM retries {info['wave_oom_retries']}" if info.get("wave_oom_retries") else ""), flush=True)
    out = _deliver(out, out_device, q.dtype, vb)
    return out, info


def _deliver(out: torch.Tensor, device, dtype, vb: bool) -> torch.Tensor:
    """Hand the result over in the caller's dtype and on the caller's device.

    The engines return fp32; the cast needs room for a second copy. When the device
    cannot take it, the cast is done in host memory and the result moved back; if
    even the half-size result does not fit next to what the caller keeps on the
    device, it is returned in host memory with a warning rather than failing after
    the work is done."""
    if out.dtype == dtype and out.device == device:
        return out
    try:
        return out.to(device=device, dtype=dtype)
    except Exception as exc:                                         # noqa: BLE001
        if not _is_oom(exc):
            raise
    host = out.to("cpu") if out.device.type != "cpu" else out
    del out
    torch.cuda.empty_cache()
    host = host.to(dtype)
    if device.type == "cpu":
        return host
    try:
        return host.to(device)
    except Exception as exc:                                         # noqa: BLE001
        if not _is_oom(exc):
            raise
    warnings.warn(f"Stream-CQSA: the result ({host.numel() * host.element_size() / 2**30:.1f} GiB) does not fit on {device} "
                  f"next to what is already there; returning it in host memory.", RuntimeWarning, stacklevel=3)
    return host


WAVE_KERNELS = {"wave": "auto", "wave-cuda": "cuda", "wave-triton": "triton"}


def _wave_desc(wave_tokens, gather, stream_from_host) -> str:
    w = f"wave budget {wave_tokens:,} tokens" if wave_tokens else "wave budget default"
    return w + (", chunk pool" if stream_from_host else (", Q/K/V gathered" if gather else ", Q/K/V read in place"))


def _engines_for(kernel: str, dtype, D: int, direction: str, on_host: bool = False) -> tuple:
    """The engines the planner may choose from for this call: what ``kernel=`` allows, and the wave engine only
    when the native build has kernels for (dtype, D). The wave backward runs on the CUDA kernel only and on
    device-resident inputs only, so a Triton-only wave request for a backward, and any backward of
    host-resident inputs (``on_host``), is the classic engine's."""
    if kernel in ("cuda", "triton"):
        return ("classic",)
    if kernel in WAVE_KERNELS:
        return ("classic",) if (direction != "fwd" and (kernel == "wave-triton" or on_host)) else ("wave",)
    if kernel != "auto":
        raise ValueError(f"kernel must be 'auto', 'wave', 'wave-cuda', 'wave-triton', 'cuda' or 'triton' (got {kernel!r})")
    try:
        from .native_wave import native_available, native_supports
        if native_available() and torch.cuda.is_available() and native_supports(dtype, int(D)):
            return ("wave", "classic")
    except Exception:                                                # noqa: BLE001
        pass
    return ("classic",)


def _pick_wave(kernel: str, q: torch.Tensor, direction: str, engine: str) -> bool:
    """Whether this call runs on the wave engine: ``kernel="wave*"`` forces it, ``"cuda"``/``"triton"`` the classic
    engine; ``"auto"`` follows ``engine``, the planner's choice by the memory model (a W=1 wave costs what a classic
    subproblem costs, so the wave is the executor of every device-resident device-accumulator call it fits).
    A backward of host-resident (or evicted) inputs is the classic engine's whatever ``kernel`` says: the wave
    backward keeps fp32 gradients on the device and cannot take host inputs."""
    if direction != "fwd" and not q.is_cuda:
        return False
    if kernel in WAVE_KERNELS:
        # the wave backward runs on the CUDA kernel only: a Triton-only wave request for a
        # backward goes to the classic engine's Triton kernels
        if direction != "fwd" and kernel == "wave-triton":
            return False
        return True
    if kernel in ("cuda", "triton"):
        return False
    if kernel != "auto":
        raise ValueError(f"kernel must be 'auto', 'wave', 'wave-cuda', 'wave-triton', 'cuda' or 'triton' (got {kernel!r})")
    if engine != "wave":
        return False
    try:
        from .native_wave import native_available, native_supports
        return native_available() and native_supports(q.dtype, int(q.shape[-1]))   # depends on the kernel set the extension was built with
    except Exception:                                                # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# dry run
# ---------------------------------------------------------------------------
def estimate(N: int, B: int = 1, H: int = 8, D: int = 64, dtype=torch.float16, causal: bool = True,
             hardware=None, direction: str = "both", print_table: bool = True, kernel: str = "auto") -> dict[str, Any]:
    """
    What would ``attention`` do for this shape on this machine, and what would it
    cost? Prints (and returns) the monolithic estimate, the chosen decomposition
    and the runner-up configurations, for the forward and the backward.
    ``kernel`` restricts the engines as ``attention(kernel=)`` does.
    """
    from .autoconfig import plan as _plan, hardware_from_dict, detect_hardware, _cname, GIB
    hw = hardware_from_dict(hardware) if isinstance(hardware, dict) else (hardware or detect_hardware())
    res: dict[str, Any] = {"N": N, "B": B, "H": H, "D": D, "dtype": str(dtype), "causal": causal, "hardware": hw.summary()}
    dirs = ("fwd", "bwd") if direction == "both" else (direction,)
    lines = [f"Stream-CQSA estimate for N={N:,} B={B} H={H} D={D} {str(dtype).replace('torch.', '')} {'causal' if causal else 'non-causal'}",
             f"  hardware: {hw.summary()}"]
    itemsize = torch.tensor([], dtype=dtype).element_size()
    for d in dirs:
        p = _plan(N=N, B=B, H=H, D=D, dtype=dtype, causal=causal, hardware=hw, direction=d, allow_distributed=False,
                  engines=_engines_for(kernel, dtype, D, d))
        mono = p.candidates[0]
        feas = sorted([c for c in p.candidates if c["ok"] and c["mode"] != "mono"], key=lambda c: c["time"])[:3]
        r = dict(plan=p.name(), est_time_s=p.est_time_s, est_peak_gib=p.est_peak_gib, est_host_gib=p.est_host_gib,
                 mono_fits=bool(mono["ok"]), mono_peak_gib=mono["peak"], mono_time_s=mono["time"], reason=p.reason,
                 runners_up=[dict(config=_cname(c), time_s=c["time"], peak_gib=c["peak"], host_gib=c["host"]) for c in feas],
                 feasible=any(c["ok"] for c in p.candidates))
        res[d] = r
        label = "forward" if d == "fwd" else "backward (fwd+bwd step)"
        lines.append(f"  {label}:")
        lines.append(f"    monolithic FlashAttention: {'fits' if mono['ok'] else 'does NOT fit'} ({mono['peak']:.1f} GiB), ~{mono['time']:.2f} s")
        if p.mode == "mono":
            lines.append(f"    -> monolithic call")
        elif r["feasible"]:
            lines.append(f"    -> Stream-CQSA {p.name()}: ~{p.est_time_s:.1f} s, {p.est_peak_gib:.1f} GiB on the device"
                         + (f", {p.est_host_gib:.0f} GiB host" if p.est_host_gib else ""))
            for c in feas[1:]:
                lines.append(f"       also: {_cname(c)} ~{c['time']:.1f} s, {c['peak']:.1f} GiB")
        else:
            floor = (4 + 4) * N * B * H * D / GIB
            lines.append(f"    -> nothing fits: the fp32 output + accumulator alone are {floor:.1f} GiB on the device; "
                         f"reduce B/H/D, add host memory, or use more devices")
    if print_table:
        print("\n".join(lines), flush=True)
    res["text"] = "\n".join(lines)
    return res


# ---------------------------------------------------------------------------
# monkeypatch F.scaled_dot_product_attention
# ---------------------------------------------------------------------------
_ORIGINAL_SDPA = None
_PATCH_MIN_TOKENS = 65536
_TRUE_SDPA = F.scaled_dot_product_attention          # captured at import, before any patch
_INSIDE = [0]                                          # re-entrancy: calls made by the engines go to the original


def _patched_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    orig = _ORIGINAL_SDPA or _TRUE_SDPA
    gqa_ok = (key.dim() == 4 and key.shape == value.shape and key.shape[0] == query.shape[0]
              and key.shape[2:] == query.shape[2:] and key.shape[1] >= 1 and query.shape[1] % key.shape[1] == 0)
    supported = (attn_mask is None and not dropout_p and query.dim() == 4
                 and query.dtype in (torch.float16, torch.bfloat16) and gqa_ok
                 and query.shape[-2] >= _PATCH_MIN_TOKENS and _INSIDE[0] == 0)
    if not supported:
        return orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale,
                    **({"enable_gqa": enable_gqa} if enable_gqa else {}))
    return attention(query, key, value, is_causal=is_causal, scale=scale, enable_gqa=bool(enable_gqa))


def patch_sdpa(min_tokens: int = 65536) -> None:
    """
    Route ``torch.nn.functional.scaled_dot_product_attention`` through
    ``stream_cqsa.attention`` for the calls it handles (4-D fp16/bf16, no mask,
    no dropout, GQA/MQA included, N >= ``min_tokens``); everything else goes to the original.
    Below the memory boundary the routed call is still the monolithic kernel, so
    the patch changes nothing until a call would have run out of memory.
    """
    global _ORIGINAL_SDPA, _PATCH_MIN_TOKENS
    _PATCH_MIN_TOKENS = int(min_tokens)
    if _ORIGINAL_SDPA is None:
        _ORIGINAL_SDPA = F.scaled_dot_product_attention
        F.scaled_dot_product_attention = _patched_sdpa
        torch.nn.functional.scaled_dot_product_attention = _patched_sdpa


def unpatch_sdpa() -> None:
    global _ORIGINAL_SDPA
    if _ORIGINAL_SDPA is not None:
        F.scaled_dot_product_attention = _ORIGINAL_SDPA
        torch.nn.functional.scaled_dot_product_attention = _ORIGINAL_SDPA
        _ORIGINAL_SDPA = None


@contextlib.contextmanager
def patched_sdpa(min_tokens: int = 65536):
    """``with stream_cqsa.patched_sdpa(): model(...)``"""
    was = _ORIGINAL_SDPA is not None
    patch_sdpa(min_tokens)
    try:
        yield
    finally:
        if not was:
            unpatch_sdpa()
