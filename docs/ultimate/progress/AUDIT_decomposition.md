# Audit: every correctness comparison forces AND asserts a decomposition

Date 2026-10-21. Scope: `tests/`, `notebooks/make_demo.py`, `notebooks/make_oom_demo.py`, `benchmarks/*.py`,
`benchmarks/ultimate/*.py`. Helpers: `tests/_decomp.py`; the rule: `tests/conftest.py` docstring.
Commits: a927d7c (helpers, conftest rule, `test_ultimate_routing.py`, `test_evict_inputs.py`) and the commit of
this document (the remaining files).

## The contributor rule

`stream_cqsa.attention(q, k, v, ...)` runs the monolithic torch SDPA (flash) call whenever the planner says the
call fits in device memory and nothing is pinned. `kernel=` alone, `evict_inputs=True` alone and a `hardware=`
budget that still holds the monolithic call do NOT decompose. The engine entry point has the same trap one level
down: `stream_cqsa_forward(itr="auto")` and `itr=0` take the classic engine's own monolithic flash branch
(`info["monolithic"] is True`, `n_subproblems == 0`). A comparison "Stream-CQSA vs SDPA / flash_attn / fp64 /
another `attention()` arm" on either of those paths compares SDPA with itself and proves nothing about the
decomposition.

Therefore every correctness comparison must

1. **force** a decomposition: pin `itr>=1` (or `c=`), give a `hardware=` budget the monolithic call cannot fit,
   or call an engine entry point (`stream_cqsa_forward` / `stream_cqsa_backward` / `wave_forward` /
   `wave_backward` / `stream_cqsa_attn` / `wave_attention`) with an int `itr >= 1`; and
2. **assert** that it happened, with `tests/_decomp.py`:
   `out, info = attention_decomposed(q, k, v, itr=1, ...)` (fails on `engine == "monolithic"` /
   `n_subproblems <= 1`; `engine=` pins "classic" / "wave"; an int `itr` / `c` in the kwargs is checked against
   what ran), or `assert_decomposed(info)` for an info dict in hand (engine entry points included).
   A test whose subject IS the monolithic path says so in a comment next to the call and asserts it with
   `assert_monolithic(info)`.

Outside `tests/` (notebooks, benchmark scripts) the same two steps are inline: `return_info=True`, an
`assert info["engine"] != "monolithic" and info["n_subproblems"] > 1` next to the comparison, and the engine
that ran printed in the row / line that states the error.

The backward-direction info of `attention()` (inputs with `requires_grad`) names `engine` / `kernel` / `itr` /
`c` but not the subproblem count: there `itr >= 1` is the evidence (the plan's int depth reaches the engine
through `Plan.engine_kwargs()`, never `"auto"`). `stream_cqsa_backward` / `bwd_itr="auto"` never has a
monolithic path (`plan_decomposition_bwd` returns `itr >= 1`), so a backward with an auto depth is a
decomposition by construction.

## Classification

| status | meaning |
|---|---|
| FORCED+ASSERTED | decomposition forced and asserted from what ran (the helpers, or an equivalent inline assert) |
| FORCED-not-asserted | forced (pinned `itr` / tight budget) but nothing checked that the engine honoured it |
| VACUOUS-RISK | nothing pinned, or `itr="auto"` / `itr=0` at the engine: the comparison ran, or can run, on the monolithic path |
| INTENTIONALLY-MONOLITHIC | the subject IS the monolithic path (pass-through, eviction/restore around a fitting call, the `itr=0` lse contract): a comment says so and `assert_monolithic` (or an equivalent) pins it |
| DIRECT | kernel-module or engine-entry comparison with an int `itr >= 1`: no monolithic branch can be taken |
| N/A | no output comparison (planner / parser / telemetry tests, shape-only checks), or a rival baseline that is not Stream-CQSA |

Counts of `attention()`-level (and engine-level auto/zero-`itr`) comparison sites, status BEFORE the audit:

| status before | tests | notebooks (cells) | benchmarks | total | after |
|---|---|---|---|---|---|
| FORCED+ASSERTED | 2 (`test_ultimate_routing::test_monolithic_oom_replans_among_decompositions` inline; `test_v13` engine test inline) | 0 | 1 (`engine_ab.py`) | 3 | 3 (unchanged; routing one now on the helper) |
| FORCED-not-asserted | 17 | 5 | 2 | 24 | 0: all converted |
| VACUOUS-RISK | 1 (`test_stable_stream::test_auto_itr_matches_flash_attention`) | 1 (`make_demo` cell 0.1: unpinned, prints max abs diff to SDPA) | 0 | 2 | 0: the test runs both branches (auto asserted monolithic + auto* asserted decomposed); the cell now prints the engine that ran and says the 0 is the pass-through |
| INTENTIONALLY-MONOLITHIC | 6 | 2 (`make_demo` table row "(nothing: planned; monolithic fits here)"; `make_oom_demo` sub-boundary rows) | 1 (`quickstart` section 2) | 9 | 9, every one commented, asserted / labelled |
| DIRECT | 19 files (every remaining engine / kernel test file, plus the engine-level tests of the files above) | the engine-layer cells (wave_forward / stream_cqsa_forward / compare_kernels, int itr) | 32 scripts | - | unchanged |

## Full table

Column "status before" is the state before a927d7c / this commit; "change" names what the commit did.

### tests/ already converted in a927d7c

| file | test | status before | change |
|---|---|---|---|
| tests/test_ultimate_routing.py | test_wave_token_ladder_and_memory_model, test_planner_wave_for_device_resident_forward_and_classic_for_host_accumulator_backward, test_tight_budget_shrinks_wave_before_depth_and_in_place_only_when_copies_do_not_fit, test_host_resident_forward_is_decided_by_memory, test_nothing_fits_fallback_under_an_engine_restriction, test_monolithic_fits_whenever_a_device_resident_wave_does | N/A (planner, `plan_only`) | none |
| tests/test_ultimate_routing.py | test_auto_routes_to_the_wave_and_matches_the_forced_paths | FORCED-not-asserted (`itr=1`; auto vs cuda vs wave-cuda arms, fp64 at N=4096) | `attention_decomposed`, `engine=` pinned on the kernel arms |
| tests/test_ultimate_routing.py | test_wave_overrides_and_unplanned_combinations | FORCED-not-asserted (`itr=1`, 7 arms vs the wave base) | `attention_decomposed` |
| tests/test_ultimate_routing.py | test_oom_inside_a_wave_halves_the_budget_before_leaving_the_engine | FORCED-not-asserted (`itr=1`, vs SDPA) | `attention_decomposed` |
| tests/test_ultimate_routing.py | test_verbose_plan_line_reports_the_wave_choice | N/A (`itr=1`, verbose text only) | none |
| tests/test_ultimate_routing.py | test_host_resident_backward_under_auto_is_classic_and_exact | FORCED-not-asserted (`itr=1`, host-resident backward) | `attention_decomposed` |
| tests/test_ultimate_routing.py | test_oom_in_the_wave_backward_is_recovered | FORCED-not-asserted (`itr=1`, wave backward vs its rerun) | `attention_decomposed` |
| tests/test_ultimate_routing.py | test_monolithic_oom_replans_among_decompositions | FORCED+ASSERTED inline (nothing pinned, forced by the simulated monolithic OOM; `n_subproblems > 1` asserted) | `attention_decomposed` (comment: forced by the OOM, not by a pin) |
| tests/test_ultimate_routing.py | test_pinned_depth_is_what_plan_only_reports_and_the_backward_reports_the_gather | FORCED-not-asserted (`itr=2`, `itr=1`) | `attention_decomposed` |
| tests/test_ultimate_routing.py | test_inputs_restored_under_a_memory_fraction | INTENTIONALLY-MONOLITHIC (routing incidental: the subject is the restore under a cap; SDPA check guards the delivered output) | comment; no `assert_monolithic` because the routing under the cap may go either way |
| tests/test_evict_inputs.py | test_planner_models_resident_evictable_inputs | N/A (planner) | none |
| tests/test_evict_inputs.py | test_evict_compute_output_on_device_then_restore | FORCED-not-asserted (TIGHT 1.2 GiB budget vs SDPA) | `attention_decomposed` |
| tests/test_evict_inputs.py | test_restore_never_and_evict_never | FORCED-not-asserted (TIGHT, 2 calls) | `attention_decomposed` |
| tests/test_evict_inputs.py | test_views_of_a_live_fused_tensor_are_reported | FORCED-not-asserted (`itr=1`) | `attention_decomposed` |
| tests/test_evict_inputs.py | test_training_through_eviction | FORCED-not-asserted (`itr=1`, autograd) | `attention_decomposed` |
| tests/test_evict_inputs.py | test_forced_eviction_on_the_monolithic_path_returns_to_the_device | INTENTIONALLY-MONOLITHIC (forced eviction around a fitting call) | comment + `assert_monolithic` |
| tests/test_evict_inputs.py | test_restore_judges_room_under_a_memory_fraction_cap | FORCED-not-asserted (TIGHT) | `attention_decomposed` |

### tests/ converted in this commit

| file | test | status before | change |
|---|---|---|---|
| tests/test_api.py | test_attention_monolithic_below_boundary | INTENTIONALLY-MONOLITHIC (asserted `p.mode == "mono"` at plan level; `torch.equal` with SDPA) | comment; `return_info=True` + `assert_monolithic(info)` (what ran, not only the plan) |
| tests/test_api.py | test_attention_forced_decomposition_is_exact | FORCED-not-asserted (`itr=1` vs fp64, within 1.2x SDPA) | `attention_decomposed(itr=1)` |
| tests/test_api.py | test_attention_kernel_choices_agree | FORCED-not-asserted (`itr=1`, kernel cuda / triton / wave vs fp64) | `attention_decomposed(itr=1, kernel=, engine=)` ("classic" for cuda / triton, "wave" for wave) |
| tests/test_api.py | test_attention_host_inputs_and_memory_cap | FORCED (0.5 GiB budget under a 1.5 GiB cap), asserted at plan level only (`p.mode != "mono"`) | `attention_decomposed(..., return_plan=True)` -> `(out, p, info)`; `info["engine"] == p.engine` |
| tests/test_api.py | test_attention_autograd | FORCED-not-asserted (`itr=1`, gradients vs fp64) | `out, _ = attention_decomposed(itr=1); out.backward(dout)` (backward-direction info names `itr=1`) |
| tests/test_api.py | test_patch_sdpa_roundtrip | INTENTIONALLY-MONOLITHIC (the patched SDPA below the boundary, `torch.equal` with the original) | comment; the patched function returns no info, so `attention(..., plan_only=True).mode == "mono"` pins what the routed call runs |
| tests/test_api.py | test_attention_rejects_unsupported, test_estimate_and_doctor, test_cost_model_cache_roundtrip | N/A | none |
| tests/test_place_and_verbose.py | test_place_inputs_moves_when_it_fits_and_keeps_on_host_otherwise | N/A (unpinned call, shape only) | comment "nothing pinned, no output comparison" |
| tests/test_place_and_verbose.py | test_verbose_configuration_lines (call 1) | INTENTIONALLY-MONOLITHIC (the verbose line of the monolithic plan) | comment; `return_info=True` + `assert_monolithic` |
| tests/test_place_and_verbose.py | test_verbose_configuration_lines (call 2) | FORCED, text-asserted ("ran: decomposed", "engine classic") | `attention_decomposed(itr=1, ..., engine="classic")` |
| tests/test_wave_triton.py | test_triton_wave_vs_fp64_and_cuda, test_triton_wave_c133_and_deterministic | DIRECT (`wave_forward`, int `itr`) | none |
| tests/test_wave_triton.py | test_api_kernel_wave_triton | FORCED-not-asserted (`itr=1`, `kernel="wave-triton"` vs fp64) | `attention_decomposed(itr=1, kernel="wave-triton", engine="wave")` + `info["kernel"] == "triton"` |
| tests/test_stable_stream.py | test_auto_itr_matches_flash_attention | VACUOUS-RISK (`stream_cqsa_forward(itr="auto")` at N=2048: the monolithic flash branch vs the SDPA reference) | two arms: `itr="auto"` asserted monolithic (intentional, the branch the planner picks here; comment) AND `itr="auto", min_itr=1` (auto*: the planner's depth, never the monolithic call) asserted decomposed, both against the same reference, same tolerance |
| tests/test_stable_stream.py | the other 71 tests | DIRECT (int `itr`) or N/A (CPU merge arithmetic, plans, estimates) | none |
| tests/test_bwd_host_stream.py | test_host_matches_device, test_host_uses_less_device_memory | DIRECT (int `itr` 1 / 2) | none |
| tests/test_bwd_host_stream.py | test_itr0_returns_lse_and_backward_works | INTENTIONALLY-MONOLITHIC (`itr=0`: the lse contract of the monolithic branch and the `itr=0` backward) | comment + `assert_monolithic(info)` |
| tests/test_bwd_host_stream.py | test_cpu_inputs_without_flag_are_rejected | N/A (argument check) | none |

### tests/ audited, no change needed (DIRECT or N/A)

| file | tests | status | note |
|---|---|---|---|
| tests/test_bf16.py | test_dtype_fwd_bwd (8 cases) | DIRECT | `stream_cqsa_forward` / `stream_cqsa_backward`, `itr=1` |
| tests/test_backward_default.py | test_default_backward_matches_sdpa (8) | DIRECT | `stream_cqsa_attn(itr=1)` |
| tests/test_wave_shapes.py | test_wave_forward_backward_shapes (8) | DIRECT | `wave_forward` / `wave_attention`, `itr=1` (no monolithic branch in the wave engine) |
| tests/test_devkit_autoconfig.py | test_backward_depth_independent | DIRECT | `stream_cqsa_attn(itr=1, bwd_itr=2)` vs `bwd_itr="auto"`: the backward never has a monolithic path |
| tests/test_devkit_autoconfig.py | test_triton_inner_exact, test_triton_backward_matches_reference | DIRECT | `compare_kernels(itr=1)`; the Triton kernel's own call vs SDPA is a kernel test, not an `attention()` call |
| tests/test_devkit_autoconfig.py | the other 7 | N/A | planner / quorum sets / cost model |
| tests/test_escalation_autograd.py | all 16 | DIRECT | `stream_cqsa_attn` / `stream_cqsa_forward` / `stream_cqsa_backward`, int `itr` everywhere (`itr=depth` from a parametrize of ints) |
| tests/test_reference.py | all 29 | DIRECT / N/A | the CPU reference (`stream_cqsa_forward_reference`, int `itr`) vs `sdpa_reference`; layout / coverage tests |
| tests/test_rect_ooc.py | 2 | N/A | the rectangular OOC rival baseline (always tiled), not Stream-CQSA |
| tests/test_ultimate_align.py | all 16 | DIRECT | `stream_cqsa_forward` / `_backward` / `dist_*`, int `itr`; one test uses `attention()` only in its docstring |
| tests/test_ultimate_scheduler.py | all 23 | DIRECT / N/A | `dist_stream_cqsa_forward` vs `stream_cqsa_forward`, `itr=1`; scheduler unit tests |
| tests/test_ultimate_wave_merge.py | all 8 | DIRECT | `wave_forward`, int `itr` |
| tests/test_ultimate_kernel_rows.py | 3 | DIRECT | the harness's layout passes (int `itr`) recomposed vs fp64, 1.2x FA-2 |
| tests/test_ultimate_v11x.py, test_ultimate_v12b.py, test_ultimate_v12f.py | 4 + 18 + 13 | DIRECT | kernel-module level (fwd_ab / bwd_ab per task) |
| tests/test_ultimate_cuobj_report.py, test_ultimate_dist_telemetry.py, test_ultimate_ncu_aligned.py | 4 + 7 + 3 | N/A | parsers, telemetry tables, no GPU output comparison |

### Read-only (owned by other packages): verdict and the exact edits

| file | test | status | needed edit |
|---|---|---|---|
| tests/test_v13.py | the other 20 tests (kernel-module level per task; one checks the interface's module selection in a subprocess) | DIRECT / N/A | none |
| tests/test_v13.py | test_engine_attention_forward_backward_on_v13 | FORCED+ASSERTED (inline, in a subprocess: `itr=1, c=7` pinned; forward call asserts `engine in ("classic","wave")` and `n_subproblems > 1`; autograd call asserts engine / `itr == 1` / `c == 7`; both outputs agree within 1e-6) | none required. Optional, to use the shared helpers inside the subprocess: in the `_run_py` source replace the two inline asserts with `sys.path.insert(0, %r)` (= `os.path.join(REPO, "tests")`) + `from _decomp import attention_decomposed` and `out_f, info_f = attention_decomposed(q.detach(), ..., kernel="cuda", itr=1, c=7)` / `out, info = attention_decomposed(q, k, v, ..., kernel="cuda", itr=1, c=7, bwd_itr=1)`; keep the `(info["kernel"], info["itr"], info["c"]) == ("cuda", 1, 7)` check |
| tests/test_wave_v2.py | all 13 (engine tests: test_engine_forward_vs_float64, test_engine_host_resident_and_host_accumulator, test_engine_backward_vs_float64; kernel tests vs v12f / v12b per task) | DIRECT | none: every call is `wave_forward` / `wave_backward` / `wave_attention` with an int `itr` from a parametrize; the wave engine has no monolithic branch. No `attention()` call in the file |
| benchmarks/ultimate/engine_ab.py | arms v3_cuda, v3_wave, v223_cuda, v223_triton | FORCED+ASSERTED (`itr`, `c`, `bwd_itr=itr` pinned; a pinned arm whose info engine is not classic / wave becomes an ERROR row; `n_subproblems` filled from `c**itr` for the backward-direction info) | none (verified) |
| benchmarks/ultimate/engine_ab.py | arm v3_auto | INTENTIONALLY unpinned ("what the user gets"), every row records the engine that ran and the report column says so (monolithic at these N) | none (verified) |

### notebooks (generators edited; the executed .ipynb files need a re-execution job, see below)

| file | cell | status before | change |
|---|---|---|---|
| notebooks/make_demo.py | 0.1 "one call": `attention(q,k,v)` at N=262144 on the device, prints max abs diff to SDPA | VACUOUS-RISK as an exactness claim (the call fits: the 0 is the pass-through) | `return_info=True`; prints `ran: monolithic`; comment says the 0 is the pass-through, the next cell forces a decomposition |
| notebooks/make_demo.py | 0.2 `hardware={"cuda:0": "1GiB"}` + max abs diff to SDPA | FORCED (budget), labelled by `plan.name()` only | `return_info=True`; `assert engine != "monolithic" and n_subproblems > 1`; prints what ran |
| notebooks/make_demo.py | 2 "when it does not fit": `hardware=SMALL` (2 GiB) at N=524288, rel err vs fp64 | FORCED, printed ("what ran") but not asserted | assert added |
| notebooks/make_demo.py | 2 "inside a model": `hardware=1.2GiB` with 0.75 GiB device Q/K/V, rel diff vs SDPA | FORCED-not-asserted | assert added; the error line prints `ran: <engine> (<n> subproblems)` |
| notebooks/make_demo.py | 3 options table (7 rows, rel err vs fp64 on sampled rows) | row 1 INTENTIONALLY-MONOLITHIC (labelled "monolithic fits here"); rows 2-7 FORCED (pinned `itr`), the subproblem column showed it | `engine` column; pinned rows assert a decomposition; row 1 commented as SDPA's own error |
| notebooks/make_demo.py | 4 training (`itr=` pinned, grad norms only) | N/A (no comparison) | none |
| notebooks/make_demo.py | 6 engines table (12 rows, `kernel=` + `itr` pinned, rel err vs fp64) | FORCED-not-asserted | `return_info=True`; assert classic / wave and `n_subproblems > 1`; `ran` column |
| notebooks/make_demo.py | 6 `plan_only` loop; `wave_forward(itr=1)` budgets; `stream_cqsa_forward(itr=1)` Triton; part B trace | N/A / DIRECT | none |
| notebooks/make_demo.py | 7 devkit `compare_kernels(itr=1/2)`, `quick_bench` (the `Config(mode="mono")` rows are labelled baselines) | DIRECT | none |
| notebooks/make_demo.py | 8 adapters `compare_kernels(itr=1)` | DIRECT | none |
| notebooks/make_oom_demo.py | N sweep (SDPA vs `attention(hardware=hw)` under a 3 GiB cap; err vs fp64 both) | INTENTIONALLY mixed: the boundary is the subject; sub-boundary rows are monolithic by design and the markdown said so, but the row was labelled by the plan only | `return_info=True`; the row is labelled `[plan; ran <engine>, <n> subproblems]`; whenever the baseline OOMs the Stream-CQSA result is asserted decomposed; `engine` / `n_subproblems` recorded in the JSON rows; markdown says the sub-boundary error column is SDPA's own |

### benchmarks

| file | site | status before | change |
|---|---|---|---|
| benchmarks/quickstart.py | section 2, N=131072 device-resident, `identical=torch.equal(out, ref)` | INTENTIONALLY-MONOLITHIC (the point of the section) | `return_info=True`; `assert engine == "monolithic"`; prints `ran: monolithic`; comment |
| benchmarks/quickstart.py | section 3, N=1M under a 6 GiB cap, `hardware=4.8GiB`, rel diff to the fp16 kernel | FORCED-not-asserted (plan checked offline: `cqsa c=7 itr=1 acc=cpu`) | assert `engine != "monolithic" and n_subproblems > 1`; prints what ran |
| benchmarks/quickstart.py | section 4, host-resident N=4M, no comparison | N/A | none |
| benchmarks/quickstart.py | section 5, autograd N=262144, `hardware=2GiB`, gradients vs SDPA | FORCED-not-asserted (plan checked offline: `bwd cqsa c=7 itr=1 acc=cpu, evicted`) | assert `engine != "monolithic" and itr >= 1` (backward-direction info) |
| benchmarks/quickstart.py | section 6, `patched_sdpa` inside a module, no comparison | N/A | none |
| benchmarks/e2e_check.py, e2e_compare.py, parallel_subproblems.py, parallel_timeline.py, pipeline_ab.py (itr 1 and 2), pipeline_ab_bwd.py, pipeline_baseline.py, pipeline_opts.py, quorum_axis.py, triton_bench.py, triton_bwd_bench.py, profile_sweep.py | `stream_cqsa_forward` / `wave_forward` with an int `itr` (`kw = dict(itr=1, ...)` or a parametrized int) | DIRECT | none |
| benchmarks/quick_bench_run.py | `compare_kernels(..., cfg=Config(itr=1/2))`; `quick_bench(default_configs(max_itr=2))` whose two `Config(mode="mono")` rows are the labelled baselines of the timing table | DIRECT | none |
| benchmarks/align_census.py, kernel_ab.py, kernel_baseline.py, layout_ab.py, ncu_one.py, profile_kernel_composition.py, triton_bench_large.py | kernel-module / task-layout level | DIRECT | none |
| benchmarks/ultimate/fold_workers_ab.py, engine_align_ab.py, wave_ab.py, wave_merge_ab.py, dist_telemetry.py | `stream_cqsa_forward` / `wave_forward` / `dist_stream_cqsa_forward`, int `itr` | DIRECT | none |
| benchmarks/ultimate/fwd_ab.py, bwd_ab.py, kernel_rows.py, kernels_f23.py, v12f_static.py, v12b_static.py, exp_build_ab.py | kernel-module level per task | DIRECT | none |
| benchmarks/ultimate/cuobj_report.py, ncu_aligned.py, dist_cost_refit.py, dist_sched_dryrun.py, dist_sched_report.py | no output comparison | N/A | none |

## Verification (vis1, shared A100-PCIe-40GB, correctness only; v13 module `next/kernel/v13/frozen/c2`)

- `tests/test_api.py tests/test_place_and_verbose.py tests/test_wave_triton.py tests/test_bwd_host_stream.py`:
  36 passed (24.9 s).
- `tests/test_stable_stream.py`: 72 passed (8.9 s).
- Negative check of the helper: `attention_decomposed(q, k, v, is_causal=True)` with nothing pinned at N=4096
  raises `AssertionError: the monolithic path ran (engine='monolithic') ...`.
- `notebooks/make_demo.py --no-exec` and `make_oom_demo.py --no-exec` written to the scratchpad: every code
  cell parses. The executed `.ipynb` files in `notebooks/` are NOT regenerated here (they carry outputs from an
  A100 run; `python notebooks/make_demo.py` / `make_oom_demo.py` need a dedicated GPU job, sizes up to a 40 GiB
  cap). Until that job runs, the notebooks' outputs predate the labels; the generators are the source of truth.
- `benchmarks/quickstart.py` is a slurm script (N up to 4M, 16 GiB caps): not run here; its two forced sections
  were checked with `plan_only=True` (section 3: `cqsa c=7 itr=1 acc=cpu n_par=1 host-resident`; section 5:
  `bwd cqsa c=7 itr=1 acc=cpu, Q/K/V evicted to host`), so the new asserts hold on the planned path.
- a927d7c's two files: 23 passed (recorded in that commit).
