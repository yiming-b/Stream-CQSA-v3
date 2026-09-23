# Stream-CQSA "ultimate kernel" program: design basis (map 00)

Written 2026-09-21 from the reader maps (classic kernel, wave kernel/engine, Triton, backward follow-up), the critic pass, and a small number of direct checks (marked "checked here"). Every number carries its source file. Paths are relative to `/scratch/gpfs/AKEY/yb2807/Stream-CQSA-v2` unless they start with `next/` (= `/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/`), `hopper/` (= `/scratch/gpfs/AKEY/yb2807/VENVs/src/flash-attention/hopper/`) or `/`.

Status labels: **[V]** verified in source or a results file by a reader or here; **[I]** inference (arithmetic on verified numbers, or a reading of code not confirmed by a measurement); **[C]** reported by the critic pass only, not re-verified.

Notation: N = global tokens, c = quorum-set size, l = quorum size (chunks per subproblem), itr = decomposition depth, L = N·(l/c)^itr = packed tokens per subproblem (3N/7 at c=7, itr=1), W = subproblems per wave launch, S = padded rows per subproblem in the uniform wave layout.

---

## 1. What exists today

### 1.1 The decomposition (shared by everything)

- A subproblem is a packed subsequence of whole cyclic-quorum chunks gathered in ascending chunk order (`sorted_gather=True`), so local order = global order and a local lower-triangular mask is exactly the global causal mask (`stream_cqsa/reference.py:120-205`, docstring 131-146; engine default `stable_stream.py:396,465,1124`) [V].
- The pair mask is not a chunk-pair table: every token carries one int64 "group bits" word (one bit per non-owner chunk run, at most 63 bits, `csrc/flash_attn/flash_api.cpp:37-220`, `stream_cqsa/cqs_mask.py:99-160`); pair (r,c) is masked iff `bits[r] & bits[c] != 0` (`csrc/flash_attn/src/mask.h:483-498`). Owner tokens have bits 0, so a non-owner chunk's whole diagonal block including (i,i) is masked [V].
- Per-64-token OR/AND summaries (`cqs_blk_or`, `cqs_blk_and`; `CQS_BLK_SIZE=64` enforced at `flash_api.cpp:789`; built in numpy at `stream_cqsa/interface.py:167-209`) classify a tile as clear (`(or_r & or_c)==0`), masked (`(and_r & and_c)!=0`, only when both ranges are block-aligned and inside seqlen, `mask.h:150-176`), or mixed [V].
- `token_ids[l]` never reach the kernel; they drive the merge (`csrc/flash_attn/src/cqsa_kernel.cu:117-170`) and adapters (`stable_stream.py:1039-1066`) [V].
- Tile census at N=131072, blk 64, causal, sorted gather, align=1: itr=1 path (0,) L=56175: 22.2% masked / 77.6% clear / 0.2% mixed; itr=2 (0,0) L=24075: 22.1/77.4/0.5; (0,1): 31.9/67.6/0.5 (`docs/LOG.md:329-338`) [V].
- Live-pair floor [I, arithmetic on the planner's own formula]: the planner counts masked pairs per level as (l-1)/l² (`stream_cqsa/autoconfig.py:233-236`). For a perfect difference set l²-l+1 = c, so c^itr · (l/c)^(2·itr) · (1-(l-1)/l²)^itr = ((l²-l+1)/c)^itr = **1.00× the monolithic pair count at every itr** if every fully masked tile costs zero, and (l²/c)^itr if masked tiles are not skipped (9/7 = 1.29× at c=7, itr=1, which `docs/LOG.md:739` states as "9/7 of the pairs"). This is the theoretical redundancy floor used in section 2.

### 1.2 Classic CUDA kernel: cqsa_cuda (v11, causal) and cqsa_cuda_nc (v9, non-causal)

FlashAttention-2 2.8.3 sm80 fork (cute/cutlass, 128×128 tiles, 4 warps, mma.sync m16n8k16, cp.async, ldmatrix; `csrc/flash_attn/src/kernel_traits.h:15-130`) with CQS additions in 7 files. `csrc/flash_attn/src` is byte-identical to `next/kernel/v11`, `csrc_nc/flash_attn/src` to `next/kernel/v9`; the two trees differ only in `flash_fwd_kernel.h` and `flash_fwd_launch_template.h` (diff -q, `setup.py:238-245`) [V].

| Item | Fact | Source |
|---|---|---|
| Params | `Flash_fwd_params`: `cqs_group_bits`, `cqs_blk_or/and`, `cqs_blk_size/num_blocks`, `cqs_out_acc_ptr` (fp32 normalised out), `cqs_block_base` + `cqs_seg_align` (in-place segmented reads) | `csrc/flash_attn/src/flash.h:144-181` |
| Host checks | int64 bits of length seqlen_q; seqlen_q == seqlen_k; blk_size must be 64; block_base int32 with seg_align>0; out_acc fp32 | `csrc/flash_attn/flash_api.cpp:734-796` |
| Cqs_mode | 0 off (upstream loop), 1 on at compile time (causal only), 2 runtime flag (non-causal); `CQS_MODE_SWITCH` never instantiates mode 1 for non-causal (ptxas hang >5 h at hdim128) | `csrc/flash_attn/src/flash_fwd_launch_template.h:54-72,88-96`; `flash_fwd_kernel.h:60-66`; `docs/LOG.md:689-691` |
| v11 steady loop | live tiles only: row summary once per CTA (`cqs_row_summary_c`), column summary two 16-B vector loads, `while` search for next non-masked n_block overlapped with the QK GEMM, no predicated regions, no `continue` | `flash_fwd_kernel.h:349-351, 461-569`; `mask.h:194-355` |
| Masking-step tiles | always run (Is_first softmax init); only clear-vs-not is computed, so a fully masked diagonal tile of a non-owner row block goes through the per-element `apply_mask` path | `flash_fwd_kernel.h:361-374, 400-403`; `mask.h:538-542, 553-630` |
| Mixed tiles | per-element test: one bits load per row, one dependent global load per column element, fully unrolled over 64 scores/thread; smem staging (v2) exists but v11 passes nullptr | `mask.h:510-700`; `flash_fwd_kernel.h:400-403, 533-536` |
| fp32 store | `acc_o` written from MMA registers right after `normalize_softmax_lse`; fp16 out + lse still written by the stock epilogue | `flash_fwd_kernel.h:573-601`; `interface.py:247-296` |
| Segmented input | `cqs_seg_delta = block_base[r/align] + r%align - r` per K/V tile (runtime div/mod + one global load per tile); chunk boundaries must be align (128) multiples | `flash_fwd_kernel.h:28-35,156-160,200-217`; `reference.py:68-104, 583-604` |
| Launch | grid (num_m_block, b, h) = (439,1,8) at L=56175; 128 threads; traits `<64,128,128,4,false,false>` for hdim64; hdim128 128×64 or 128×32/64×64 by smem | `flash_fwd_launch_template.h:83-84, 211-227, 256-331`; `next/logs/cqsa_ncu11_13780907.out:6` |
| Dispatch | `CQSA_CUDA_MODULE` (default `cqsa_cuda`) serves causal; `CQSA_CUDA_MODULE_NONCAUSAL` (default `cqsa_cuda_nc`) serves non-causal; backward always `cqsa_cuda` | `stream_cqsa/interface.py:13-47, 73, 234, 318, 400` |
| v9 loop | runtime flag, per-tile masked verdict, guarded V load/QK GEMM/K prefetch, `if (cqs_skip_block) continue;`, sCqsBits staging | `csrc_nc/flash_attn/src/flash_fwd_kernel.h:463-539` |
| Backward | seqk-parallel (`grid (n_blocks,b,h)`), hdim64 traits `<64,128,128,8,4,4,4,false,false>` on A100; dead tiles skip only 5 GEMMs + apply_mask + exp2 via guards (no `continue`); verdicts use runtime-blk_size helpers every iteration; single fp32 `dq_accum` with atomicAdd, `deterministic=False` hard-coded | `csrc/flash_attn/src/flash_bwd_kernel.h:485-600, 501-509, 515-776`; `flash_bwd_launch_template.h:74-82, 187-188`; `interface.py:427-440` |
| Build sets | `CQSA_KERNEL_SET` = common (default: fp16+bf16, hdim 64+128, fwd causal+non-causal, bwd 64+128), full (all hdims 32..256 fwd incl. split-KV; bwd only 64/128 exist), a100_fp16_hdim64_128, a100_fp16_hdim128, a100_fp16_hdim64_noncau; sm80 unless `FLASH_ATTN_CUDA_ARCHS` | `setup.py:17-31, 77-150, 173-204`; `flash_api.cpp:463-495` |
| Bit-identity | v11 and v9 bit-identical to the shipped kernel on real/zero/all-masked/random/half/ragged patterns (maxdiff 0.0 in every A/B row) | `results/kernel_ab_clean_v11.json`; `docs/kernel_technical_note.md:109-114` |

Version history (all clean-node A/B, `docs/kernel_technical_note.md:61-84`, `docs/LOG.md:408-729`) [V]: v1 verdicts from params (-3..5%); v2 smem staging of mixed-tile bits (no effect); v3 skip K/V loads of masked tiles (5-9% slower: second verdict in an issue-bound loop); v4 row half hoisted (-9..10% live tiles); v5 compile-time blk 64 + column prefetch (masked tile 63%→45% of a live tile, live tiles regressed); v6 compile-time Is_cqs, one Check_inf softmax, early fp32 store (CQS-off == FA-2, non-causal 3.3× slower); v7 16-B vector column loads (best causal so far, -14.6%); v8/v9 = v5+v7 (+early store) without template flag (non-causal fine); v10 three-way Cqs_mode; v11 live-tile loop (-19% vs shipped at 131K causal).

### 1.3 Wave CUDA kernel: cqsa_native (`native/`)

v11 extended so that one launch runs W subproblems using FA-2's varlen batch machinery; every per-token table is packed with a prefix array and offset once per CTA (`native/csrc/flash_attn/src/flash.h:181-200`; `flash_fwd_kernel.h:109-125`) [V].

| Item | Fact | Source |
|---|---|---|
| Modes | 0 off, 1 causal, 2 runtime (non-causal, wave via `params.cqs_wave`), 3 causal wave UNIFORM layout, 4 non-causal wave (never came out of ptxas); `Wave_ok = (kHeadDim == 64)` | `native/.../flash_fwd_kernel.h:60-78`; `flash_fwd_launch_template.h:57-78,104-111` |
| Uniform layout | every subproblem padded to S = ceil(max_L/128)·128 rows; grid (S/128, W, H); tables at `blockIdx.y*stride`; padding keys hidden by the causal mask, padding queries computed and discarded, padding block-map entries alias the last real block; adopted because the varlen variant held table pointers in registers at REG 255 and ran 12% slower at W=1 (19.9 vs 17.7 ms) | `native/.../flash.h:181-200`; `docs/LOG.md:996-1013`; `stream_cqsa/native_wave.py:175-218` |
| In-place reads | batch stride 0 and per-128-block delta on the Q tile and every K/V tile, next tile's delta looked up during the QK GEMM (cut the penalty from 5% to ~2%); requires device-resident input with N % 128 == 0, else the wave is gathered with zero padding (0·NaN hazard in the PV GEMM) | `native/.../flash_fwd_kernel.h:30-33,208-211,255-283,579-597`; `native_wave.py:491-507` |
| Epilogue | `o_ptr == nullptr` → fp32 out + lse only, no fp16 store | `native/.../flash_fwd_kernel.h:135-151,646-686` |
| Merge | `wave_merge_kernel`: grid (U tokens, H), rows stably sorted by global token on the host, one thread block per (token, head), max-shifted fold in fixed entry order, no atomics; reruns bit-identical, wave partitions agree to ~1e-8 | `native/csrc/flash_attn/src/wave_kernels.cu:28-73,101-166`; `native_wave.py:219-227`; `results/native/native_test_13812109.out:1-33` |
| Backward | `bwd_wave` = the same FA-2 CQS backward with packed-table offsets (12 differing lines), varlen layout, gathered inputs only, global lse + global rowsum(dO·O) in a padded layout (`cu[w] + 128·w`), per-wave `dq_accum` zeros, `deterministic=false`, `wave_scatter_add` into fp32 [N,H,D] gradients | `native/csrc/flash_attn/flash_api.cpp:2693-2771`; `native/.../flash_bwd_kernel.h:103-107`; `wave_kernels.cu:82-97` |
| Entry points | `fwd_wave` (varlen, non-causal only; dispatches to `mha_fwd_wave_uniform` when `uniform_S > 0`), `bwd_wave`, `wave_merge`, `wave_scatter_add`; rejects H_k != H, dropout, masks, softcap | `native/csrc/flash_attn/flash_api.cpp:2494-2690, 2780-2798` |
| Kernel set | `native_common64`: hdim64 fp16/bf16 causal+non-causal forwards + hdim64/128 fp16/bf16 backwards; `run_mha_fwd` TORCH_CHECKs d == 64 | `native/setup.py:73-78,120-132,174-211`; `native/.../flash_api.cpp:481-483` |
| Built artifact | `next/native/cqsa_native.cpython-311-x86_64-linux-gnu.so` (76.5 MB, Sep 13); the v2 repo's `native/` holds no .so; CQSA311 site-packages has no cqsa_native entry | ls (reader map 2) |

### 1.4 Triton kernels (`stream_cqsa/triton_kernel.py`, 678 lines, no build)

| Item | Fact | Source |
|---|---|---|
| Forward | one kernel serves classic (grid (cdiv(L,128), B·H)) and wave (grid (S/128, W·H)) through `HAS_LENS` and per-batch table strides; row-side verdict once per program; tiles 128×64, 4 warps, 3 stages | `triton_kernel.py:45-165, 305-348, 387-421` |
| Skipping | host-built live-run tables (`live_runs` 255-288, `live_runs_batched` 351-384); the main loop iterates runs so the tile body has no conditional (a scalar `if` cost +27-31% at every L); diagonal tiles in a second loop with `CHECK_MASKED=True` | `triton_kernel.py:120-154`; `docs/LOG.md:842-851` |
| Numerics | fp32 QK, base-2 online softmax in fp32, P cast to V dtype, fp32 out, natural-log lse with -inf for empty rows (CUDA contract); 1.6-3.0e-4 rel vs CUDA, lse |d| <= 6e-6 | `triton_kernel.py:156-165, 183-236`; `docs/LOG.md:820-822` |
| Backward | two deterministic kernels, dK/dV (4 tl.dot) and dQ (3 tl.dot), no atomics, 64×64, 4 warps, 2 stages; package default backward when Triton imports | `triton_kernel.py:477-673`; `interface.py:341-397` |
| Wave | uniform layout only, always gathered into packed fp16 copies, merged by `merge_rows_torch`; no Triton wave backward | `native_wave.py:293-333, 456-483`; `next/LOG.md:1390, 1428` |
| Not supported | shared_chunks / block_base (raises), hdim > 128, autotune (none), TMA/warp specialisation | `triton_kernel.py:424-431`; grep autotune (none) |
| Triton 3.6 on Hopper | compiles to sm_90a; tensor descriptors lowered to TMA only when capability >= 9 and rewritten to pointer loads otherwise; hopper warpspec + pipeline passes for cap 8x/9x; `tl.range(warp_specialize=True)` documented Blackwell-only; Gluon dialect has `warpgroup_mma`, TMA async copies, mbarriers | `/scratch/gpfs/AKEY/yb2807/VENVs/CQSA311/lib/python3.11/site-packages/triton/backends/nvidia/compiler.py:99-102, 230-236, 268-303`; `triton/language/core.py:3273-3284`; `triton/experimental/gluon/language/nvidia/hopper/__init__.py:15-125` |

### 1.5 Classic engine (`stream_cqsa/stable_stream.py`, `api.py`)

One subproblem per kernel launch, up to `max_parallel` in flight on streams (`n_par`), pinned-host streaming with double-buffered staging, host or device accumulator, pipelined merge (`CQSA_FWD_MERGE_ASYNC=1` default per memory note `stream-cqsa-next-work.md`), `shared_chunks` (itr=1 only, host-resident only, `stable_stream.py:1387-1399`), `task_subset` for sharding (`stable_stream.py:1303-1305`), inner-kernel selection (`local_stats_flash` CUDA or `triton_inner`; shared_chunks forced off for non-CUDA inners, `stable_stream.py:1214-1217`) [V]. Tasks are built with `seg_align = 0` unless segmented/shared_chunks mode, because padding a refined child to 128 tokens made it larger than its parent during escalation (`stable_stream.py:1286-1301`, checked here) [V]. `CQSA_SCATTER_ASYNC` default is off (`stable_stream.py:105-113`, comment says "default ON" — contradiction, section 7).

### 1.6 Wave engine (`stream_cqsa/native_wave.py`)

Greedy contiguous wave planning under a packed-token budget from free memory (fwd H·D·4 + H·4 + 40 B per packed token; bwd 7·H·D·itemsize + H·D_r·4 + 8H + 32), cap `min(2M, max(512K, N))` since v2.2.1, memory fraction 0.85, 256 MiB reserve (`native_wave.py:40-50, 256-285, 363-367, 393-403`); ChunkPool of zero-initialised 128-aligned level-1 chunk slots for host-resident inputs (`native_wave.py:406-418, 652-734`); host accumulator path is synchronous per wave (`native_wave.py:333-360`); `wave_attention` autograd with `bwd_itr` (`native_wave.py:625-646`) [V]. `wave_tasks` builds tasks with `seg_align=SEG_ALIGN=128` (`native_wave.py:134-139`, checked here) [V]. Routing: `attention(kernel="auto")` takes the wave only for device-accumulator forwards with >= 20 subproblems (`WAVE_MIN_SUBPROBLEMS`) when `native_supports(dtype, D)`; OOM in the wave retries on the classic engine (`api.py:410-431, 496-534`) [V]. No distributed wave (`grep -c wave distributed.py` = 0; `docs/LOG.md:1088-1091`) [V].

### 1.7 Distributed (`stream_cqsa/distributed.py`, 181 lines)

Every rank builds the identical task list and takes a round-robin shard (`_shard`, `distributed.py:47-48`; `task_subset` at 72-82), runs the unmodified single-device engine, then merges by `all_reduce(MAX)` of m and `all_reduce(SUM)` of the rescaled acc and l (`distributed.py:97-105`); backward = per-rank shard + one chunked `all_reduce(SUM)` per gradient (`distributed.py:126-175`) (checked here) [V]. No per-rank rate, no cost-aware assignment, no reduce_scatter, no wave path [V].

### 1.8 Planner (`stream_cqsa/autoconfig.py`, 634 lines; `memory_fitting.py`, 498 lines)

`HardwareSpec` (devices with `model`, `budget_bytes`, `n_sm`, `identical_group`; `host_budget_bytes`, `n_cpu`, `interconnect_gbs` default 25, NVLink detected via `nvidia-smi topo`) and `CostModel` with defaults from A100-SXM4 v11 measurements: `pair_rate 7.4e11`, `kernel_ratio 1.40`, `task_overhead_s 3e-3`, `bwd_ratio 3.0`, host per-token stage costs, `overlap 0.85/0.92`, `dist_merge_bytes_per_tok 8192`, `dist_imbalance 1.35`, `dist_fixed_s 0.4`; `cqsa_time()` models multi-device as `t·ceil(n_tasks/world)/n_tasks·dist_imbalance + comm + fixed` — i.e. homogeneous ranks only (`autoconfig.py:89-112, 145-160, 200-262`, checked here) [V]. `plan()` searches c ∈ QUORUM_SETS (c=3..133), itr, acc, n_par, world (`autoconfig.py:58, 348-398`); `calibrate()` refits from a small-N run (`autoconfig.py:476-539`); a calibrated model exists for one A100-PCIe-40GB interactive node (`results/costmodel_a100_pcie40_interactive.json`: pair_rate 4.98e11, kernel_ratio 1.30, depth_factor 1.45) and for the 1M quorum-axis run (`results/quorum_axis_1M.json` calib: pair_rate 6.04e11, kernel_ratio 1.50, task_overhead 1.7e-3) [V]. The memory model has no wave term (reader map 2, grep) [V].

### 1.9 Capability matrix

| Row | classic CUDA kernel (v11 / v9) | wave CUDA kernel | Triton | classic engine | wave engine | distributed |
|---|---|---|---|---|---|---|
| forward | yes, 1 subproblem/launch (`flash_fwd_launch_template.h:83-84`) | yes, W/launch (`flash_api.cpp:2494-2690`) | yes (`triton_kernel.py:45`) | yes | yes | yes (`distributed.py:51`) |
| backward | CUDA bwd (`flash_bwd_kernel.h`), 2.0× FA-2 | `bwd_wave`, gathered/varlen only | two deterministic kernels (`triton_kernel.py:477-630`) | default Triton; `CQSA_BACKWARD=cuda` (`interface.py:380-399`) | CUDA only (`native_wave.py:606-611`) | yes (`distributed.py:126-175`) |
| dtypes | fp16, bf16 (common set) | fp16, bf16 (hdim64 fwd) | fp16, bf16 tested; fp32 untested | as kernel | as kernel | as engine |
| head dims | common 64/128 fwd+bwd; full 32-256 fwd | fwd 64 only (`Wave_ok`, `flash_fwd_kernel.h:76`); bwd 64/128 | 16/32/64/128 (`triton_kernel.py:305-348`) | 64/128 | 64 (128 routed to classic, `docs/LOG.md:1191-1200`) | as engine |
| causal / non-causal | causal → v11; non-causal → v9 extension (`interface.py:13-47`) | causal = uniform mode 3; non-causal = varlen mode 2 | both | both | both | both |
| host streaming | segmented in-place reads (`cqs_block_base`) | ChunkPool block map | no (`triton_kernel.py:424-431`) | pinned double-buffered, shared_chunks itr=1 | ChunkPool | yes (acc=CPU measured, `results/dist_test_w2_itr1_cpu_bwd.json`) |
| acc placement | n/a (fp32 out + lse) | n/a | n/a | GPU or host (`low_memory`) | GPU or host (`accumulate_on_gpu`) | GPU or host per rank + all_reduce |
| multi-subproblem / launch | no | yes | yes (`HAS_LENS`) | no (streams, `max_parallel`) | yes | no |
| multi-device | - | - | - | via `task_subset` | none | round-robin shard |
| bwd_itr | - | - | - | `attention(bwd_itr=)` (commit 24a4ff7, memory note) | `wave_attention` (`native_wave.py:625-646`) | not verified |
| task_subset | - | - | - | yes (`stable_stream.py:1303`) | not verified | uses it (`distributed.py:82,160`) |

---

## 2. Where we stand vs FA-2

All Stream-CQSA/FA-2 ratios below are on the same GPU within one job unless stated. Redundancy floor per section 1.1: kernel-level per subproblem = 0.78× FA-2 on the same L (22.2% tiles skipped at itr=1); end-to-end live-pair floor = 1.00× monolithic FA-2 at any itr for perfect-difference-set quorums (1.29× at c=7 itr=1 if masked tiles are not skipped). Host-stage and merge costs are on top.

### 2.1 Kernel-only, one subproblem, fp16, B=1, H=8, D=64

| L (layout) | causal | FA-2 ms | CQS kernel ms | ratio | floor | GPU / job | Source |
|---|---|---|---|---|---|---|---|
| 56175 (N=131K, path (0,), align=1) | yes | 17.46 | v11 24.43 (shipped 30.22; v11 plain 18.91; zero 23.18) | 1.40 | 0.78 | A100-SXM4-80GB, 13780947 | `results/kernel_ab_clean_v11.json` |
| 56064 (N=131K, `wave_tasks` tasks[3], align=128) | yes | 18.09 | v11 17.68; wave W=1 packed 17.91; in place 18.42 | 0.98 | 0.78 | A100-SXM4-80GB, 13812109 | `results/native/wave_bench.json` kernel_ms; `next/native/bench_wave.py:30-51` |
| 56175 | yes | 16.93 | Triton CQS 22.47 (Triton plain 19.50) | 1.33 | 0.78 | A100-SXM4-80GB | `results/triton_bench.json` |
| 7023 / 14044 / 28088 (16K-64K) | yes | 0.47 / 1.44 / 5.06 | v11 1.36 / 3.35 / 8.01 | 2.9 / 2.3 / 1.58 | 0.78 | SXM4, 13780947 | `results/kernel_ab_clean_v11.json` |
| 112K / 449K / 899K (256K-2M) | yes | 72.5 / 1090 / 4411 | v11 83.4 / 1153 / 4456; Triton 84.9 / 1274 / 5042 | 1.15 / 1.06 / 1.01 | 0.78 | SXM4, 13799779/80 | `docs/LOG.md:859-871` |
| 56175 | no | 34.20 | v9 48.32 (v11 mode-2 183.66 pathological; v11 plain 33.48); Triton 36.42 | 1.41 (Triton 1.07) | 0.78 | SXM4, 13780947 | `results/kernel_ab_clean_v11.json`; `results/triton_bench.json` |
| 449K / 899K | no | 2192 / 9015 | v9 2686 / 10698; Triton 2153 / 8587 | 1.23 / 1.19 (Triton 0.98 / 0.95) | 0.78 | SXM4 | `docs/LOG.md:867-871` |

The two L≈56K causal rows disagree by 38% for the same v11 binary (section 7, item 1). Per-pattern costs on an idle A100-PCIe-40GB (v11, L=56175 causal): plain 20.9, real 26.2, zero 27.0, all-masked 19.2 ms; a random all-mixed pattern is 5.7× the zero pattern (`docs/LOG.md:459-467, 693-703`).

### 2.2 Single GPU, device-resident inputs, end-to-end forward (A100-SXM4-80GB, fp16 causal H=8 D=64)

| N | c, itr | monolithic FA-2 | classic engine | wave engine | best ratio | floor (live pairs) | Source |
|---|---|---|---|---|---|---|---|
| 131072 | 7, 1 | 87.5 ms | 147.6 ms (n_par=2) | 133.3 ms (one wave) | 1.52 | 1.00 | `results/native/wave_bench.json` n131k[0], mono_ms_131k |
| 131072 | 31, 1 | 87.5 | 180.0 | 142.6 | 1.63 | 1.00 | same, n131k[1] |
| 131072 | 73, 1 | 87.5 | 219.4 | 153.8 | 1.76 | 1.00 | same, n131k[2] |
| 1048576 | 7, 1 | 5905 ms | 7763 | 7937 | 1.31 | 1.00 | same, n1m |
| 1048576 | 31, 1 | 5905 | 7858 | 7870 | 1.33 | 1.00 | same, n1m |
| 512K | 7, 1 | (no FA-2 arm) | 2.004 s / 4.74 GiB | 2.008 s / 6.64 GiB | - | - | `results/profile_sweep/fits.json` N_itr1_gpu |
| 2M | 7, 1 | 24.18 s (device-resident, job 13780966) | 31.52 s | 32.43 s | 1.30 | 1.00 | `fits.json`; `results/e2e_compare_v11.json` |

Time fits (x = N/1e6, s): classic CUDA 7.222x² - 0.148x + 0.060; wave CUDA 7.431x² - 0.133x + 0.029; classic Triton 8.046x² - 0.044x + 0.031; wave Triton 8.402x² - 0.092x + 0.047 (itr=1 acc=GPU); itr=2: classic 7.206x² + 0.403x, wave 7.356x² + 0.032x (`results/profile_sweep/fits.json:90,128,166,212,426,502`). FA-2 monolithic at 1M = 5.96 s → 5.96 s/M² (`results/e2e_compare_v11.json`), so the quadratic coefficients are 1.21-1.41× FA-2's [I].

### 2.3 Host-streamed forward (A100-SXM4-80GB, della-l07g7, job 13780966, `results/e2e_compare_v11.json`, `docs/LOG.md:733-737`)

Stream-CQSA arms stream pinned host Q/K/V with the host accumulator (n_par=1, shared_chunks, v11); FA-2/SDPA arms hold Q/K/V on the device (different memory condition).

| N | FA-2 (device) | shipped Stream-CQSA | next (v11) | next / FA-2 | vs shipped | rel.err |
|---|---|---|---|---|---|---|
| 256K | 0.36 s / 1.3 GiB | 2.17 s | 1.17 s / 1.3 GiB | 3.2 | 1.86× | 2.2e-4 |
| 1M | 5.96 s / 5.0 GiB | 15.51 s | 9.49 s / 5.3 GiB | 1.59 | 1.63× | 3.0e-4 |
| 2M | 24.18 s / 10.1 GiB | 50.97 s | 35.31 s / 10.6 GiB | 1.46 | 1.44× | 4.5e-4 |

Wave vs classic with host accumulator (raw, `results/profile_sweep/results.jsonl`): 2M itr=1 wave 35.41-35.76 s vs classic 33.70-34.10 s; 2M itr=2 38.8-39.0 vs 34.4-34.8 s; 1M itr=1 9.41 vs 9.08-9.44 s. Reason: the wave host merge is synchronous per wave (`native_wave.py:348-360`). The OOC-rivals grid (FA-2 resident vs rectangular OOC vs `cqsa_auto` under 10/20/40/80 GiB budgets, jobs 13952418-25) is submitted but held pending the flash-attn rebuild (memory notes `ooc-rivals-study.md`, `stream-cqsa-evict-and-pending.md`); `results/ooc_rivals/` holds only an empty `logs/` (ls, checked here).

### 2.4 Multi-GPU (classic engine, round-robin shard; `results/dist_test_*.json`, checked here). GPU model is not recorded in these JSON rows; the repo's `slurm/run4.slurm` / `run_multinode.slurm` request `--gres=gpu:a100:4` (grep of SBATCH headers, checked here), and `docs/LOG.md:19` lists a100:4 ×79 nodes and one a100:8 node.

| world | itr | acc | N | single (s) | dist (s) | speedup | tasks/rank | Source |
|---|---|---|---|---|---|---|---|---|
| 4 | 1 | gpu | 1M / 2M / 4M | 12.92 / 50.62 / 201.15 | 5.06 / 16.91 / 62.27 | 2.55 / 2.99 / 3.23 | 2 | `results/dist_test_w4_itr1-2_gpu.json` |
| 4 | 2 | gpu | 1M / 2M / 4M | 19.89 / 65.95 / 243.36 | 6.87 / 20.11 / 68.77 | 2.90 / 3.28 / 3.54 | 13 | same |
| 8 (2 nodes) | 2 | gpu | 2M / 4M | 55.83 / 206.10 | 11.78 / 40.42 | 4.74 / 5.10 | 7 | `results/dist_test_w8_itr2_gpu.json` |
| 2 | 2 | (not recorded) | 256K / 1M / 2M | 3.03 / 21.00 / 64.12 | 2.53 / 12.47 / 36.47 | 1.20 / 1.68 / 1.76 | 25 | `results/dist_test_w2_itr2.json` |
| 2 | 1 | cpu (fwd) | 64K / 256K / 1M | 0.42 / 1.28 / 12.71 | 0.51 / 1.41 / 9.93 | 0.83 / 0.91 / 1.28 | 4 | `results/dist_test_w2_itr1_cpu_bwd.json` |
| 2 | 1 | cpu (bwd) | 1M | 43.72 | 29.66 (26.87 local + 2.58 merge) | 1.47 | 4 | `docs/LOG.md:516-522` |

Floor for round-robin: with 7 tasks on 4 ranks the busiest rank holds 2/7 of the work, so the best possible speedup is 3.5× (measured 2.55-3.23×); with 49 tasks on 4 ranks (13 per rank) 3.77× (measured 2.90-3.54×); 49 on 8 (7 per rank) 7.0× (measured 4.74-5.10×) [I, arithmetic]. The planner models the remainder as `dist_imbalance 1.35` + `dist_fixed_s 0.4` + comm at 25 GB/s (`autoconfig.py:221-223, 248-250`) [V]. The planner's recorded verdict: "2 GPUs never beat one FA-2 call, 4 GPUs do from ~1M" (memory note `stream-cqsa-next-work.md`; not re-derived here).

### 2.5 Backward (A100-SXM4-80GB, fp16, H=8, D=64, itr=1, `results/triton_bwd_bench.json`, job 13799930)

| L | causal | FA-2 bwd | CUDA CQS bwd | Triton CQS bwd | CUDA/FA-2 | Triton/CUDA |
|---|---|---|---|---|---|---|
| 7023 | yes | 1.06 ms | 2.13 | 3.45 | 2.01 | 1.61 |
| 14044 | yes | 3.30 | 6.83 | 6.63 | 2.07 | 0.97 |
| 28088 | yes | 11.80 | 23.43 | 19.04 | 1.99 | 0.81 |
| 56175 | yes | 44.08 | 88.50 | 68.11 | 2.01 | 0.77 |
| 56175 | no | 88.57 | 127.86 | 131.51 | 1.44 | 1.03 |

Engine backward (acc=GPU): 256K CUDA 2.48 s vs Triton 1.87 s; 1M 39.12 vs 28.98 s (`results/triton_bwd_bench.json` engine). Wave backward N=131K c=7: 619.4 ms vs classic 639.7 ms vs FA-2 fwd+bwd 327.2 ms; peaks 4.93 vs 2.26 GiB (`results/native/wave_bench.json` bwd_131k_c7). Paper re-run (A100-PCIe-80GB): Triton backward 8.4M 2917 s vs CUDA 4113 s; 16.8M 11522 vs 14420 s (`docs/LOG.md:1265-1276`). Floor: the same 0.78× per subproblem; the backward is "~3× the fwd per subproblem" (`docs/LOG.md:551`) and no backward ncu profile exists (`ls next/logs`, follow-up map).

---

## 3. Where time and memory go

### 3.1 Kernel level (forward, v11, causal, zero pattern, L=56175; ncu job 13780907, A100 PCIe-class node, SM clock locked 1.06 GHz; `next/logs/cqsa_ncu11_13780907.out:157-308`)

| Metric | v11 | FA-2 2.8.3 | Source |
|---|---|---|---|
| executed instructions | 4.03 G | 3.71 G (v3 5.22 G) | `:260`; `cqsa_ncu_13765519.out:104`; `cqsa_ncu_13765554.out:873` |
| warp cycles per issued instruction | 6.58 | 5.20 (non-causal 4.84) | `:231`; `cqsa_ncu_13765554.out:75,229` |
| registers / smem / CTAs per SM | 255 / 50.18 KB / 2 (register- and smem-limited) | 255 / 49.15 KB / 2 | `:127,143-144,211-215`; `cqsa_ncu_13765519.out:125` |
| occupancy theoretical / achieved | 12.5% / 12.3%; 1.97 active, 0.35 eligible warps per scheduler; "No Eligible" 70.1% | - | `:278-302` |
| stall | 43.3% fixed-latency dependency | - | `:231` |
| DRAM / L2 hit / L1 hit / SM throughput | 2.27% / 94.3% / 61.9% / 44.8% | - | `:171-202` |
| SASS size (CQS-on instantiation) | 74.8K instructions (CQS-off 3.7-7.8K), STL 226 / LDL 307 | - | `docs/LOG.md:671, 1004` |

Reading (from `docs/kernel_technical_note.md:100-107` and reader map 1): the kernel is issue-latency bound at 2 warps per scheduler, not memory bound; of the 1.40× at L=56K (align=1), 0.78× is ideal pair work, the live-tile loop is within 9% of FA-2's instruction count, and ~27% slower issue × ~9% more instructions ≈ 1.38× [I]. Contributors named but never separately measured: Check_inf softmax select, apply_mask entry test, column-summary loads inside the search loop, I-cache pressure from the 20× larger body. At small L the mixed-tile per-element path (5.5-5.7× a live tile, `docs/LOG.md:468`) and the never-skipped fully-masked masking-step diagonal tiles dominate (2.9× FA-2 at N=16K); inferred cost of the latter at L=56175 ≈ 4% of causal kernel time (reader map 1, [I]). The `kernel_technical_note.md:104-105` attribution to "the !Is_even_MN column-limit pass on every tile" is not supported by the code (that pass occurs only in the 1-2 masking-step tiles, identically in FA-2) [I, reader map 1].

Non-causal v11 mode-2 (mis-compiled): 23.60 G instructions, L1 hit 34.9%, 11.28 cycles/instr, 34.5% of stalls on branch-target resolution, stack 1024 B in the report (cuobjdump 616-680 B), 0 spill requests (`cqsa_ncu11_13780907.out:20,36-37,44,58-59,78,86-93,106,130`; `docs/LOG.md:704-712`) [V].

Backward (static only): hdim64 fp16 128×128 seqk-parallel instances at REG 255 with STACK 224-312 B vs upstream 96-144 B (`next/logs/01_cuobjdump_res_usage.txt:385-416`; `01_cuobjdump_upstream_fa2.txt:1553-1592`); dead tiles still execute cp_async wait + barriers, dP_sum gather, P convert/store, the full dS scaling loop (with a runtime `cqsa_numden_mode` branch per element), dS convert/store, Q/dO prefetch, LSE reload, and a kBlockM×kHeadDim fp32 atomicAdd of zeros into `dq_accum` (`flash_bwd_kernel.h:515-776`, follow-up map) [V]. The Triton backward does 7 GEMMs per live tile in two kernels yet beats the fused 5-GEMM CUDA kernel by 1.3× at L=56K, so the CUDA loss is the per-tile non-GEMM path, not GEMM work [I].

Wave kernel: W=1 packed 17.91 vs v11 17.68 ms (noise ~0.4 ms); in-place adds ~2% after the block-map prefetch (was 5%) (`docs/LOG.md:1035-1041, 1064-1067`). The c=73 wave at N=131K: one 152 ms kernel + 9 ms merge vs 73 launches over 227 ms in the classic engine; the merge reads lse uncoalesced ("could be halved") (`docs/LOG.md:1051-1056, 1084-1087`) [V].

### 3.2 Engine level

- Split of the 1.63× end-to-end gain at N=1M (host-streamed, v11 vs shipped): pipelined merge/d2h overlap ~1.15×, shared_chunks gather ~1.09×, v11 kernel ~1.19× on the compute stage (`docs/LOG.md:739`) [V].
- Per-launch overheads: the wave removes the classic engine's itr=2 linear term (0.403 → 0.032 s per M tokens, `fits.json:426,502`); at c=73 the wave is 30% faster (`wave_bench.json`); at 1M c=7 the two engines coincide because per-launch costs are <1% of a 1.1 s subproblem [V].
- Planner cost constants (A100-SXM4 v11): `task_overhead_s 3e-3` per subproblem, host gather 0.5 s / 3.1M tokens (shared_chunks), host merge 2.25 s / 3.1M tokens at 8 threads, d2h 0.25 s / 3.1M, device merge 0.10 s / 3.1M (`autoconfig.py:210-219`) [V].
- Host-side pathologies: torch CPU ops with 80 OpenMP threads (19 MB copy 96 ms at 80 threads vs 0.2 ms at 8; `torch.cat` ~200 ms on 56K-element tensors), hence numpy table packing and `CQSA_CPU_THREADS=8` (`native_wave.py:38-40,183-184`; `docs/LOG.md:1018-1021`) [V].
- Backward: kernel-bound (async host scatter buys 0.94-1.01×, `docs/LOG.md:551`); global-dpsum precompute loses 2.4× when streaming from host (N=262144 itr=2: 3985 vs 9630 ms) but the wave backward always uses it (`stable_stream.py:115-135`; `native_wave.py:563-568`) [V].

### 3.3 Memory

| Engine / mode | Peak model (GiB; N, W in M tokens) | R² | Source |
|---|---|---|---|
| classic CUDA itr=1 acc=GPU | 9.043·N | - | `results/profile_sweep/fits.json:92` |
| classic CUDA itr=1 acc=CPU / itr=2 GPU / itr=2 CPU | 5.333·N / 6.832·N / 4.768·N | - | `fits.json:260,428,596` |
| wave CUDA itr=1 acc=GPU | 7.149·N + 1.78·W_wave + 0.031 | 0.99994 | `fits.json:176` |
| wave CUDA itr=2 GPU / itr=1 CPU | 6.902·N + 1.924·W + 0.032 / 3.968·N + 2.396·W - 0.063 | 0.99994 / 0.9993 | `fits.json:344,512` |
| wave Triton itr=1 GPU | 5.572·N + 6.499·W + 0.998 | 0.9968 | `fits.json:222` |

The W term is the fp32 partial output resident for a whole wave (2048 B per packed token at H=8 D=64, ~1.9 GiB per M; padded to W·S in the uniform layout) [I, reader map 2]; v2.2.1's budget cap gives 4 waves and 4.8 GiB instead of 6.6 at 512K for <= 1.3% time (`next/LOG.md:1432-1434`). Wave Triton gathers packed fp16 Q/K/V copies and merges through a [U, maxc, H, D] gather: 13.9 GiB vs 6.6 at 512K c=7, 20.1 GiB at c=133 (`fits.json` c/wave_triton) [V]. Host memory rule of thumb (memory note `slurm-memory-requests.md`): ~6 KiB/token/rank forward and ~7 KiB/token/rank backward for the host-streamed engine, plus pinned staging ×1.5 and a token-major copy when inputs are head-major.

---

## 4. Gaps and known bugs, ranked (by impact on the program's three objectives)

| # | Gap / bug | Evidence | Objective hit |
|---|---|---|---|
| 1 | The kernel baseline is ambiguous: v11 at L≈56K is 1.40× FA-2 on the classic engine's layout (align=1, L=56175) and 0.98× on the wave layout (align=128, L=56064). No same-job A/B of the two layouts exists; the classic engine deliberately builds tasks with `seg_align=0` (`stable_stream.py:1286-1301`). If the 38% is layout, the top kernel item is an engine-side chunk alignment, not the per-element path. | `results/kernel_ab_clean_v11.json`; `results/native/wave_bench.json`; `next/native/bench_wave.py:33`; `native_wave.py:134-139` | A100 kernel |
| 2 | CUDA backward is 2.0× FA-2 (1.44× non-causal) at every L; never profiled (no bwd ncu, no CQSA_BWD_NO_CQS run); dead tiles still run all bookkeeping + zero atomics; runtime-blk verdicts every iteration; dq non-reproducible (6.4e-6..1.1e-5 cross-run drift, `docs/LOG.md:539-551`). Wave engine backward is CUDA-only, so every wave backward pays this. | `results/triton_bwd_bench.json`; `flash_bwd_kernel.h:485-776`; `setup.py:185-189` | all |
| 3 | Non-causal CQS-on instantiations of v10/v11 are mis-compiled by ptxas (23.6 G instructions, 3.4-3.8× slower); production uses a second extension (v9) whose non-causal path is 1.41× FA-2 with masked tiles still costing ~45% of a live tile; the hdim128 wave instantiations and the compile-time non-causal wave mode never left ptxas (>70 min / >5 h). Root cause (unrolled per-element mask path, 74-78K SASS) only hypothesised; no `-lineinfo` investigation. | `docs/LOG.md:689-691, 704-712, 1191-1200`; `cqsa_ncu11_13780907.out` | A100 kernel, H100 kernel (any port keeping the path) |
| 4 | Small-L subproblems (deep itr, N<=64K, or large c) are 2.3-2.9× FA-2: mixed tiles cost 5.5-5.7× a live tile and fully masked masking-step diagonal tiles are never skipped. The wave engine's target regime (large c, 16K-25K subproblems) lives here: c=73 wave kernel is 1.55× monolithic at N=131K. | `results/kernel_ab_clean_v11.json`; `docs/LOG.md:468, 1084-1087`; `flash_fwd_kernel.h:365-374` | A100 kernel |
| 5 | Head dim 128 has no wave forward (ptxas), no kernel A/B, no ncu; bf16 has no kernel timing; every measured number is fp16 hdim64. | `docs/LOG.md:1191-1200`; `benchmarks/kernel_ab.py` (D=64) | "all capabilities" |
| 6 | Multi-device: round-robin over c^itr tasks with no rate model; 4 GPUs reach 2.55-3.54× (floor 3.5-3.77×), 8 GPUs 4.74-5.10× (floor 7.0×); 2 GPUs lose at 64K-256K acc=CPU; the CostModel assumes homogeneous ranks (`per_rank = ceil(n_tasks/world)/n_tasks`); `interconnect_gbs` is a single number; no wave distributed path; no reduce_scatter. | `distributed.py:47-48,72-105`; `autoconfig.py:248-250`; `results/dist_test_*.json` | heterogeneous scheduling |
| 7 | Wave engine memory: fp32 partial output for the whole wave (1.78-1.92 GiB per M packed tokens) makes its peak exceed the classic engine's (6.64 vs 4.74 GiB at 512K); the planner has no wave memory term and routes by subproblem count only; OOM is handled by a retry on the classic engine. | `fits.json:176`; `api.py:424-431` | single-device multi-subproblem |
| 8 | Wave host-accumulator path is synchronous per wave (`torch.cuda.synchronize` + host fold), so `attention(auto)` never uses the wave for acc=CPU; the classic engine's overlapped D2H wins by 2-12%. | `native_wave.py:333-360`; `results.jsonl` | host streaming |
| 9 | Wave in-place reads need N % 128 == 0 and device-resident input; otherwise the wave (W·S rows, padded) is gathered into three zero-initialised fp16 copies; uniform layout pads every subproblem to max_L (unequal children at itr>=2 waste MMA work; unmeasured). | `native_wave.py:491-507`; `flash.h:196-197` | multi-subproblem |
| 10 | Wave backward: gathered only (bwd loop advances Q/dO by pointer decrement), varlen only, CUDA only, fp32 gradients resident (4.93 vs 2.26 GiB at 131K), only 3% faster than classic. | `docs/LOG.md:1086-1090`; `wave_bench.json` | backward |
| 11 | No Hopper-native kernel anywhere: the release wheel compiles the sm80 design for sm_80 and sm_90 (`native/wheel_build.sh`); Triton 3.6 uses none of TMA/wgmma-async/warp specialisation for this kernel; FA-3 has no CQS hooks. | `native/wheel_build.sh`; reader map 3 | H100 |
| 12 | Triton: host-side run-table build ~1 ms per call (`torch.unique` + Python loop with `.tolist()` sync), no autotune, fixed tiles from one A100 sweep, 128-wide backward key tiles pathological (up to 1955 ms), no shared_chunks, no wave backward. | `triton_kernel.py:255-288`; `results/triton_bwd_bench.json` | all |
| 13 | Segmented addressing cost on the live path (div/mod + block_base load per K/V tile) never isolated (kernel_ab passes no block_base); 1 KB dead sCqsBits smem still allocated in v11. | `benchmarks/kernel_ab.py:50-54`; `flash_fwd_launch_template.h:76` | A100 kernel |
| 14 | Dropout, split-KV, local/alibi/softcap template features inherited but untested with CQS; GQA (H_k != H) rejected by wave entry points. | `flash_api.cpp:620-660`; `native/.../flash_api.cpp:2515, 2620` | "all capabilities" |
| 15 | Environment state (2026-09-21): flash-attn 2.8.3 is being rebuilt from source into CQSA311 (job 14240707 pending); 12 held jobs, the notebooks, the OOC-rivals report and the two flash-attn tests depend on it; the Sep-13 `cqsa_native.so` may not import under the rebuilt torch 2.10.0+cu130 (unverified). | memory notes `stream-cqsa-evict-and-pending.md`, `venvs-location.md`; reader map 2 | recipes |

---

## 5. Design hooks

### 5.1 Unified A100 kernel (one binary: causal + non-causal, single and multi-subproblem, in-place, fp32 out)

| Hook | Where | What plugs in |
|---|---|---|
| Mask representation | `interface.py:167-209` (`cqs_block_summaries`), `flash.h:144-181`, `native/.../flash.h:181-200` | Keep int64 bits + 64-token OR/AND summaries unchanged; they are kernel-agnostic. The natural generalisation is a host-built per-row-block live-run table (as Triton's `live_runs`, `triton_kernel.py:255-288`), which removes the in-kernel search and the masking-step special case. |
| KV loop | `csrc/.../flash_fwd_kernel.h:461-569` (first-live search 470-490, loop 492, next-live search 514-521, K prefetch 525-530) | Any new loop must keep FA-2's straight-line body (v3 and the Triton `if` both lost 5-31%); a run table would replace the `while` search with run bounds. |
| Masking-step tiles | `flash_fwd_kernel.h:365-374` computes only `cqs_tile_clear_c` | Add the masked verdict; for a fully masked first tile either fill `acc_s` with -INF and still run the Is_first softmax, or start from the first live tile (accumulation order unchanged if the visit order stays descending). |
| Mixed-tile path | `mask.h:510-700` (`apply_mask`, `cqs_tile_state==1`) | Replace the per-element dependent global load with a per-column packed bit word tested 8 columns at a time, or a 128×128 bitmask built once per tile in smem by 128 threads (deferred idea, `docs/kernel_technical_note.md:72-84`). Outlining (`__noinline__`) or a separate instantiation tests the I-cache hypothesis and may be what unblocks ptxas for non-causal and hdim128. |
| Compile-time dispatch | `flash_fwd_launch_template.h:57-72` (`CQS_MODE_SWITCH`), `flash_fwd_kernel.h:60-78` (`Cqs_mode`, `Wave_ok`) | One translation unit per (causal, wave) flag with `-lineinfo` and `__launch_bounds__`/maxrregcount experiments to retire `csrc_nc`; add hdim128 wave instantiations behind `Wave_ok`. |
| Wave layout | `native/.../flash_fwd_kernel.h:109-125` (per-CTA offsets), `flash_api.cpp:2494-2690` | A causal varlen wave would remove the uniform padding but re-introduces the 12% register regression seen at W=1 (`docs/LOG.md:996-1013`); rematerialising offsets from `blockIdx` (as mode 3 does) is the pattern to keep. |
| fp32 contract | `flash_fwd_kernel.h:573-601`; `native/.../flash_fwd_kernel.h:646-686`; `cqsa_kernel.cu`; `wave_kernels.cu:28-73` | New kernels must emit normalised fp32 out + lse in [B,L,H,D] / [B,H,L] (or [W·S,H,D] / [W,H,S]) so `local_stats_flash`, `wave_merge` and `merge_rows_torch` stay unchanged. |
| In-place reads | `flash_fwd_kernel.h:28-35, 200-217`; `reference.segment_block_base` (`reference.py:583-605`); `native_wave.py:235-247` | Contract: align = 128, every tile size divides it, one base offset per 128-row block; a per-tile map lookup prefetched one tile ahead. |
| Backward | `flash_bwd_kernel.h:485-600` (verdicts/guards), 501-509 (why no `continue`), 648-678 (dS loop with runtime numden branch), 757-766 (dq_accum atomics); `flash_bwd_launch_template.h:78-82,129-135` (deterministic split buffers already wired); `flash_api.cpp:1794-1798` | Port v11's structure with roles swapped (per-CTA column summary, per-m_block row summary prefetched one tile ahead, CTA-uniform verdict), convert pointer decrements to index-based `seg_q(m_block)` so a `continue` is legal, guard the zero atomicAdd, template the numden branch, expose `deterministic=True`. Shared `flash_bwd_kernel.h` means the wave engine inherits every change. |
| Bench/A/B | `next/kernel/mk_variant.sh`, `build.sh`, `benchmarks/kernel_ab.py`, `next/native/bench_wave.py`, `benchmarks/triton_bwd_bench.py`; ncu on a compute node with SM clock locked (`docs/kernel_technical_note.md:132-147`) | Add a bwd A/B (`CQSA_BWD_NO_CQS` build, zero vs real mask) and an align=1 vs align=128 forward A/B in one job. |

### 5.2 Heterogeneous / homogeneous multi-device scheduling

| Hook | Where | What plugs in |
|---|---|---|
| Sharding | `distributed.py:47-48` (`_shard`), 72-82 (`task_subset` = round-robin) | Replace with a cost-aware assignment: LPT/list scheduling over tasks of cost t(L_i, pattern_i, device_r) with per-rank rate; tasks at itr>=2 have unequal L (children of unequal chunk sizes), which the round-robin ignores. |
| Per-rank cost | `autoconfig.py:200-262` (`CostModel.cqsa_time`, homogeneous `per_rank`), `DeviceSpec.model/n_sm/identical_group` (`autoconfig.py:89-96`), `calibrate()` (476-539), cached models per GPU key (`_gpu_key`, 273-293) | Give `DeviceSpec` a rate (`pair_rate` per device model, from `calibrate` per GPU) and make `cqsa_time` sum per-rank makespans; seed curves: v11 t(L) at L=7K/14K/28K/56K (`results/kernel_ab_clean_v11.json`), PCIe-40GB model (`results/costmodel_a100_pcie40_interactive.json`), SXM defaults (`autoconfig.py:206-223`). No H100 or GH200 point exists. |
| Merge | `distributed.py:97-105` (all_reduce MAX/SUM of m, acc, l), 168-175 (chunked all_reduce of gradients); `dist_merge_bytes_per_tok 8192`, `interconnect_gbs 25` default, NVLink detection (`autoconfig.py:145-152`) | A reduce_scatter of per-token partials (LOG.md:746 notes the forward analogue) cuts per-rank memory; per-pair link bandwidth (NVLink vs PCIe vs inter-node) replaces the scalar. |
| Wave in a rank | `native_wave.py:267-277` (`plan_waves`), 685-696 (`ChunkPool.plan`), 280-285 (`_budget_tokens`) | Shard the task list before `plan_waves` (as `distributed.py` shards) so each rank runs waves; `docs/LOG.md:1089-1091` lists this as not done. |
| Entry point | `api.py:82-104` (`attention(..., hardware=, kernel=, plan_only=, overrides=)`), `auto_attention` → `dist_stream_cqsa_forward` (`autoconfig.py:586-630`); `hardware_from_dict` (`autoconfig.py:154`) | The `hardware` dict already carries per-device budgets and models; a heterogeneous plan is a `Plan` with per-rank task lists instead of `world`. |
| Measurements to add | `results/dist_test_*.json` schema (N, itr, world, acc, t_single_s, t_dist_s, speedup, tasks_per_rank) | Record GPU model per rank, per-rank local time, merge time, and link topology (`nvidia-smi topo -m`) in every row. |

### 5.3 H100 kernel on the FA-3 codebase

FA-3 sources: `hopper/` (67 entries incl. `mainloop_fwd_sm90_tma_gmma_ws.hpp`, `flash_fwd_kernel_sm90.h`, `flash_bwd_kernel_sm90.h`, `mainloop_bwd_sm90_tma_gmma_ws.hpp`, `tile_size.h`, `tile_scheduler.hpp`, `block.h`, `paged_kv.h`, `generate_kernels.py`, `instantiations/` with 451 files, `benchmark_attn.py`, `test_flash_attn.py`, `test_flash_attn_bwd_determinism.py`; ls checked here) [V].

Structural parameters the CQS port must fit (checked here unless noted):

| Parameter | Value | Source |
|---|---|---|
| hdim<=64 fp16 forward tile | kBlockM=192, kBlockN=128 when causal/local/paged else 192, MmaPV_is_RS = (kBlockN==128), IntraWGOverlap=true; comment: 192×128 got slower non-causal under Cutlass 3.8 | `hopper/tile_size.h:14-27` |
| hdim<=128 fp16 forward tile | 128 × (128 causal / 176 non-causal), RS, IntraWGOverlap | `hopper/tile_size.h:33-36` |
| pipeline stages | kStages = 2 on Arch >= 90 | `hopper/flash_fwd_launch_template.h:47` |
| warpgroups / registers | NumMmaWarpGroups ∈ {1,2,3}; producer 24 regs (2 WGs with TMA KV), consumer 240; MinBlocksPerMultiprocessor = 1 | `hopper/flash_fwd_kernel_sm90.h:75-83` |
| TMA usage | Use_TMA_Q = !PackGQA; Use_TMA_KV = !PagedKVNonTMA; one producer warp when TMA, a full warpgroup otherwise | `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp:56-57, 121` |
| cluster | ClusterM=2 only for fp16 hdim>=128, non-causal, non-local, non-split, non-paged, non-varlen | `hopper/flash_fwd_launch_template.h:212-219` |
| scheduler | persistent on sm90 unless (Split && !Varlen); Varlen → `VarlenDynamicPersistentTileScheduler` with LPT/Sort and `prepare_varlen_num_blocks` | `hopper/flash_fwd_launch_template.h:63-75, 163-164`; `hopper/flash_prepare_scheduler.cu:219` |
| n_block range | `BlockMN::get_n_block_min_max` → contiguous [n_block_min, n_block_max), split-KV partitions it evenly | `hopper/block.h:14-58` |
| loop sites | producer 608-614 (early exit if empty), 793-861; consumer 977-983, 1060-1163, masking loop 1211-1232 (`n_block_min_causal_local_mask`), 1316-1332; non-overlap variant 1371-1423 | `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp` (grep) |
| overlap structure | "Each step does gemm0 for iter n_block, gemm1 for iter n_block + 1, and softmax for iter n_block" | `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp:1170` |
| paged path | `PagedKVManager` maps `row_idx = n_block*kBlockN + row` through `page_size_divmod`; TMA variant via `get_indices_for_K_TMA` | `hopper/paged_kv.h:80-81, 143-145`; `mainloop:693-694, 752-757` |
| backward | seqk-parallel; producer WG (warp 0 TMA loads, warp 1 `store_dq` TMA bulk reduce-add), 2-3 consumer WGs at 240/160 regs; hdim64 128×128 SdP_swapAB; hdim128 causal 64×128; deterministic via dq_semaphore + LPT scheduler; separate masking iterations only for hdim<=64; Varlen supported | `hopper/flash_bwd_kernel_sm90.h:58-65, 211-264`; `mainloop_bwd_sm90_tma_gmma_ws.hpp:257, 264, 438-677, 704-1046`; `flash_bwd_launch_template.h:27-31, 78-99, 312-364` (follow-up map) |
| build | `*_sm90.cu` get `-gencode arch=compute_sm90a,code=sm_90a`, `*_sm80.cu` sm_80; `FLASH_ATTENTION_DISABLE_*` env flags prune hdims/features; `NVIDIA_TOOLCHAIN_VERSION` pins nvcc 12.6.85 / ptxas 12.8.93 but a CUDA >= 13.0 branch exists; requires CUDA >= 12.3; runs `git submodule update --init ../csrc/cutlass` | `hopper/setup.py:51-76, 488, 496, 508-511` |
| cutlass | `csrc/cutlass` is EMPTY in this checkout (no `include/cutlass/version.h`); compute nodes have no internet (memory note), so the submodule must be fetched on the login node first | ls checked here |

Contiguous-KV assumptions in FA-3 that a live-tile skip list breaks (each must be handled, [V] for the code location, [I] for the consequence):

1. `BlockMN::get_n_block_min_max` returns one contiguous interval and every loop is `for (; n_block >= n_block_min; --n_block)` (`block.h:14-58`; mainloop 861, 1215, 1225, 1232, 1423). A skip list needs an iterator (run table or summary search) at all of these sites.
2. Producer and consumer iterate the interval independently and advance `smem_pipe_write` / `smem_pipe_read` once per iteration (mainloop 752-757, 1172-1176); if only one side skips, `PipelineTmaAsync` deadlocks. Both must derive the same live sequence from the same tables (constant memory or smem).
3. The consumer's software pipeline runs gemm1 for n_block+1 while doing gemm0/softmax for n_block (mainloop 1170); the next live tile must be known one iteration ahead (the analogue of v11's prefetched column summary, `flash_fwd_kernel.h:514-521`).
4. The masking iterations (1211-1232) apply `mask_fn` to the diagonal tiles and the first iteration initialises the online softmax; a fully masked diagonal tile of a non-owner row block must either still run (as sm80 does today) or the initialiser must move to the first live tile.
5. TMA coordinates: `tma_load_K.get_tma_tensor(shape_K)` is indexed by `n_block*kBlockN` in the global tensor (mainloop 656-673, 752-753). In-place reads become one TMA coordinate per tile = `block_base[tile]`, but the tensor map must describe the ORIGINAL [N,H,D] tensor (shape N, not L), and kBlockN must divide the block-map align (128): true for hdim64/128 causal (kBlockN=128), false for the non-causal 192/176 tiles (`tile_size.h:24, 35`).
6. kBlockM = 192 for hdim64 (`tile_size.h:24`): the 64-token summaries tile it (3 words) but the 128-row Q block map does not; Q must be gathered or the map granularity reduced to 64 rows (the contract in `native_wave.py:235-247` says every tile size must divide the align).
7. Cluster multicast of K/V across two CTAs (launch template 212-219) requires identical live lists; it is already disabled for causal and varlen, so CQS should force ClusterM = 1.
8. The persistent/LPT scheduler and `prepare_varlen_num_blocks` estimate work per m_block from the n_block count (launch template 63-75, 163-164); with skipping the estimate is wrong (suboptimal ordering, not incorrect results).
9. Split-KV partitions [n_block_min, n_block_max) evenly (`block.h:52-54`); with skipping the splits are unbalanced; disable Split for CQS.
10. `PagedKVManager` is the existing non-contiguous mechanism (`paged_kv.h:143-145`): with page size 128 the CQS block map is a page table, but `PagedKVNonTMA` gives up TMA and a producer warp (mainloop 121; consumer regs 232 vs 240, `flash_fwd_kernel_sm90.h:83`); whether the TMA paged path (`get_indices_for_K_TMA`, mainloop 752) accepts a 128-row page table without a copy is unverified.
11. The epilogue writes O in the input dtype plus LSE (`epilogue_fwd.hpp`); the wave contract needs fp32 out + lse. FA-3's Split path writes fp32 partial O and LSE for `flash_fwd_combine` (files exist: `hopper/flash_fwd_combine.cu`, `flash_fwd_combine_kernel.h`); whether that format matches `wave_merge`'s is unverified [I].
12. The backward has the same three-way agreement problem (producer load loop 528-549, consumer `bwd_step` 1005-1033, `store_dq` warp 634-660 with `dq_semaphore` in deterministic mode) (follow-up map).

Two implementation routes, both untested here: (a) CUDA/cute port of the CQS live-tile iterator into `CollectiveMainloopFwdSm90` with the FA-3 varlen path carrying the wave packing; (b) Triton 3.6 with tensor descriptors (TMA on sm90, pointer loads on sm80 from one source, `compiler.py:235-236, 302-303`) — no intra-warpgroup pingpong is expressible without Gluon (`core.py:3273-3284`; reader map 3, [I]). The classic sm80 kernel compiled for sm_90 (release wheel) is the fallback and has no H100 measurement.

H100 test bench, ready without hardware: `hopper/benchmark_attn.py` (FA-3 baseline), `hopper/test_flash_attn.py`, `hopper/test_flash_attn_bwd_determinism.py`; the repo's `benchmarks/kernel_ab.py`, `triton_bench.py`, `triton_bwd_bench.py` (L=3N/7, H=8, D=64 grid) with FA-3's `flash_attn_interface` as the reference arm; `tests/test_wave_shapes.py` (dtype × D × causal with build probes) and `tests/test_wave_triton.py`. Missing everywhere: a bit-identity rerun check on dq (`torch.equal`), an align=1 vs align=128 A/B, hdim128 rows.

---

## 6. Build / test / bench recipes and cluster constraints

### 6.1 Environment

- Python env: `/scratch/gpfs/AKEY/yb2807/VENVs/CQSA311` (python 3.11.14, torch 2.10.0+cu130, triton 3.6.0); all envs and pip/conda caches under `VENVs/`, never `/home` (memory note `venvs-location.md`). Triton's own ptxas bundle compiles sm_90a without the system toolkit (`triton/backends/nvidia/bin`, reader map 3).
- CUDA toolkits on the login node: `/usr/local/cuda-{11.8,12,12.6,12.8,12.9,13.0,13.1,13.2,13.3}` (ls checked here); the memory note says CPU nodes lack `/usr/local/cuda-13.0` and compute nodes have no internet. The flash-attn 2.8.3 rebuild (job 14240707, `VENVs/build_flash_attn.slurm`) is pending on gpu-short.
- Dev tree: `next/` used via `PYTHONPATH`; run native scripts with `python -P` (setup stub shadows the package, memory note `stream-cqsa-native-wave.md`).

### 6.2 Builds

| What | Command / set | Time | Source |
|---|---|---|---|
| classic variant | `bash next/kernel/build.sh <v>` (`CQSA_KERNEL_SET=a100_fp16_hdim64_128`), run `next/kernel/watchdog.sh` (ptxas can hang for hours) | 15 min (build.sh comment) / ~35 min (LOG) / ~40 min vis1 MAX_JOBS=8 (memory) | `next/kernel/build.sh:2`; `docs/LOG.md:585`; memory note |
| release classic | `CQSA_KERNEL_SET=common` (fp16+bf16, ~2× the variant time) | ~33 min at MAX_JOBS=16 (shipped) | `docs/LOG.md:43-45, 743-745` |
| native | `bash native/build.sh` (arch 80, `native_common64`, MAX_JOBS=8, NVCC_THREADS=4) | 62 min (LOG) / 35-45 min vis1 MAX_JOBS=4 | `native/build.sh`; `docs/LOG.md:1204-1206, 996-999` |
| release wheel | `native/wheel_build.sh` (`FLASH_ATTN_CUDA_ARCHS='80;90'`, `CQSA_KERNEL_SET=common CQSA_BUILD_NATIVE=1`) | ~2.5 h | `native/wheel_build.sh`; `docs/LOG.md:1214-1216` |
| measurement builds | `CQSA_BWD_NO_CQS=1` (backward CQS compiled out; A/B only, never run) | - | `setup.py:185-189` |
| FA-3 | `cd hopper && python setup.py ...` after `git submodule update --init ../csrc/cutlass` on the login node; prune with `FLASH_ATTENTION_DISABLE_*` (e.g. keep HDIM64/128, fp16, disable SPLIT/PAGEDKV/APPENDKV/LOCAL/SOFTCAP/PACKGQA/FP8/SM80) | unknown here (451 instantiation files) | `hopper/setup.py:51-72, 496` |

Known ptxas hangs: non-causal Cqs_mode 1 at hdim128 (>5 h), hdim128 wave modes (>70 min), non-causal wave mode 4 (>60 min) (`docs/LOG.md:689-691, 1191-1200, 1010-1013`).

### 6.3 Tests

- Full suite: `next/slurm/test_suite.slurm` (48G); 315 pass on 2026-09-21, 12 bf16 failures are the fp16-only dev kernel (memory note `stream-cqsa-evict-and-pending.md`).
- Kernel bit-identity: `benchmarks/kernel_ab.py` (plain/zero/mask, maxdiff vs shipped fp32 out) (`kernel_ab.py:24-58, 112-129`).
- Wave: `next/native/test_wave.py` (30 checks) + `bench_wave.py` via `native/run_test.slurm` (gpu-test, 59 min, 120G, gpu80); `tests/test_wave_shapes.py`, `tests/test_wave_triton.py`.
- Backward: `tests/test_backward_default.py`, `tests/test_devkit_autoconfig.py`; no test asserts backward determinism (follow-up map).
- Distributed: launch with `python -m torch.distributed.run` (the `torchrun` on PATH belongs to another env; memory note); `slurm/run4.slurm`, `run_multinode.slurm` (`--gres=gpu:a100:4`, `--nodes=2`).

### 6.4 Benchmarks and profiling

- Kernel: `benchmarks/kernel_ab.py`; `next/native/bench_wave.py`; `benchmarks/triton_bench.py` (block sweep grid), `triton_bench_large.py`, `triton_bwd_bench.py`.
- Engine: `benchmarks/profile_sweep.py` (`KERNELS` tuple line 26; one subprocess per configuration, 1 warm-up + 5 runs) with `slurm/profile_sweep_{cuda,triton,wave_cuda,wave_triton}.slurm` (`--gres=gpu:a100:1 --constraint="sxm&gpu80"`, 8 CPUs, 32G).
- ncu: `slurm/ncu.slurm`; method: compute node, SM clock locked (1.06 GHz in the existing logs), one kernel module per process (`docs/kernel_technical_note.md:132-147`); logs in `next/logs/cqsa_ncu*.out`, cuobjdump in `next/logs/01_cuobjdump_*.txt`.

### 6.5 Cluster constraints and standing user rules

| Rule | Detail | Source |
|---|---|---|
| Node types | `--qos=gpu-short --constraint=gpu80` = A100 80GB PCIe (della-l01g13..l05g7, 59 nodes); A100-SXM4-80GB only via `--qos=gpu-test --constraint=sxm` (1 h limit; della-l07/l08 + one 8-GPU node); SXM ~15% faster on this workload; never mix node types in one comparison; print `nvidia-smi --query-gpu=name` in every log | memory note `della-gpu-node-types.md`; `next/LOG.md:1404-1421` |
| H100 / GH200 | `docs/LOG.md:19` (2026-09-11): "H100:4/8 also present". Critic [C]: sinfo lists gpu:h100 nodes della-h19g[1-4], h20g[1-5], h21g[1-4] (×4) and della-j11..k18 (×8) in partitions all/pli/pli-lc, and gpu:gh200 nodes della-l06g13 (all, grace) and della-h23g2 (grace); account access unverified. GH200 is aarch64: the x86_64 env, every .so and the wheel are unusable there [I]. |
| Host memory | Size `--mem` from the host tensors (~6 KiB/token/rank fwd, ~7 KiB bwd, ×1.5 pinned staging, + token-major copy for head-major inputs); user was warned by admins for 320-480G requests; read `sacct` MaxRSS of the `.0` step, not `.batch` | memory note `slurm-memory-requests.md` |
| Harness edits | never edit a harness that running `--isolate` jobs re-import (a 13 h measurement was lost) | memory note `harness-edits-under-running-jobs.md` |
| Envs | everything under `VENVs/`; do not tag a release for a single edit; no `pkill -f` on patterns matching the own shell | memory notes `venvs-location.md`, `stream-cqsa-evict-and-pending.md` |
| Login node | load ~330 at the time of writing; small commands only | task statement |
| SBATCH defaults in repo | `--account=akey`, `--qos=gpu-test --time=00:59:00` (10 scripts), `--qos=gpu-short` (3), `--constraint=gpu80` (7), `"sxm&gpu80"` (4), mem 32G-320G | grep of `slurm/*.slurm` headers, checked here |
| Kernel threads | `CQSA_CPU_THREADS=8`; 80 OpenMP threads pathological on della-vis1 | `native_wave.py:38-40` |

---

## 7. Contradictions found and open questions

### 7.1 Contradictions

1. **Kernel baseline.** v11 at L≈56K causal on A100-SXM4: 24.43 ms vs FA-2 17.46 (1.40×; `results/kernel_ab_clean_v11.json`, path (0,), align=1, L=56175, Is_even_MN=0) versus 17.68 ms vs FA-2 18.09 (0.98×; `results/native/wave_bench.json`, `wave_tasks` with `seg_align=128`, `tasks[3]`, L=56064; `next/native/bench_wave.py:33`, `native_wave.py:134-139`). The planner's `kernel_ratio 1.40` (`autoconfig.py:207`) and the calibrated 1.30-1.50 (`results/costmodel_*.json`, `quorum_axis_1M.json`) encode the first. Layout (mixed tiles, ragged tile, two masking steps) is the likely cause [I]; run-to-run noise is ~0.4 ms (`docs/LOG.md:1067`). Not resolved by any same-job A/B.
2. **FA-2 baseline scatter.** 17.46 / 16.97 / 16.93 / 18.09 ms at L≈56K on "A100-SXM4-80GB" (`kernel_ab_clean_v11.json`, `kernel_baseline.json`, `triton_bench.json`, `wave_bench.json`); node/clock state is not recorded per job; ncu jobs ran on a PCIe-class node at a locked 1.06 GHz.
3. **Build time.** ~15 min (`next/kernel/build.sh:2`) vs ~35 min (`docs/LOG.md:585`) vs ~33 min at MAX_JOBS=16 (`docs/LOG.md:43-45`) vs ~40 min (memory note).
4. **Wave host-accumulator deficit.** "5-7% faster" (`README.md:202`, `next/LOG.md:1423`) vs raw 2-12% (`results/profile_sweep/results.jsonl`).
5. **Skippable fraction at itr=2.** `flash_bwd_kernel.h:492-495` says 38%; the forward census gives 22.1% / 31.9% for (0,0) / (0,1) (`docs/LOG.md:329-335`); the backward's `cqs_tile_clear` uses a per-warp window of MMA_N_SdP·16 columns, a different geometry (`flash_bwd_kernel.h:485-600`).
6. **`CQSA_SCATTER_ASYNC`.** Comment says "next/: default ON", default is '0' (`stable_stream.py:105-113`; `docs/LOG.md:551`).
7. **Toolchain location.** Task statement: nvcc 13.0 on GPU nodes only; login node has `/usr/local/cuda-13.0/bin` (ls checked here); memory note: CPU nodes lack it.
8. **H100 availability.** User premise "in a few weeks" vs `docs/LOG.md:19` and the critic's sinfo listing [C]; account access to `all`/`pli`/`grace` unverified.
9. **Backward determinism.** LOG claims the wave's gradients are "reduced deterministically" (`docs/LOG.md:968-975`) while `bwd_wave` hard-codes `deterministic=false` with atomic `dq_accum` (`native/.../flash_api.cpp:2751`) and the classic path measured 6.4e-6..1.1e-5 cross-run dq drift (`docs/LOG.md:539-551`); tests bound wave-partition differences at 2e-3 only (`next/native/test_wave.py:144-149`).
10. **kernel_technical_note attribution.** "!Is_even_MN column-limit pass on every tile" (`docs/kernel_technical_note.md:104-105`) is not supported by `flash_fwd_kernel.h:533-536` / `mask.h:510-545` (steady loop uses `Is_even_MN=true`, as upstream) [I].

### 7.2 Open questions (ranked by decision impact)

1. Which layout is the design target, and what does v11 cost on each in one job (align=1 L=56175 vs align=128 L=56064, plus FA-2 on both)? What do 128-aligned chunk boundaries cost at itr>=2 (child sizes, coverage; `stable_stream.py:1288-1294` explains why the classic engine avoids them)?
2. Backward: what fraction of the 2× is dead-tile bookkeeping/zero-atomics vs live-tile slowdown vs spills? Needs `CQSA_BWD_NO_CQS` timing, zero-vs-real mask timing and a bwd ncu run (none exist). Is per-wave dq bit-reproducible? Is the intended end state a native backward for both engines or Triton-classic + CUDA-wave?
3. What does ptxas demote in the non-causal and hdim128 CQS-on instantiations (0 spill requests, L1 hit 35%, branch-target stalls; `cqsa_ncu11_13780907.out`)? Does outlining the per-element path or separate translation units fix it? This decides whether the sm80 design can be unified and whether it can be ported at all.
4. How much of the live-tile 6.58 vs 5.20 cycles/instr is Check_inf, the apply_mask entry, the search-loop loads, or I-cache misses? Can occupancy be raised above 2 CTAs/SM (255 regs) for the CQS-on path, e.g. 128×64 tiles?
5. Which H100/GH200 partitions can this account use today, and if GH200 is the target, what aarch64 torch/triton stack exists on Della? Does FA-3's hopper tree build under nvcc 13.0 with the (not yet fetched) cutlass submodule?
6. Does FA-3's producer/consumer pipeline tolerate skipped stages (both sides skipping the same n_blocks) without breaking `PipelineTmaAsync` accounting, and can the TMA paged path serve a 128-row block map without losing TMA?
7. For heterogeneous scheduling: what are t(L, pattern) per device type (A100 PCIe, SXM4, H100, GH200), per-link bandwidths (NVLink / PCIe / inter-node) and the merge cost per node type? Only A100 points exist; per-rank local times are not recorded in `dist_test_*.json`.
8. hdim128 and bf16: kernel and engine timings, tile configs actually selected, and whether the wave modes can be compiled (no numbers exist).
9. Wave memory: how much of the 7.15 GiB/M a-term and the W-term is uniform-layout padding vs scratch, and should the planner get a wave memory term rather than routing by subproblem count?
10. Is the Sep-13 `cqsa_native.so` importable under the rebuilt CQSA311 (torch 2.10.0+cu130), and does the pending flash-attn 2.8.3 rebuild (job 14240707) succeed within 10 h under nvcc 13.0?
11. Would a Triton-on-Hopper kernel (descriptor loads → TMA, tl.dot → wgmma if the backend chooses it; unverifiable from the Python tree) be competitive enough to make a cute/FA-3 port unnecessary?
