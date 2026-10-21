/******************************************************************************
 * next/native_v2: the ALIGNED CQS-on forward on the uniform wave layout.
 *
 * v12f's ALIGNED forward (kernels/v12f/csrc/flash_attn/src/flash_fwd_aligned.h)
 * for W >= 1 subproblems per launch: grid (S / kBlockM, W, H), subproblem bidb
 * reads its run table, block map and route word at bidb * stride
 * (Flash_fwd_params::cqs_wave_*; every stride is 0 for a classic call, which
 * is thus a W=1 wave with no extra work), and its real length lens[bidb] from
 * BlockInfo. On a 128-aligned CQS task every tile the forward visits is fully
 * live or fully masked (results/ultimate/align_census.json: mixed 0.000%), so
 * the CQS-on kernel needs neither the per-element mask path nor any per-tile
 * verdict. The census (cqs_census.cu) lists each row block's live column tiles
 * as runs in descending n -- the order v11 visits them in -- and this kernel
 * is upstream's KV loop driven by that list: masked tiles, including the fully
 * masked diagonal tile of a non-owner row block, are simply absent. A
 * subproblem whose census finds a mixed tile runs on LEGACY (v11's CQS forward
 * with the same per-subproblem offsets, flash_fwd_kernel.h) instead; both
 * kernels are launched and each CTA reads its subproblem's route word.
 *
 * In-place reads (cqs_block_base): the next live tile's block-map delta is
 * looked up together with the run-table step, before the QK GEMM whose result
 * the K prefetch waits for, so the dependent global load never sits between
 * the barrier and the cp.async issue (the native wave's measured 5% at L=56K).
 *
 * Bit-identity with v11: the same tiles are accumulated in the same order
 * with the same instructions. The one difference is that the online softmax
 * starts from (row_max, row_sum, acc_o) = (-inf, 0, 0) and always takes the
 * non-Is_first, Check_inf rescale, because the first visited tile need not be
 * the diagonal one. That reproduces Is_first exactly: max(-inf, x) = x, the
 * Check_inf guard maps an all -inf row to 0 so the scale is exp2(-inf) = 0,
 * then row_sum = 0 * 0 + s and acc_o = 0 * 0 before the PV MMA. Skipping a
 * fully masked tile is exact as well: v11 computes it as all -inf scores,
 * which rescale by exp2(0) = 1 and add exp2(-inf) = 0 (or, before any live
 * tile, leave (-inf, 0, 0) unchanged).
 *
 * Caveat (F23 review): the bit-identity holds for FINITE K/V. v11 runs the
 * fully masked tiles of the causal masking window (e.g. the diagonal tile of
 * a non-owner row block) as P = 0 followed by P @ V, so an inf/NaN in those V
 * rows makes v11's rows NaN (0 * inf); ALIGNED never loads them, so its rows
 * stay finite and match the masked math reference. All finite elements agree.
 ******************************************************************************/

#pragma once

#include "flash_fwd_launch_template.h"
#include "cqs_census.h"

namespace FLASH_NAMESPACE {

using namespace cute;

// Production configuration only: no dropout, local window, alibi, softcap or
// returned softmax, and d == kHeadDim (flash_api.cpp routes anything else to
// LEGACY). seqlen_q == seqlen_k per subproblem, as the group-bits mode
// requires, so the causal n_block_max is never below 1 and there is no
// empty-row-block exit.
template<typename Kernel_traits, bool Is_causal, bool Is_even_MN, typename Params>
inline __device__ void compute_attn_1rowblock_aligned(const Params &params, const int bidb, const int bidh, const int m_block) {
    constexpr bool Is_even_K = true;

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;
    // A caller-stated lens_even that is false would make this instantiation
    // load the ragged tile unpredicated (in place: rows past the tensor); a
    // loud failure, not a silent one.
    if constexpr (Is_even_MN) {
        if (binfo.actual_seqlen_q % kBlockM != 0 || binfo.actual_seqlen_k % kBlockN != 0) { __trap(); }
    }

    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if (Is_causal) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q, kBlockN));
    }

    // This subproblem's tables (offsets rematerialised from blockIdx, not held).
    const int *cqs_bb_p = params.cqs_block_base == nullptr ? nullptr
        : params.cqs_block_base + bidb * params.cqs_wave_bb_stride;
    const int cqs_seg_align = params.cqs_seg_align;
    // This row block's live tiles: run {hi, lo} covers hi, hi - 1, ..., lo.
    // Only a run boundary reads the table again, and that load is issued
    // before the QK GEMM whose result it is needed after.
    const int2 *run_next = params.cqs_runs + (bidb * params.cqs_wave_run_stride + m_block) * kCqsRunSlots;
    const int2 run0 = run_next[0];
    ++run_next;
    // Block-map delta of a key tile (0 when reading a packed wave).
    auto seg_delta_of = [&](int n) { return cqs_bb_p == nullptr ? 0 : cqs_seg_delta(cqs_bb_p, cqs_seg_align, n * kBlockN); };

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                          + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    if (cqs_bb_p != nullptr) {
        gQ.data() = gQ.data()
                  + cqs_seg_delta(cqs_bb_p, cqs_seg_align, m_block * kBlockM) * params.q_row_stride;
    }
    Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr)
                                          + binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.k_row_stride, params.k_head_stride, _1{}));
    Tensor gK = local_tile(mK(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(_, 0));  // (kBlockN, kHeadDim, nblocksN)
    Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr)
                                          + binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.v_row_stride, params.v_head_stride, _1{}));
    Tensor gV = local_tile(mV(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(_, 0));  // (kBlockN, kHeadDim, nblocksN)

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    // Careful we're using the same smem for sQ and sK | sV if Share_Q_K_smem;
    Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)),
                            typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K, nblocksN)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K, nblocksN)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    // Segmented K/V, as in v11: same tile, shifted base pointer; the delta is
    // supplied by the caller (looked up one tile ahead).
    auto seg_k = [&](int n, int delta) {
        auto t = tKgK(_, _, _, n);
        if (cqs_bb_p != nullptr) { t.data() = t.data() + delta * params.k_row_stride; }
        return t;
    };
    auto seg_v = [&](int n, int delta) {
        auto t = tVgV(_, _, _, n);
        if (cqs_bb_p != nullptr) { t.data() = t.data() + delta * params.v_row_stride; }
        return t;
    };

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    // Identity layouts for the predicated (ragged) copies; K is always even.
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    clear(acc_o);

    FLASH_NAMESPACE::Softmax<2 * size<1>(acc_o)> softmax;
    cute::fill(softmax.row_max, -INFINITY);
    cute::fill(softmax.row_sum, 0.f);

    // A row block with no live tile (a leading non-owner chunk under causal)
    // loads nothing and goes straight to the epilogue, which writes what v11
    // writes for fully masked rows: out 0, lse +inf.
    if (run0.x >= 0) {
        int n_block = run0.x;
        int run_lo = run0.y;
        int delta_cur = seg_delta_of(n_block);
        // The next live tile after n_block: one lower inside the run, else the
        // next run's top (-1 after the last run: the sentinel).
        auto next_live = [&]() {
            if (n_block > run_lo) { return n_block - 1; }
            const int2 r = *run_next;
            ++run_next;
            run_lo = r.y;
            return r.x;
        };

        // Prologue

        // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
        FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                           binfo.actual_seqlen_q - m_block * kBlockM);
        if (Kernel_traits::Is_Q_in_regs) { cute::cp_async_fence(); }

        if (Kernel_traits::Share_Q_K_smem) {
            FLASH_NAMESPACE::cp_async_wait<0>();
            __syncthreads();
            Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
            CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
            cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
            __syncthreads();
        }

        // The first live tile may be the ragged last one: predicated load.
        // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
        FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, seg_k(n_block, delta_cur), tKsK, tKVcKV, tKVpKV,
                                           binfo.actual_seqlen_k - n_block * kBlockN);
        cute::cp_async_fence();

        if (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
            FLASH_NAMESPACE::cp_async_wait<1>();
            __syncthreads();
            Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
            CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
            cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        }

        FLASH_NAMESPACE::Mask<Is_causal, /*Is_local=*/false, /*Has_alibi=*/false> mask(
            binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right);

        // Masking steps: the tiles upstream masks (the causal diagonal, the
        // ragged last tile) are the top n_masking_steps tiles of the row, as
        // in v11; the live ones among them take the masked body. Only the
        // first visited tile can be the ragged one (it is the highest n).
        constexpr int n_masking_steps = !Is_causal
            ? 1
            : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
        const int n_mask_lo = std::max(n_block_max - n_masking_steps, 0);
        #pragma unroll
        for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step) {
            if (n_block < n_mask_lo) { break; }
            Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
            clear(acc_s);
            FLASH_NAMESPACE::cp_async_wait<0>();
            __syncthreads();

            if (masking_step > 0) {
                FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, seg_v(n_block, delta_cur), tVsV, tKVcKV, tKVpKV);
            } else {
                // Clear the smem tiles to account for predicated off loads
                FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
                    gmem_tiled_copy_QKV, seg_v(n_block, delta_cur), tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
                );
            }
            cute::cp_async_fence();
            const int n_next = next_live();
            const int delta_next = n_next >= 0 ? seg_delta_of(n_next) : 0;

            FLASH_NAMESPACE::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
                acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
                smem_thr_copy_Q, smem_thr_copy_K
            );

            mask.template apply_mask<Is_causal, Is_even_MN>(
                acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16,
                m_block * kBlockM, kBlockM, kBlockN, /*cqs_tile_state=*/0
            );

            FLASH_NAMESPACE::cp_async_wait<0>();
            __syncthreads();
            if (n_next >= 0) {
                FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, seg_k(n_next, delta_next), tKsK, tKVcKV, tKVpKV);
                // This cp_async_fence needs to be in the if block, otherwise the synchronization
                // isn't right and we get race conditions.
                cute::cp_async_fence();
            }

            softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/true>(
                acc_s, acc_o, params.scale_softmax_log2);

            // Convert acc_s from fp32 to fp16/bf16
            Tensor rP = FLASH_NAMESPACE::convert_type<Element>(acc_s);
            // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
            // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
            Tensor tOrP = make_tensor(rP.data(), FLASH_NAMESPACE::convert_layout_acc_Aregs<typename Kernel_traits::TiledMma>(rP.layout()));
            FLASH_NAMESPACE::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);

            n_block = n_next;
            delta_cur = delta_next;
        }

        // Steady state: upstream's straight-line body, no mask.
        while (n_block >= 0) {
            Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
            clear(acc_s);
            FLASH_NAMESPACE::cp_async_wait<0>();
            __syncthreads();
            FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, seg_v(n_block, delta_cur), tVsV, tKVcKV, tKVpKV);
            cute::cp_async_fence();
            const int n_next = next_live();
            const int delta_next = n_next >= 0 ? seg_delta_of(n_next) : 0;

            FLASH_NAMESPACE::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
                acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
                smem_thr_copy_Q, smem_thr_copy_K
            );

            FLASH_NAMESPACE::cp_async_wait<0>();
            __syncthreads();
            if (n_next >= 0) {
                FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, seg_k(n_next, delta_next), tKsK, tKVcKV, tKVpKV);
                // This cp_async_fence needs to be in the if block, otherwise the synchronization
                // isn't right and we get race conditions.
                cute::cp_async_fence();
            }

            softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/true>(
                acc_s, acc_o, params.scale_softmax_log2);

            Tensor rP = FLASH_NAMESPACE::convert_type<Element>(acc_s);
            // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
            // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
            Tensor tOrP = make_tensor(rP.data(), FLASH_NAMESPACE::convert_layout_acc_Aregs<typename Kernel_traits::TiledMma>(rP.layout()));
            FLASH_NAMESPACE::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);

            n_block = n_next;
            delta_cur = delta_next;
        }
    }

    // Epilogue: v11's (fp32 out from the MMA registers, lse, and -- unless the
    // caller asked for the fp32 copy alone, as wave launches do -- the
    // Element out through smem).

    Tensor lse = softmax.template normalize_softmax_lse</*Is_dropout=*/false>(acc_o, params.scale_softmax, params.rp_dropout);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    if (params.cqs_out_acc_ptr != nullptr) {
        Tensor mOacc = make_tensor(
            make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.cqs_out_acc_ptr)
                          + binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
            make_shape(binfo.actual_seqlen_q, params.h, params.d),
            make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gOacc = local_tile(mOacc(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                  make_coord(m_block, 0));
        Tensor tOgOacc = thr_mma.partition_C(gOacc);
        const int row_limit = binfo.actual_seqlen_q - m_block * kBlockM;
        #pragma unroll
        for (int i = 0; i < size(acc_o); ++i) {
            const int row = get<0>(taccOcO(i));
            const int col = get<1>(taccOcO(i));
            if (row < row_limit && col < params.d) { tOgOacc(i) = acc_o(i); }
        }
    }

    static_assert(decltype(size<0>(taccOcO))::value == 4);
    // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
    Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
    CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M
    Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);
    if (params.o_ptr == nullptr) {
        if (get<1>(taccOcO_row(0)) == 0) {
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = get<0>(taccOcO_row(mi));
                if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
            }
        }
        return;
    }

    // Convert acc_o from fp32 to fp16/bf16
    Tensor rO = FLASH_NAMESPACE::convert_type<Element>(acc_o);
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});    // (SMEM_M,SMEM_N)
    // Partition sO to match the accumulator partitioning
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // sO has the same size as sQ, so we don't need to sync here.
    if (Kernel_traits::Share_Q_K_smem) { __syncthreads(); }

    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                          + binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    __syncthreads();

    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    if (get<1>(taccOcO_row(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO_row(mi));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
        }
    }

    // Construct identity layout for sO
    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
    );
}

template<typename Kernel_traits, bool Is_causal, bool Is_even_MN>
__global__ void flash_fwd_aligned_kernel(KERNEL_PARAM_MODIFIER const Flash_fwd_params params) {
    #if defined(ARCH_SUPPORTS_FLASH)
        // The census found a mixed tile or overflowed a run table of this
        // subproblem: LEGACY, launched right after this kernel, computes it.
        if (__ldg(params.cqs_route + blockIdx.y * params.cqs_wave_route_stride) != 0) { return; }
        const int m_block = params.cqs_reverse_m ? int(gridDim.x) - 1 - int(blockIdx.x) : int(blockIdx.x);
        FLASH_NAMESPACE::compute_attn_1rowblock_aligned<Kernel_traits, Is_causal, Is_even_MN>(
            params, blockIdx.y, blockIdx.z, m_block);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

// Whether every subproblem's rows and keys fill whole tiles (the uniform
// layout's S is a multiple of 128 by contract; lens decide the rest).
template<typename Kernel_traits>
inline bool cqs_wave_even_mn(const Flash_fwd_params &params) {
    static_assert(128 % Kernel_traits::kBlockM == 0 && 128 % Kernel_traits::kBlockN == 0,
                  "cqs_wave_lens_even states lengths are multiples of 128");
    return params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0
        && (params.cqs_wave_lens == nullptr || params.cqs_wave_lens_even);
}

template<typename Kernel_traits, bool Is_causal>
void run_flash_fwd_aligned(Flash_fwd_params &params, cudaStream_t stream) {
    // One census per table set: the subproblems of a wave, or the one set a
    // classic call's batch entries share (every stride 0).
    run_cqs_fwd_census(params, Kernel_traits::kBlockM, Kernel_traits::kBlockN,
                       params.cqs_wave_route_stride != 0 ? params.b : 1, stream);
    // No sCqsBits staging area: this kernel never reads group bits.
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.b, params.h);
    BOOL_SWITCH(cqs_wave_even_mn<Kernel_traits>(params), IsEvenMNConst, [&] {
        auto kernel = &flash_fwd_aligned_kernel<Kernel_traits, Is_causal, IsEvenMNConst>;
        if (smem_size >= 48 * 1024) {
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        }
        kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

// The entry of a forward translation unit (flash_fwd_hdim*_sm80.cu): CQS off
// runs upstream's instantiations, CQS on the census and ALIGNED.
template<typename T, int Headdim, bool Is_causal>
void run_mha_fwd_plain_or_aligned(Flash_fwd_params &params, cudaStream_t stream) {
    if (params.cqs_enabled) {
        using Tile = CqsAlignedTile<Headdim>;
        run_flash_fwd_aligned<Flash_fwd_kernel_traits<Headdim, Tile::kBlockM, Tile::kBlockN, 4, false, false, T>, Is_causal>(
            params, stream);
    } else if constexpr (Headdim == 64) {
        run_mha_fwd_hdim64<T, Is_causal>(params, stream);
    } else {
        static_assert(Headdim == 128, "native_v2 builds the forward for head dims 64 and 128 only");
        run_mha_fwd_hdim128<T, Is_causal>(params, stream);
    }
}

}  // namespace FLASH_NAMESPACE
