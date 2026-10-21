/******************************************************************************
 * next/native_v2: the ALIGNED backward on the uniform wave layout (and, without
 * a run table, the CQS-off one).
 *
 * v12b's ALIGNED backward (kernels/v12b/csrc/flash_attn/src/flash_bwd_aligned.h)
 * for W >= 1 subproblems per launch: grid (S / kBlockN, W, H), subproblem bidb
 * reads its run table and route word at bidb * stride
 * (Flash_bwd_params::cqs_wave_*, 0 for a classic call) and its real length
 * lens[bidb] from BlockInfo; lse / dP_sum / dQaccum sit in the batch layout
 * [W, H, S]. On a 128-aligned CQS task every tile the backward visits is fully
 * live or fully masked (results/ultimate/align_census.json: mixed 0.000%), so
 * the CQS-on backward needs neither the per-element mask path nor any
 * per-tile verdict. The census (cqs_census_bwd.cu) lists each column block's
 * live m blocks as runs in descending m -- the order v11's backward visits
 * them in -- and this kernel is upstream FlashAttention-2's seqk-parallel
 * backward loop driven by that list: the function below IS the text of
 * compute_dq_dk_dv_1colblock as v11x compiles it (-DCQSA_BWD_NO_CQS:
 * upstream's loop with our contract), with the `#ifdef CQSA_BWD_NO_CQS` sides
 * selected and every change marked `next/ v12b` and folded away by
 * `if constexpr` / constant `m_step == 1` when Use_runs = false, so the
 * CQS-off instantiation is v11x's loop (hand-specialised copies ran 3-5%
 * slower; docs/ultimate/progress/P2_backward_kernel.md). With runs: the first
 * visited m block is the first run's hi; the next one is decided where the Q
 * prefetch first needs it (m - 1 inside a run, the next run's hi at a
 * boundary); the Q / dO / LSE / dPsum / dQaccum pointers take upstream's
 * one-block step plus the gap (pointer arithmetic, no predication); the sQ
 * double buffer toggles per VISITED tile instead of by m parity. Fully masked
 * tiles, including the fully masked diagonal tile of a non-owner column
 * block, are absent; the owner diagonal takes upstream's causal mask and the
 * ragged tail upstream's !Is_even_MN path. A subproblem whose census finds a
 * mixed tile (or more runs than slots) runs on LEGACY (v11's CQS backward with
 * the same per-subproblem offsets) instead, decided on the device. Dpsum_smem:
 * dP_sum staged in smem by cp.async next to the Q prefetch instead of read
 * from global memory (default for CQS-on calls; env CQSA_WAVE_DPSUM_SMEM=0/1
 * overrides it per call for A/B runs).
 *
 * Contract kept from v11 (the engine's): global lse in, caller-supplied
 * dsoftmax_sum (or the preprocess pass), fp32 dq_accum by atomics, dk/dv
 * written in Element. Production configuration only: no dropout, local
 * window, alibi, softcap, varlen or numden mode, d == kHeadDim
 * (cqs_bwd_aligned_eligible); anything else runs on LEGACY.
 *
 * Exactness vs v11's CQS backward: a live tile runs the same instructions on
 * the same data in the same (descending m) order, so acc_dk / acc_dv are
 * bit-identical; a dead tile contributes exact zeros in v11 (its GEMMs are
 * guarded off, or, for a ragged dead tile v11 cannot prove dead, P == 0
 * everywhere), so leaving it out changes nothing. dQ is accumulated by fp32
 * atomics across column CTAs in any case (non-deterministic order).
 ******************************************************************************/

#pragma once

#include <cstdlib>   // getenv (the Dpsum_smem A/B switch)

#include "flash_bwd_launch_template.h"
#include "cqs_census_bwd.h"

namespace FLASH_NAMESPACE {

using namespace cute;

// Dpsum_smem: dP_sum rows are staged in shared memory with 16-byte cp.async (the same
// commit groups as the Q / dO prefetches), past the kernel's regular smem.
__device__ __forceinline__ void cqs_cp_async_16(void *smem_dst, const void *gmem_src) {
    const uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(gmem_src));
}

// Bytes of shared memory the ALIGNED kernel adds to kSmemSize1colblock: two
// buffers of kBlockM dP_sum values (the current tile's and the next one's).
template<typename Kernel_traits>
constexpr int kCqsBwdDpsumSmem = 2 * Kernel_traits::kBlockM * int(sizeof(typename Kernel_traits::ElementAccum));

// Default of the dP_sum source when CQSA_V12B_DPSUM_SMEM is unset, per path, from job 14323878 (A100-SXM4-80GB,
// results/ultimate/bwd_ab.md; both sources are bit-identical): CQS-on (run table) smem -- equal at hdim64 causal,
// 1.6-2.1% faster at hdim128 causal and hdim64 non-causal; CQS-off global (upstream's reads) -- equal at hdim64,
// smem 2.8% slower at hdim128 causal.
constexpr bool kCqsBwdDpsumSmemCqsOn = true;
constexpr bool kCqsBwdDpsumSmemCqsOff = false;

// next/ v12b: compute_dq_dk_dv_1colblock of v11x (CQSA_BWD_NO_CQS side), extended with Use_runs / Dpsum_smem.
template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Is_first, bool Is_last, bool Seq_parallel, bool Use_runs, bool Dpsum_smem, typename Params>
inline __device__ void compute_dq_dk_dv_1colblock_aligned(const Params &params, const int bidb, const int bidh, const int n_block) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int MMA_N_SdP = kBlockN / decltype(typename Kernel_traits::TiledMmaSdP{}.template tile_size_mnk<1>())::value;
    constexpr int AtomLayoutMS = Kernel_traits::AtomLayoutMSdP;
    constexpr bool Double_buffer = !Kernel_traits::No_double_buffer;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (n_block * kBlockN >= binfo.actual_seqlen_k) return;

    int m_block_max = cute::ceil_div(binfo.actual_seqlen_q, kBlockM);
    if (Is_local) {
        m_block_max = std::min(m_block_max, cute::ceil_div((n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left, kBlockM));
    }

    // next/ v12b: the first m block this CTA visits: upstream's m_block_max - 1, or with runs the column's
    // first run's hi (-1 when no tile of the column is live; then only the dK = dV = 0 exit below runs).
    [[maybe_unused]] int2 run0 = make_int2(0, 0);
    // The column's run table of this subproblem (offset rematerialised from blockIdx).
    [[maybe_unused]] const int2 *col_runs = params.cqs_bwd_runs
        + (bidb * params.cqs_wave_bwd_run_stride + n_block) * kCqsBwdRunSlots;
    if constexpr (Use_runs) { run0 = col_runs[0]; }
    const int m_first = Use_runs ? run0.x : m_block_max - 1;

    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + m_first * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
        + n_block * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
        + n_block * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_do = binfo.q_offset(params.do_batch_stride, params.do_row_stride, bidb)
        + m_first * kBlockM * params.do_row_stride + bidh * params.do_head_stride;
    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + m_first * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb)
        + m_first * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
    const index_t row_offset_dq_accum = binfo.q_offset(params.seqlen_q_rounded * params.h * params.d_rounded, params.h * params.d_rounded, bidb)
        + (m_first * kBlockM + (params.cu_seqlens_q == nullptr ? 0 : 128ll * bidb)) * params.h * params.d_rounded + bidh * params.d_rounded
        // If deterministic, each thread block will do atomicAdd to a different dQ_accum buffer.
        + (!params.deterministic ? 0 : blockIdx.x * params.dq_accum_split_stride);
    const index_t row_offset_lse = (params.unpadded_lse? bidh * params.total_q + binfo.q_offset(params.seqlen_q, 1, bidb): (bidb * params.h + bidh) * params.seqlen_q) + m_first * kBlockM;
    // Regarding 128 * params.b see a comment in mha_varlen_bwd about padding of dq_accum and softmax_d
    const index_t row_offset_dpsum = (params.unpadded_lse? bidh * (params.total_q + 128 * params.b) + binfo.q_offset(params.seqlen_q_rounded, 1, bidb) + 128 * bidb: (bidb * params.h + bidh) * params.seqlen_q_rounded) + m_first * kBlockM;

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gdO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.do_ptr) + row_offset_do),
                             Shape<Int<kBlockM>, Int<kHeadDim>>{},
                             make_stride(params.do_row_stride, _1{}));
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.o_row_stride, _1{}));
    Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
                             Shape<Int<kBlockM>, Int<kHeadDim>>{},
                             make_stride(params.dq_row_stride, _1{}));
    Tensor gdQaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dq_accum_ptr) + row_offset_dq_accum),
                                  Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                  make_stride(params.h * params.d_rounded, _1{}));
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                              Shape<Int<kBlockM>>{}, Stride<_1>{});
    Tensor gdPsum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dsoftmax_sum) + row_offset_dpsum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQdO{});
    Tensor sQt = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQdOtransposed{});
    Tensor sQtNoSwizzle = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQdOtransposedNoSwizzle{});
    // Double buffer for sQ
    Tensor sdO = make_tensor(sQ.data() + (Double_buffer ? 2 : 1) * size(sQ), typename Kernel_traits::SmemLayoutQdO{});
    Tensor sdOt = make_tensor(sdO.data(), typename Kernel_traits::SmemLayoutQdOtransposed{});
    Tensor sdOtransposedNoSwizzle = make_tensor(sdO.data(),
                                                typename Kernel_traits::SmemLayoutQdOtransposedNoSwizzle{});
    Tensor sK = make_tensor(sdO.data() + size(sdO), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sKt = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKtransposed{});
    Tensor sKtNoSwizzle = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKtransposedNoSwizzle{});
    Tensor sdS = make_tensor(!Kernel_traits::Is_V_in_regs ? sV.data() + size(sV) : sK.data() + size(sK),
                             typename Kernel_traits::SmemLayoutPdS{});
    Tensor sdSt = make_tensor(sdS.data(), typename Kernel_traits::SmemLayoutPdStransposed{});
    Tensor sdStNoSwizzle = make_tensor(sdS.data(), typename Kernel_traits::SmemLayoutPdStransposedNoSwizzle{});
    Tensor sP = make_tensor(sdS.data() + size(sdS), typename Kernel_traits::SmemLayoutPdS{});
    Tensor sPt = make_tensor(sP.data(), typename Kernel_traits::SmemLayoutPdStransposed{});
    Tensor sPtNoSwizzle = make_tensor(sP.data(), typename Kernel_traits::SmemLayoutPdStransposedNoSwizzle{});
    // sP and sdQ share the same memory so be careful
    Tensor sdQ = make_tensor(sP.data(), typename Kernel_traits::SmemLayoutdQ{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
    using GmemTiledCopydO = std::conditional_t<
        Is_first,
        typename Kernel_traits::GmemTiledCopydO,
        typename Kernel_traits::GmemTiledCopyQKV
    >;
    GmemTiledCopydO gmem_tiled_copy_dO;
    auto gmem_thr_copy_dO = gmem_tiled_copy_dO.get_thread_slice(tidx);
    typename Kernel_traits::GmemTiledCopydQ gmem_tiled_copy_dQ;
    auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);
    using GmemLayoutAtomdQaccum = std::conditional_t<
        !Seq_parallel,
        typename Kernel_traits::GmemTiledCopydQaccum,
        typename Kernel_traits::GmemTiledCopydQaccumAtomicAdd
    >;
    GmemLayoutAtomdQaccum gmem_tiled_copy_dQaccum;
    auto gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tdOgdO = gmem_thr_copy_dO.partition_S(gdO);
    Tensor tdOsdO = gmem_thr_copy_dO.partition_D(sdO);
    Tensor tdOgO = gmem_thr_copy_dO.partition_S(gO);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
    Tensor tdQsdQ = gmem_thr_copy_dQ.partition_S(sdQ);    // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);
    Tensor tdQgdQaccum = gmem_thr_copy_dQaccum.partition_D(gdQaccum);
    // if (cute::thread0()) { print(tdQgdQaccum.layout()); printf("\n"); }
    // __syncthreads();
    // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx < 64) {
    //     printf("tidx = %d, tdQgdQaccum = 0x%p\n", tidx, tdQgdQaccum.data());
    // }

    typename Kernel_traits::TiledMmaSdP tiled_mma_sdp;
    auto thr_mma_sdp = tiled_mma_sdp.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma_sdp.partition_fragment_A(sQ);         // (MMA,MMA_N,MMA_K)
    Tensor tSrK = thr_mma_sdp.partition_fragment_B(sK);         // (MMA,MMA_N,MMA_K)
    Tensor tdPrdO = thr_mma_sdp.partition_fragment_A(sdO);      // (MMA,MMA_N,MMA_K)
    Tensor tdPrV = thr_mma_sdp.partition_fragment_B(sV);        // (MMA,MMA_N,MMA_K)

    typename Kernel_traits::TiledMmadKV tiled_mma_dkv;
    auto thr_mma_dkv = tiled_mma_dkv.get_thread_slice(tidx);
    Tensor tdKrdSt = thr_mma_dkv.partition_fragment_A(sdStNoSwizzle); // (MMA, MMA_N, MMA_N)
    Tensor tdKrQt = thr_mma_dkv.partition_fragment_B(sQtNoSwizzle);   // (MMA, MMA_K, MMA_N)
    Tensor tdVrPt = thr_mma_dkv.partition_fragment_A(sPtNoSwizzle);   // (MMA, MMA_N, MMA_N)
    Tensor tdVrdO = thr_mma_dkv.partition_fragment_B(sdOtransposedNoSwizzle); // (MMA, MMA_K, MMA_N)

    typename Kernel_traits::TiledMmadQ tiled_mma_dq;
    auto thr_mma_dq = tiled_mma_dq.get_thread_slice(tidx);
    Tensor tdQrdS = thr_mma_dq.partition_fragment_A(sdS);                      // (MMA, MMA_N, MMA_N)
    Tensor tdQrKt = thr_mma_dq.partition_fragment_B(sKtNoSwizzle);    // (MMA, MMA_K, MMA_N)

    Tensor acc_dk = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});  // MMA, MMA_N, MMA_K
    Tensor acc_dv = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});  // MMA, MMA_N, MMA_K

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_QdO = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto smem_thr_copy_QdO = smem_tiled_copy_QdO.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_QdO.partition_S(sQ);
    Tensor tdPsdO = smem_thr_copy_QdO.partition_S(sdO);

    // auto smem_thr_copy_KV = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp).get_thread_slice(tidx);
    auto smem_tiled_copy_KV = make_tiled_copy_B_warpcontiguousN<MMA_N_SdP>(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto smem_thr_copy_KV = smem_tiled_copy_KV.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_KV.partition_S(sK);
    // if (cute::thread(0, 0) && n_block == 0) { printf("sK layout: "); print(sK.layout()); printf("\n"); }
    // if (cute::thread(0, 0) && n_block == 0) { print(tSsK.layout()); printf("\n"); }
    Tensor tdPsV = smem_thr_copy_KV.partition_S(sV);

    // Partition sP and sdS to match the accumulator partitioning
    // This has to be tiled_mma_sdp, not tiled_mma_dkv
    // auto smem_thr_copy_PdS = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomPdS{}, tiled_mma_sdp).get_thread_slice(tidx);
    auto smem_tiled_copy_PdS = make_tiled_copy_C_warpcontiguousN<MMA_N_SdP>(typename Kernel_traits::SmemCopyAtomPdS{}, tiled_mma_sdp);
    auto smem_thr_copy_PdS = smem_tiled_copy_PdS.get_thread_slice(tidx);
    Tensor tPsP = smem_thr_copy_PdS.partition_D(sP);      // ((Atom,AtomNum),PIPE_M,PIPE_N)
    // if (cute::thread(0, 0) && n_block == 0) { printf("sP layout: "); print(sP.layout()); printf("\n"); }
    // if (cute::thread(0, 0) && n_block == 0) { print(tPsP.layout()); printf("\n"); }
    // if (n_block == 0 && blockIdx.x == 0 && blockIdx.y == 0 && tidx < 64) {
    //     printf("tidx=%d, tPsP = 0x%p\n", tidx, tPsP.data());
    // }
    Tensor tdSsdS = smem_thr_copy_PdS.partition_D(sdS);   // ((Atom,AtomNum),PIPE_M,PIPE_N)

    auto smem_tiled_copy_PdSt = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_dkv);
    auto smem_thr_copy_PdSt = smem_tiled_copy_PdSt.get_thread_slice(tidx);
    Tensor tdVsPt = smem_thr_copy_PdSt.partition_S(sPt);
    Tensor tdKsdSt = smem_thr_copy_PdSt.partition_S(sdSt);

    auto smem_tiled_copy_QdOt = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_dkv);
    auto smem_thr_copy_QdOt = smem_tiled_copy_QdOt.get_thread_slice(tidx);
    Tensor tdVsdOt = smem_thr_copy_QdOt.partition_S(sdOt);
    Tensor tdKsQt = smem_thr_copy_QdOt.partition_S(sQt);

    auto smem_tiled_copy_dS = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_dq);
    auto smem_thr_copy_dS = smem_tiled_copy_dS.get_thread_slice(tidx);
    Tensor tdQsdS = smem_thr_copy_dS.partition_S(sdS);

    auto smem_tiled_copy_Kt = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_dq);
    auto smem_thr_copy_Kt = smem_tiled_copy_Kt.get_thread_slice(tidx);
    Tensor tdQsKt = smem_thr_copy_Kt.partition_S(sKt);

    auto smem_tiled_copy_dQ = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdQ{}, tiled_mma_dq);
    auto smem_thr_copy_dQ = smem_tiled_copy_dQ.get_thread_slice(tidx);
    Tensor taccdQsdQ = smem_thr_copy_dQ.partition_D(sdQ);  // ((Atom,AtomNum),PIPE_M,PIPE_N)

    //
    // PREDICATES
    //

    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tQcQ = gmem_thr_copy_QKV.partition_D(cQ);
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_D(cKV);

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    // Prologue

    // We'll advance gdQ and gdQaccum before the 1st read/write.
    tdQgdQ.data() = tdQgdQ.data() + kBlockM * params.dq_row_stride;
    tdQgdQaccum.data() = tdQgdQaccum.data() + kBlockM * params.h * params.d_rounded;

    int m_block = m_first;
    int m_block_min = (!Is_causal && !Is_local)
        ? 0
        : std::max(0, (n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right) / kBlockM);
    // next/ v12b, runs: the lower end of the current run, and (slot of the current run << 1) | the sQ buffer
    // the current tile's Q is in. Upstream derives the buffer from m_block's parity, which alternates because
    // every m block is visited; across a run gap it need not, so with runs it toggles per visited tile.
    [[maybe_unused]] int run_lo = Use_runs ? run0.y : m_block_min;
    [[maybe_unused]] int run_state = m_first & 1;
    // If not local, we're guaranteed that m_block_min <= m_block:
    // We checked earlier that n_block * kBlockN < actual_seqlen_k, so in the causal case,
    // n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k < actual_seqlen_q.
    // So m_block_min <= (actual_seqlen_q - 1) / kBlockM.
    // Recall that m_block_max = cute::ceil_div(binfo.actual_seqlen_q, kBlockM) = (actual_seqlen_q + kBlockM - 1) / kBlockM.
    // So m_block_m - 1 = (actual_seqlen_q - 1) / kBlockM.
    // We conclude that m_block_min <= m_block, so we will always have at least 1 iteration of the for loop.
    // However, if local, then this possible to have some blocks of K & V not attending to any query.
    // We might need to exit early and write 0 to dK and dV for those blocks.
    // Otherwise we get wrong result for the case where we don't enter the for loop.
    // And we might read OOB elements from gQ and gdO.
    // This also covers the case where actual_seqlen_q == 0
    if ((Is_local || !Is_even_MN || Use_runs) && m_block < m_block_min) {
        const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
          + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
        const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
          + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
        Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                                 Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                 make_stride(params.dk_row_stride, _1{}));
        Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                                 Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                 make_stride(params.dv_row_stride, _1{}));
        typename Kernel_traits::GmemTiledCopydKV gmem_tiled_copy_dKV;
        auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);
        Tensor tdKgdK = gmem_thr_copy_dKV.partition_D(gdK);
        Tensor tdVgdV = gmem_thr_copy_dKV.partition_D(gdV);
        Tensor tdKrdK = make_tensor<Element>(shape(tdKgdK));
        Tensor tdVrdV = make_tensor<Element>(shape(tdVgdV));
        clear(tdKrdK);
        clear(tdVrdV);
        Tensor cdKV = make_identity_tensor(make_shape(size<0>(gdK), size<1>(gdK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
        Tensor tdKVpdKV = make_tensor<bool>(make_shape(size<2>(tdKgdK)));
        #pragma unroll
        for (int k = 0; k < size(tdKVpdKV); ++k) { tdKVpdKV(k) = get<1>(tdKVcdKV(0, 0, k)) < params.d; }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdKrdK, tdKgdK, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
        );
        FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdVrdV, tdVgdV, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
        );
        return;
    }

    if (Double_buffer && m_block % 2 == 1) {  // Double buffer for sQ
        tQsQ.data() = tQsQ.data() + size(sQ);
        tSsQ.data() = tSsQ.data() + size(sQ);
        tdKsQt.data() = tdKsQt.data() + size(sQ);
    }

    if ((!Is_first && !Seq_parallel) || params.deterministic) { __syncthreads(); }

    if (Kernel_traits::Is_V_in_regs) {
        // Clear the smem tiles to account for predicated off loads
        FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
            gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
        );
        FLASH_NAMESPACE::cp_async_fence();
    }

    Tensor tdOrdO = make_fragment_like(tdOgdO);
    Tensor tdOrO = make_fragment_like(tdOgO);
    if (!Is_first) {
        // Clear the smem tiles to account for predicated off loads
        FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
            gmem_tiled_copy_dO, tdOgdO, tdOsdO, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM
        );
    } else {
        FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
            gmem_tiled_copy_dO, tdOgdO, tdOrdO, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM
        );
        FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
            gmem_tiled_copy_dO, tdOgO, tdOrO, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM
        );
    }
    FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM
    );
    // next/ v12b, Dpsum_smem: a tile's dP_sum rows go to an smem buffer (visited-tile parity) by cp.async with
    // the Q prefetch, and the loop reads them there after the barrier instead of from global memory (the
    // dsoftmax_sum buffer is seqlen_q_rounded long, so a whole tile's 16-byte chunks are in bounds).
    [[maybe_unused]] ElementAccum *sdPsum = reinterpret_cast<ElementAccum *>(smem_ + Kernel_traits::kSmemSize1colblock);
    constexpr int kDpsumChunks = kBlockM * int(sizeof(ElementAccum)) / 16;
    static_assert(kDpsumChunks <= Kernel_traits::kNThreads);
    if constexpr (Dpsum_smem) {
        if (tidx < kDpsumChunks) { cqs_cp_async_16(sdPsum + (m_block & 1) * kBlockM + tidx * 4, &gdPsum(tidx * 4)); }
    }

    Tensor caccS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});    // (BLK_M,BLK_N) -> (blk_m,blk_n)
    Tensor taccScS = thr_mma_sdp.partition_C(caccS);                           // (MMA,MMA_N,MMA_N)
    static_assert(decltype(size<0>(taccScS))::value == 4);
    // Convert to ((2, 2), MMA_N, MMA_N) then take only the row indices.
    Tensor taccScS_row = logical_divide(taccScS, Shape<_2>{})(make_coord(0, _), _, 0);
    Tensor lse = make_tensor<ElementAccum>(Shape<Int<decltype(size(taccScS_row))::value>>{});
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
        const int row = get<0>(taccScS_row(mi));
        lse(mi) = Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM ? gLSE(row) : INFINITY;
    }
    // We want LSE = inf if the row is OOB. In that case Q would be zero, K would be zero,
    // and scores would be zero. With LSE = 0, probs will be all 1's, and when we multiply
    // with V (which would be zero), we're fine. However, with ALiBi, we might modify these
    // scores, and probs can become NaN. Instead if we set LSE = inf for OOB rows, probs are always 0.

    // Tensor tKrK = make_fragment_like(tKsK);
    // // cute::copy(gmem_tiled_copy_QKV, tKgK(_, _, _, 0), tKrK);
    // cute::copy(gmem_tiled_copy_QKV, tKgK, tKrK);
    // // if (cute::thread(1, 0)) { print(tKrK); }

    FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
    if (!Kernel_traits::Is_V_in_regs) {
        FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
            gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
        );
    }
    FLASH_NAMESPACE::cp_async_fence();

    // if (cute::thread0()) { print(tdOgdO.layout()); printf("\n"); print(tdOrdO); print(tdOrO); }
    if (Is_first) {
        cute::copy(tdOrdO, tdOsdO);
        dot_do_o<Kernel_traits::kGmemThreadsPerRow>(tdOrdO, tdOrO, gdPsum,
                                                    Kernel_traits::kNThreads / (Kernel_traits::kGmemThreadsPerRow), params.p_dropout);
    }

    if (Kernel_traits::Is_V_in_regs) {
        cute::cp_async_wait<1>();
        __syncthreads();
        Tensor tdPrV_copy_view = smem_thr_copy_KV.retile_D(tdPrV);
        CUTE_STATIC_ASSERT_V(size<1>(tdPsV) == size<1>(tdPrV_copy_view));            // M
        cute::copy(smem_tiled_copy_KV, tdPsV, tdPrV_copy_view);
    }

    FLASH_NAMESPACE::Dropout dropout(params.rng_state[0], params.rng_state[1], params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    clear(acc_dv);
    clear(acc_dk);

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr
        ? 0.0f
        : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    // A/B switch for attributing the backward's cost. Building with
    // -DCQSA_BWD_NO_CQS feeds the Mask compile-time constants, so nvcc
    // propagates them and eliminates every CQS member and code path. The
    // result is the stock FlashAttention backward, which is the baseline
    // needed to tell "our CQS additions cost registers" apart from "this
    // kernel is just slow here". Correctness is not preserved under this
    // macro -- it is a measurement build only.
    // next/ v11x: the macro now also folds the dead-tile verdict and its
    // guards, restricts the mask to the tiles upstream masks (diagonal and
    // ragged), and folds the legacy Num/Den branch, so the loop body is
    // upstream's straight line with only our contract left: global lse in,
    // caller-supplied dsoftmax_sum, fp32 dq_accum.
    FLASH_NAMESPACE::Mask<Is_causal, Is_local, Has_alibi> mask(
        binfo.actual_seqlen_k,
        binfo.actual_seqlen_q,
        params.window_size_left,
        params.window_size_right,
        alibi_slope,
        /*cqs_enabled=*/false, /*num_chunks=*/0, /*owner_chunk=*/0,
        /*chunk_ends=*/nullptr, /*group_bits=*/nullptr,
        /*blk_or=*/nullptr, /*blk_and=*/nullptr,
        /*blk_size=*/0, /*num_blocks=*/0);

    // next/ v12b: with runs the next m block is decided mid-iteration (m_next_carry); without, --m_block.
    [[maybe_unused]] int m_next_carry = -1;
    for (; m_block >= m_block_min; m_block = Use_runs ? m_next_carry : m_block - 1) {
        Tensor acc_s = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_N, MMA_N)
        clear(acc_s);

        // Fully-masked tile skip. Every (row, col) pair here shares a group bit,
        // so this tile's contribution to dQ/dK/dV is exactly zero and its five
        // GEMMs are wasted work. Measured skippable fraction of the tiles causal
        // masking already keeps: 22% at itr=1, 38% at itr=2.
        //
        // The verdict depends only on (m_block, n_block) and the block
        // summaries, never on tidx, so it is uniform across the CTA -- every
        // thread takes the same branch and the __syncthreads() below stay
        // collectively executed.
        //
        // Correctness does not rest on getting the guards exactly right. acc_s
        // and acc_dp are both cleared above, so skipping the first GEMM and the
        // exp2 leaves scores == 0, hence dS == 0, hence zero contribution --
        // the tile is algebraically inert whether or not the later GEMMs run.
        // Skipping them is then purely an optimisation. What must NOT be
        // skipped is the loop's bookkeeping: the dq_accum round trip, the sQ/sdO
        // double-buffer swaps, the cp_async prefetches and the pointer
        // advances all carry state into the next iteration, which is why this
        // is a set of guards rather than a `continue`.
        constexpr bool cqs_skip_tile = false;
        cute::cp_async_wait<0>();
        __syncthreads();

        Tensor dP_sum = make_fragment_like(lse);
        if constexpr (Dpsum_smem) {
            const ElementAccum *sdPsum_cur = sdPsum + ((Use_runs ? run_state : m_block) & 1) * kBlockM;
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) { dP_sum(mi) = sdPsum_cur[get<0>(taccScS_row(mi))]; }
        } else {
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) { dP_sum(mi) = gdPsum(get<0>(taccScS_row(mi))); }
        }

        // if (cute::thread0()) { print(sK); }
        // Tensor tSrK_copy_view = smem_thr_copy_KV.retile_D(tSrK);
        // #pragma unroll
        // for (int k = 0; k < size<2>(tSrK_copy_view); ++k) {
        //     cute::copy(smem_tiled_copy_KV, tSsK(_, _, k), tSrK_copy_view(_, _, k));
        // }
        // if (cute::thread0()) { print(tSrK); }
        if (!cqs_skip_tile) {
            FLASH_NAMESPACE::gemm(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma_sdp,
                    smem_tiled_copy_QdO, smem_tiled_copy_KV, smem_thr_copy_QdO, smem_thr_copy_KV);
        }

        if constexpr (Is_softcap) {
            FLASH_NAMESPACE::apply_softcap(acc_s, params.softcap);
        }

        // Reshape acc_s from (MMA=4, MMA_N, MMA_N) to (row=(2, MMA_N), col=(2, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_s.layout()));
        // if (cute::thread(32, 0)) { print(scores); }

        // Softcapping - calculating dTanh and scaling dS later with it
        [[maybe_unused]] Tensor dtanh = make_tensor_like(scores);
        if constexpr (Is_softcap) {
            FLASH_NAMESPACE::calculate_dtanh(scores, dtanh, params.softcap);
        }

        // The last three arguments are the CQS tile window, and omitting them is
        // not free: without a tile to test, apply_mask cannot take its O(1)
        // early-out and every tile falls into the generic per-element masking
        // loop, which reads cqs_group_bits from global memory twice per element
        // in the kernel's hottest loop. Measured at 9.1x slower than the stock
        // kernel with identical traits and an all-zero mask -- the same effect
        // the forward hit and documented in mask.h (~4.3x there).
        //
        // col_idx_offset_ already carries this warp's column offset, so the
        // matching extent is the warp's own column span, not kBlockN. That is
        // tighter than the tile-wide test the forward uses. cqs_range_or rounds
        // the range out to block-summary boundaries, so unaligned windows stay
        // conservative (it can only ever fail to skip, never skip wrongly).
        // O(1) per-tile verdict, read from params rather than from Mask members.
        // A clear tile skips the per-element CQS check; more importantly the
        // summary pointers stay in constant memory instead of occupying
        // registers across the kernel.
        const int cqs_col_base = n_block * kBlockN + (tidx / 32 / AtomLayoutMS) * MMA_N_SdP * 16;
        const int cqs_tile_state = 0;
        // Skipped for an inert tile too. apply_mask writes -INFINITY into the
        // masked entries of acc_s; leaving it to run would take scores off zero
        // and feed -inf into the dS product. Nothing reads that smem for a
        // skipped tile, but keeping acc_s exactly zero is what makes the
        // "algebraically inert" argument above hold unconditionally.
        // Upstream masks only the ragged last tile and, when causal, the
        // diagonal tiles; Mask<> with CQS off computes exactly what upstream's
        // apply_mask / apply_mask_causal compute there (window_size_right is
        // 0 for causal calls), and every other tile skips the element loop.
        const bool tile_needs_mask = Is_local
            || (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k)
            || (Is_causal && m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k);
        if (tile_needs_mask) {
            mask.template apply_mask<Is_causal, Is_even_MN>(
                acc_s,
                cqs_col_base,
                m_block * kBlockM + get<0>(taccScS_row(0)),
                AtomLayoutMS * 16,
                m_block * kBlockM,
                kBlockM,
                MMA_N_SdP * 16,
                cqs_tile_state
            );
        }

        // if (cute::thread(32, 0)) { print(scores); }
        // Compute the exponential value.
        // Skipped for an inert tile: acc_s stays cleared, so scores stays 0 and
        // P == 0 exactly. Running exp2 here would instead give exp(-lse) != 0.
        if (!cqs_skip_tile) {
            FLASH_NAMESPACE::scale_apply_exp2</*scale_max=*/false>(scores, lse, params.scale_softmax_log2);
        }
        if constexpr (Is_dropout) {
            int warp_id = tidx / 32;
            int block_row_idx = m_block * (kBlockM / 16) + warp_id % AtomLayoutMS;
            // Need col to be multiples of 32, since we're doing dropout with block of 16 x 32
            static_assert(MMA_N_SdP % 2 == 0);
            int block_col_idx = n_block * (kBlockN / 32) + (warp_id / AtomLayoutMS) * (MMA_N_SdP / 2);
            dropout.template apply_dropout</*encode_dropout_in_sign_bit=*/true>(
                acc_s, block_row_idx, block_col_idx, AtomLayoutMS
            );
        }
        // Convert scores from fp32 to fp16/bf16
        Tensor rP = !Is_dropout
            ? FLASH_NAMESPACE::convert_type<Element>(acc_s)
            : FLASH_NAMESPACE::convert_type_relu<Element>(acc_s);
        // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_N, MMA_N / 2)
        // if using m16n8k16 or (4, MMA_N, MMA_N) if using m16n8k8.
        Tensor tPrP = make_tensor(rP.data(), FLASH_NAMESPACE::convert_layout_acc_Aregs<Kernel_traits::TiledMmaSdP>(rP.layout()));
        Tensor tPaP = smem_thr_copy_PdS.retile_S(tPrP);     // ((Atom,AtomNum), MMA_N, MMA_N)
        cute::copy(smem_tiled_copy_PdS, tPaP, tPsP);
        // if (cute::thread0()) { print(tPaP); }
        // __syncthreads();
        // if (cute::thread0()) { print(sP); }

        Tensor acc_dp = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_N, MMA_N)
        CUTE_STATIC_ASSERT_V(size<0>(acc_dp) == size<0>(acc_s));                     // MMA
        CUTE_STATIC_ASSERT_V(size<1>(acc_dp) == size<1>(acc_s));                     // MMA
        CUTE_STATIC_ASSERT_V(size<2>(acc_dp) == size<2>(acc_s));                     // MMA

        clear(acc_dp);
        // Tensor acc_dp_reshaped = make_tensor(acc_dp.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_dp.layout()));
        // #pragma unroll
        // for (int mi = 0; mi < size<0>(acc_dp_reshaped); ++mi) {
        //     #pragma unroll
        //     for (int ni = 0; ni < size<1>(acc_dp_reshaped); ++ni) {
        //         acc_dp_reshaped(mi, ni) = -dP_sum(mi);
        //     }
        // }

        // if (cute::thread0()) { print(dP_sum); }

        if (!cqs_skip_tile) {
            FLASH_NAMESPACE::gemm</*A_in_regs=*/false, /*B_in_regs=*/Kernel_traits::Is_V_in_regs>(
                acc_dp, tdPrdO, tdPrV, tdPsdO, tdPsV, tiled_mma_sdp,
                smem_tiled_copy_QdO, smem_tiled_copy_KV, smem_thr_copy_QdO, smem_thr_copy_KV
            );
        }

        // Reshape acc_dp from (MMA=4, MMA_N, MMA_N) to (row=(2, MMA_N), col=(2, MMA_N))
        Tensor dS = make_tensor(acc_dp.data(), scores.layout());
        constexpr bool cqsa_numden_mode = false;
        #pragma unroll
        for (int mi = 0; mi < size<0>(dS); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(dS); ++ni) {
                float scaled_ds;
                if (cqsa_numden_mode) {
                    // Stream-CQSA Num/Den backward:
                    // do_ptr is pre-scaled as do = dNum * Den on host side.
                    // Here dS (from acc_dp) equals Den * (dNum @ V^T), while dP_sum stores dDen.
                    // Then:
                    // dR = exp(R) * (dNum @ V^T + dDen_row)
                    //    = p * [ Den * (dNum @ V^T) + Den * dDen_row ]
                    //    = p * [ dS + dDen_row * Den ].
                    const float lse_row = lse(mi);
                    const float den_row = isfinite(lse_row) ? expf(lse_row) : 0.0f;
                    float p = scores(mi, ni);
                    if constexpr (Is_dropout) {
                        // Dropout-encoded sign bit path is unsupported in cqsa_numden_mode.
                        p = p >= 0.f ? p : 0.f;
                    }
                    scaled_ds = p * (dS(mi, ni) + dP_sum(mi) * den_row);
                } else {
                    // Standard FA softmax backward:
                    // dR = p * (dP - row_sum(dP * p)).
                    const float p = scores(mi, ni);
                    scaled_ds = p * (!Is_dropout || p >= 0 ? dS(mi, ni) - dP_sum(mi) : dP_sum(mi));
                }
                if constexpr (Is_softcap) { scaled_ds *= dtanh(mi, ni); }
                dS(mi, ni) = scaled_ds;
            }
        }
        // if (cute::thread0()) { print(dS); }

        Tensor acc_dq = partition_fragment_C(tiled_mma_dq, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_N, MMA_K
        tdQgdQaccum.data() = tdQgdQaccum.data() + (-int(kBlockM * params.h * params.d_rounded));
        if (Is_first || Seq_parallel) {
            clear(acc_dq);
        } else {
            // Reshape acc_dq from (4, 1, 2) to (4, 2, 1) to write to gdQaccum
            Tensor acc_dq_reshaped = make_tensor(acc_dq.data(),
                                                 make_layout(get<0>(acc_dq.layout()),
                                                             get<2>(acc_dq.layout()),
                                                             get<1>(acc_dq.layout())));
            cute::copy(gmem_tiled_copy_dQaccum, tdQgdQaccum, acc_dq_reshaped);
        }

        // next/ v12b: the next m block to visit, decided where the Q prefetch first needs it. Runs: the next
        // tile of the run, or at a run boundary (1-3 per column on aligned tasks) the next run's hi.
        [[maybe_unused]] int m_step = 1;
        if constexpr (Use_runs) {
            int m_next;
            if (m_block > run_lo) {
                m_next = m_block - 1;
            } else {
                run_state += 2;
                const int2 run = col_runs[run_state >> 1];
                m_next = run.x;
                run_lo = run.y;
            }
            m_next_carry = m_next;
            m_step = m_block - m_next;
        }
        const bool has_next = Use_runs ? m_next_carry >= 0 : m_block > m_block_min;

        if (Double_buffer && has_next) {
            // Double buffer for sQ
            const int sQ_offset = (Use_runs ? (run_state & 1) : m_block % 2) == 0 ? size(sQ) : -size(sQ);
            tQsQ.data() = tQsQ.data() + sQ_offset;
            tSsQ.data() = tSsQ.data() + sQ_offset;
            // Advance gQ
            tQgQ.data() = tQgQ.data() + (-int(kBlockM * params.q_row_stride));
            if constexpr (Use_runs) { tQgQ.data() = tQgQ.data() + (-index_t(m_step - 1) * kBlockM * params.q_row_stride); }
            FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ);
            if constexpr (Dpsum_smem) {   // and the next tile's dP_sum into the other smem buffer
                gdPsum.data() = gdPsum.data() + (-int(m_step * kBlockM));
                if (tidx < kDpsumChunks) {
                    cqs_cp_async_16(sdPsum + (((Use_runs ? run_state : m_block) & 1) ^ 1) * kBlockM + tidx * 4, &gdPsum(tidx * 4));
                }
            }
            FLASH_NAMESPACE::cp_async_fence();
        }

        Tensor dS_reshaped = make_tensor(dS.data(), acc_dp.layout());
        // Convert dS from fp32 to fp16
        Tensor tdSrdS = FLASH_NAMESPACE::convert_type<Element>(dS_reshaped);
        // if (cute::thread0()) { print(tPrP); }
        Tensor tdSadS = smem_thr_copy_PdS.retile_S(tdSrdS);                                          // ((Atom,AtomNum), MMA_N, MMA_N)
        cute::copy(smem_tiled_copy_PdS, tdSadS, tdSsdS);
        __syncthreads();

        // Layout p_l = tPrP.layout();
        // Tensor tdVrPt = make_tensor(tPrP.data(), make_layout(get<0>(p_l), get<2>(p_l), get<1>(p_l)));
        // FLASH_NAMESPACE::gemm_rs(acc_dv, tdVrPt, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_thr_copy_QdOt);
        // Tensor tdKrdSt = make_tensor(tdSrdS.data(), tdVrPt.layout());
        // FLASH_NAMESPACE::gemm_rs(acc_dk, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_thr_copy_QdOt);
        if (!cqs_skip_tile) {
            FLASH_NAMESPACE::gemm(acc_dv, tdVrPt, tdVrdO, tdVsPt, tdVsdOt, tiled_mma_dkv,
                    smem_tiled_copy_PdSt, smem_tiled_copy_QdOt, smem_thr_copy_PdSt, smem_thr_copy_QdOt);
        }
        // if (cute::thread0() && n_block == 0 && m_block == 0) { print(tdVrPt); }
        // if (cute::thread0()) { print(acc_dv); }

        __syncthreads(); // Need syncthreads since we're writing to the same sdO location

        if (has_next) {
            // Advance gdO
            tdOgdO.data() = tdOgdO.data() + (-int(kBlockM * params.do_row_stride));
            if constexpr (Use_runs) { tdOgdO.data() = tdOgdO.data() + (-index_t(m_step - 1) * kBlockM * params.do_row_stride); }
            if (Is_first) {
                tdOgO.data() = tdOgO.data() + (-int(kBlockM * params.o_row_stride));
                FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_dO, tdOgdO, tdOrdO, tQcQ, tQpQ);
                FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_dO, tdOgO, tdOrO, tQcQ, tQpQ);
            } else {
                FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_dO, tdOgdO, tdOsdO, tQcQ, tQpQ);
                FLASH_NAMESPACE::cp_async_fence();
            }
        }

        if (!cqs_skip_tile) {
            FLASH_NAMESPACE::gemm(acc_dq, tdQrdS, tdQrKt, tdQsdS, tdQsKt, tiled_mma_dq,
                    smem_tiled_copy_dS, smem_tiled_copy_Kt, smem_thr_copy_dS, smem_thr_copy_Kt);
        }
        // if (cute::thread0()) { print(acc_dq); }

        if (has_next) {
            gLSE.data() = gLSE.data() + (-int(m_step * kBlockM));
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) { lse(mi) = gLSE(get<0>(taccScS_row(mi))); }
            if constexpr (!Dpsum_smem) { gdPsum.data() = gdPsum.data() + (-int(m_step * kBlockM)); }
        }

        if (!Is_last) {
            // Reshape acc_dq from (4, 1, 2) to (4, 2, 1) to write to gdQaccum
            Tensor acc_dq_reshaped = make_tensor(acc_dq.data(),
                                                 make_layout(get<0>(acc_dq.layout()),
                                                             get<2>(acc_dq.layout()),
                                                             get<1>(acc_dq.layout())));
            if (!Seq_parallel) {
                cute::copy(gmem_tiled_copy_dQaccum, acc_dq_reshaped, tdQgdQaccum);
            } else {
                // if (cute::thread0()) { print(acc_dq.layout()); printf("\n"); print(acc_dq_reshaped.layout()); printf("\n"); print(tdQgdQaccum.layout()); printf("\n"); }
                CUTE_STATIC_ASSERT_V(size(acc_dq) == size(tdQgdQaccum));
                #pragma unroll
                for (int i = 0; i < size(acc_dq); ++i) { atomicAdd(&tdQgdQaccum(i), acc_dq(i)); }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < size(acc_dq); ++i) { acc_dq(i) *= params.scale_softmax_rp_dropout; }
            // Convert acc_dq from fp32 to fp16
            Tensor rdQ = FLASH_NAMESPACE::convert_type<Element>(acc_dq);
            Tensor taccdQrdQ = smem_thr_copy_dQ.retile_S(rdQ);  // ((Atom,AtomNum), MMA_N, MMA_N)
            cute::copy(smem_tiled_copy_dQ, taccdQrdQ, taccdQsdQ);
        }

        if (!cqs_skip_tile) {
            FLASH_NAMESPACE::gemm(acc_dk, tdKrdSt, tdKrQt, tdKsdSt, tdKsQt, tiled_mma_dkv,
                    smem_tiled_copy_PdSt, smem_tiled_copy_QdOt, smem_thr_copy_PdSt, smem_thr_copy_QdOt);
        }
        // if (cute::thread0()) { print(acc_dk); }
        if (Double_buffer) {  // Double buffer for sQ
            tdKsQt.data() = tdKsQt.data() + ((Use_runs ? (run_state & 1) : m_block % 2) == 0 ? size(sQ) : -size(sQ));
        }
        if (!Double_buffer && has_next) {
            __syncthreads();
            // Advance gQ
            tQgQ.data() = tQgQ.data() + (-int(kBlockM * params.q_row_stride));
            if constexpr (Use_runs) { tQgQ.data() = tQgQ.data() + (-index_t(m_step - 1) * kBlockM * params.q_row_stride); }
            FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ);
            if constexpr (Dpsum_smem) {   // and the next tile's dP_sum into the other smem buffer
                gdPsum.data() = gdPsum.data() + (-int(m_step * kBlockM));
                if (tidx < kDpsumChunks) {
                    cqs_cp_async_16(sdPsum + (((Use_runs ? run_state : m_block) & 1) ^ 1) * kBlockM + tidx * 4, &gdPsum(tidx * 4));
                }
            }
            FLASH_NAMESPACE::cp_async_fence();
        }

        if (Is_first && has_next) {
            cute::copy(tdOrdO, tdOsdO);
            dot_do_o<Kernel_traits::kGmemThreadsPerRow>(tdOrdO, tdOrO, gdPsum,
                                                        Kernel_traits::kNThreads / (Kernel_traits::kGmemThreadsPerRow), params.p_dropout);
        }

        if (Is_last) {
            __syncthreads();
            Tensor tdQrdQ = make_tensor<Element>(shape(tdQgdQ));
            cute::copy(gmem_tiled_copy_dQ, tdQsdQ, tdQrdQ);
            tdQgdQ.data() = tdQgdQ.data() + (-int(kBlockM * params.dq_row_stride));
            Tensor cdQ = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
            Tensor tdQcdQ = gmem_thr_copy_dQ.partition_D(cdQ);
            #pragma unroll
            for (int m = 0; m < size<1>(tdQgdQ); ++m) {
                if (Is_even_MN || get<0>(tdQcdQ(0, m, 0)) < binfo.actual_seqlen_q - m_block * kBlockM) {
                    cute::copy(gmem_tiled_copy_dQ, tdQrdQ(_, m, _), tdQgdQ(_, m, _));
                }
            }
        }

        if constexpr (Use_runs) {
            // dQaccum: the top of the next iteration steps one block; a run gap needs the rest.
            tdQgdQaccum.data() = tdQgdQaccum.data() + (-index_t(m_step - 1) * kBlockM * params.h * params.d_rounded);
            run_state ^= 1;
        }

    }

    // Epilogue

    if (Is_dropout) {
        #pragma unroll
        for (int i = 0; i < size(acc_dv); ++i) { acc_dv(i) *= params.rp_dropout; }
    }
    #pragma unroll
    for (int i = 0; i < size(acc_dk); ++i) { acc_dk(i) *= params.scale_softmax_rp_dropout; }

    // Convert acc_dv from fp32 to fp16
    Tensor rdK = FLASH_NAMESPACE::convert_type<Element>(acc_dk);
    Tensor rdV = FLASH_NAMESPACE::convert_type<Element>(acc_dv);

    Tensor sdK = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutdKV{});  // (SMEM_N, SMEM_K)
    Tensor sdV = make_tensor(sdK.data() + size(sdK), typename Kernel_traits::SmemLayoutdKV{}); // (SMEM_N, SMEM_K)

    // Partition sdV and sdK to match the accumulator partitioning
    auto smem_tiled_copy_dKV = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdKV{}, tiled_mma_dkv);
    auto smem_thr_copy_dKV = smem_tiled_copy_dKV.get_thread_slice(tidx);
    Tensor taccdKrdK = smem_thr_copy_dKV.retile_S(rdK);       // ((Atom,AtomNum), MMA_N, MMA_N)
    Tensor taccdKsdK = smem_thr_copy_dKV.partition_D(sdK);   // ((Atom,AtomNum),PIPE_M,PIPE_N)
    Tensor taccdVrdV = smem_thr_copy_dKV.retile_S(rdV);       // ((Atom,AtomNum), MMA_N, MMA_N)
    Tensor taccdVsdV = smem_thr_copy_dKV.partition_D(sdV);    // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // We need syncthreads here since we're writing to the same location as sK and sV.
    // Without syncthreads, some thread might modify the location of sK while another thread
    // is reading it for dQ gemm, leading to a race condition.
    // If Is_last, there's already a __syncthreads() at the end of the loop.
    if (!Is_last) { __syncthreads(); }

    cute::copy(smem_tiled_copy_dKV, taccdKrdK, taccdKsdK);
    cute::copy(smem_tiled_copy_dKV, taccdVrdV, taccdVsdV);

    const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
       + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
    const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
       + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
    Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                             Shape<Int<kBlockN>, Int<kHeadDim>>{},
                             make_stride(params.dk_row_stride, _1{}));
    Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                             Shape<Int<kBlockN>, Int<kHeadDim>>{},
                             make_stride(params.dv_row_stride, _1{}));

    typename Kernel_traits::GmemTiledCopydKV gmem_tiled_copy_dKV;
    auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);
    Tensor tdKsdK = gmem_thr_copy_dKV.partition_S(sdK);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdKgdK = gmem_thr_copy_dKV.partition_D(gdK);
    Tensor tdVsdV = gmem_thr_copy_dKV.partition_S(sdV);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdVgdV = gmem_thr_copy_dKV.partition_D(gdV);

    __syncthreads();
    Tensor tdKrdK = make_tensor<Element>(shape(tdKgdK));
    cute::copy(gmem_tiled_copy_dKV, tdKsdK, tdKrdK);
    Tensor tdVrdV = make_tensor<Element>(shape(tdVgdV));
    cute::copy(gmem_tiled_copy_dKV, tdVsdV, tdVrdV);
    Tensor cdKV = make_identity_tensor(make_shape(size<0>(sdK), size<1>(sdK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
    Tensor tdKVpdKV = make_tensor<bool>(make_shape(size<2>(tdKgdK)));
    #pragma unroll
    for (int k = 0; k < size(tdKVpdKV); ++k) { tdKVpdKV(k) = get<1>(tdKVcdKV(0, 0, k)) < params.d; }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dKV, tdKrdK, tdKgdK, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
    FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dKV, tdVrdV, tdVgdV, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
    );

}

// With runs, the CTAs return at once unless the census routed the call here
// (route 0); LEGACY, launched right after on the same stream, does the reverse.
template<typename Kernel_traits, bool Is_causal, bool Is_even_MN, bool Use_runs, bool Dpsum_smem>
__global__ void flash_bwd_dq_dk_dv_loop_seqk_parallel_aligned_kernel(KERNEL_PARAM_MODIFIER const Flash_bwd_params params) {
    #if defined(ARCH_SUPPORTS_FLASH)
        if constexpr (Use_runs) {
            if (__ldg(params.cqs_bwd_route + blockIdx.y * params.cqs_wave_route_stride) != 0) { return; }
        }
        // The block index for the batch.
        const int bidb = blockIdx.y;
        // The block index for the head.
        const int bidh = blockIdx.z;
        // If deterministic, each thread block will do atomicAdd to a different dQ_accum buffer.
        for (int n_block = blockIdx.x; n_block < (params.seqlen_k + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN; n_block += gridDim.x) {
            compute_dq_dk_dv_1colblock_aligned<Kernel_traits, /*Is_dropout=*/false, Is_causal, /*Is_local=*/false,
                /*Has_alibi=*/false, Is_even_MN, /*Is_even_K=*/true, /*Is_softcap=*/false, /*Is_first=*/false,
                /*Is_last=*/false, /*Seq_parallel=*/true, Use_runs, Dpsum_smem>(params, bidb, bidh, n_block);
        }
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

// The calls the ALIGNED kernel serves; flash_api.cpp allocates the census only
// for CQS-on calls that pass this (plus group bits and summaries).
template<typename Kernel_traits>
inline bool cqs_bwd_aligned_eligible(const Flash_bwd_params &params) {
    return !params.cqsa_numden_mode
        && params.p_dropout >= 1.f                      // keep probability: no dropout
        && params.alibi_slopes_ptr == nullptr
        && params.softcap == 0.f
        && (params.is_causal || (params.window_size_left < 0 && params.window_size_right < 0))
        && params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr
        && params.d == Kernel_traits::kHeadDim;
}

template<typename Kernel_traits, bool Is_causal>
void run_flash_bwd_aligned_kernel(Flash_bwd_params &params, cudaStream_t stream) {
    const int gridDimx = bwd_grid_dimx<Kernel_traits>(params);
    dim3 grid_n(gridDimx, params.b, params.h);
    static_assert(128 % Kernel_traits::kBlockM == 0 && 128 % Kernel_traits::kBlockN == 0,
                  "cqs_wave_lens_even states lengths are multiples of 128");
    const bool is_even_MN = params.seqlen_q % Kernel_traits::kBlockM == 0 && params.seqlen_k % Kernel_traits::kBlockN == 0
        && (params.cqs_wave_lens == nullptr || params.cqs_wave_lens_even);
    // A/B switch for the dP_sum source, read per call (unset: the per-path defaults above).
    // Only "0" and "1" are accepted (any other value, e.g. "true", used to
    // select the global source silently and mislabel an A/B arm).
    const char *dps_env = std::getenv("CQSA_WAVE_DPSUM_SMEM");
    const bool dps_set = dps_env != nullptr && dps_env[0] != '\0';
    TORCH_CHECK(!dps_set || ((dps_env[0] == '0' || dps_env[0] == '1') && dps_env[1] == '\0'),
                "CQSA_WAVE_DPSUM_SMEM must be 0 or 1 (or unset for the per-path default)");
    const bool dpsum_smem = dps_set
        ? dps_env[0] == '1'
        : (params.cqs_enabled ? kCqsBwdDpsumSmemCqsOn : kCqsBwdDpsumSmemCqsOff);
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        BOOL_SWITCH(params.cqs_enabled, UseRunsConst, [&] {
            BOOL_SWITCH(dpsum_smem, DpsumSmemConst, [&] {
                constexpr int smem_size_dq_dk_dv = Kernel_traits::kSmemSize1colblock + (DpsumSmemConst ? kCqsBwdDpsumSmem<Kernel_traits> : 0);
                auto kernel = &flash_bwd_dq_dk_dv_loop_seqk_parallel_aligned_kernel<Kernel_traits, Is_causal, IsEvenMNConst, UseRunsConst, DpsumSmemConst>;
                if (smem_size_dq_dk_dv >= 48 * 1024)  {
                    C10_CUDA_CHECK(cudaFuncSetAttribute(
                        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_dq_dk_dv));
                }
                kernel<<<grid_n, Kernel_traits::kNThreads, smem_size_dq_dk_dv, stream>>>(params);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
        });
    });
}

// The whole backward of one call (declared in flash_bwd_launch_template.h):
// preprocess -> [census] -> ALIGNED and/or LEGACY -> convert_dQ, all on the
// stream. CQS off and eligible: the ALIGNED loop without a run table. CQS on
// with a census buffer (flash_api.cpp): census, ALIGNED, then LEGACY, each
// kernel's CTAs returning at once unless the route picks it. Anything else:
// LEGACY alone, as v11.
template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_bwd_main(Flash_bwd_params &params, cudaStream_t stream) {
    using T = typename Kernel_traits::Element;
    run_flash_bwd_preprocess<Kernel_traits>(params, stream);
    bool aligned = false;
    if constexpr (!Is_dropout) {
        aligned = cqs_bwd_aligned_eligible<Kernel_traits>(params)
            && (!params.cqs_enabled || (params.cqs_bwd_route != nullptr && params.cqs_group_bits != nullptr
                                        && params.cqs_blk_or != nullptr && params.cqs_blk_and != nullptr));
    }
    if (aligned) {
        if (params.cqs_enabled) {
            run_cqs_bwd_census(params, Kernel_traits::kBlockM, Kernel_traits::kBlockN,
                               params.cqs_wave_route_stride != 0 ? params.b : 1, stream);
        }
        run_flash_bwd_aligned_kernel<Kernel_traits, Is_causal>(params, stream);
    } else {
        params.cqs_bwd_route = nullptr;   // LEGACY alone computes the call
    }
    if (!aligned || params.cqs_enabled) {
        run_mha_bwd_legacy_<T, Kernel_traits::kHeadDim, Is_causal>(params, stream);
    }
    run_flash_bwd_convert_dq<Kernel_traits>(params, stream);
}

}  // namespace FLASH_NAMESPACE
