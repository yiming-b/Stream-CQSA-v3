/******************************************************************************
 * next/native_v2: tile census for the ALIGNED wave backward (see
 * cqs_census_bwd.h). Structure and verdicts as the forward census
 * (cqs_census.cu), transposed: one warp per COLUMN block, lanes walk its m
 * tiles downward.
 ******************************************************************************/

#include "namespace_config.h"
#include <c10/cuda/CUDAException.h>  // C10_CUDA_CHECK, C10_CUDA_KERNEL_LAUNCH_CHECK

#include "cqs_census_bwd.h"

namespace FLASH_NAMESPACE {

namespace {

constexpr int kSumBlk = 64;          // CQS_BLK_SIZE: tokens per OR/AND summary entry
constexpr int kCensusWarps = 4;      // one warp per column block

// OR and AND of the group bits of tokens [start, end), start a multiple of
// kSumBlk; the ragged last summary block (tail_blk, -1 if none) comes from the
// exact values over its valid tokens (see cqs_census.cu).
__device__ __forceinline__ void range_summary(const int64_t *__restrict__ blk_or, const int64_t *__restrict__ blk_and,
                                              const int start, const int end, const int tail_blk,
                                              const int64_t tail_or, const int64_t tail_and,
                                              int64_t &r_or, int64_t &r_and) {
    r_or = 0;
    r_and = ~int64_t(0);
    for (int b = start / kSumBlk; b * kSumBlk < end; ++b) {
        if (b == tail_blk) {
            r_or |= tail_or;
            r_and &= tail_and;
        } else {
            r_or |= blk_or[b];
            r_and &= blk_and[b];
        }
    }
}

// Grid: (column blocks of S / kCensusWarps, subproblems); one warp per column
// block of block_n columns of subproblem w = blockIdx.y, whose L = lens[w]
// (seqlen_q == seqlen_k == L). The tiles visited are those of the backward: m
// in [m_min, m_max) with m_min the causal limit (n * block_n / block_m; 0
// without causal), so a column's runs are exactly the tiles ALIGNED computes,
// in the order v11 computes them (descending m). Column blocks past L get the
// sentinel alone.
__global__ void __launch_bounds__(kCensusWarps * 32)
cqs_bwd_census_kernel(const Flash_bwd_params params, const int block_m, const int block_n) {
    __shared__ int64_t s_tail[2];
    const int w = blockIdx.y;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int L = params.cqs_wave_lens != nullptr ? params.cqs_wave_lens[w] : params.seqlen_q;
    const int64_t *bits = params.cqs_group_bits + (int64_t)w * params.cqs_wave_tok_stride;
    const int64_t *blk_or = params.cqs_blk_or + (int64_t)w * params.cqs_wave_blk_stride;
    const int64_t *blk_and = params.cqs_blk_and + (int64_t)w * params.cqs_wave_blk_stride;
    int2 *runs = const_cast<int2 *>(params.cqs_bwd_runs) + ((int64_t)w * params.cqs_wave_bwd_run_stride) * kCqsBwdRunSlots;
    int *route = params.cqs_bwd_route + w * params.cqs_wave_route_stride;
    const int tail_blk = (L % kSumBlk) != 0 ? L / kSumBlk : -1;
    if (warp == 0) {
        int64_t o = 0, a = ~int64_t(0);
        if (tail_blk >= 0) {
            for (int t = tail_blk * kSumBlk + lane; t < L; t += 32) { o |= bits[t]; a &= bits[t]; }
        }
        #pragma unroll
        for (int off = 16; off > 0; off /= 2) {
            o |= __shfl_xor_sync(0xffffffffu, o, off);
            a &= __shfl_xor_sync(0xffffffffu, a, off);
        }
        if (lane == 0) { s_tail[0] = o; s_tail[1] = a; }
    }
    __syncthreads();
    const int num_n = (L + block_n - 1) / block_n;
    const int n = blockIdx.x * kCensusWarps + warp;
    int2 *col_runs = runs + n * kCqsBwdRunSlots;
    if (n >= num_n) {
        if (lane == 0 && n * block_n < params.seqlen_k) { col_runs[0] = make_int2(-1, -1); }
        return;
    }
    const int64_t tail_or = s_tail[0], tail_and = s_tail[1];

    int64_t col_or, col_and;
    range_summary(blk_or, blk_and, n * block_n, min((n + 1) * block_n, L), tail_blk, tail_or, tail_and, col_or, col_and);
    const int m_max = (L + block_m - 1) / block_m;
    const int m_min = params.is_causal ? (n * block_n) / block_m : 0;

    // Lane j classifies tile m = base - j; lane 0 turns the ballot into runs.
    // Bit j of `above` is the liveness of tile m + 1 (for j = 0: the lowest
    // tile of the previous group of 32), so a run starts where a live tile has
    // a dead one above it and ends where a dead tile has a live one above it.
    int nr = 0, hi = -1;
    bool overflow = false, mixed = false, prev_live = false;
    for (int base = m_max - 1; base >= m_min; base -= 32) {
        const int m = base - lane;
        bool live = false, mix = false;
        if (m >= m_min) {
            int64_t r_or, r_and;
            range_summary(blk_or, blk_and, m * block_m, min((m + 1) * block_m, L), tail_blk, tail_or, tail_and, r_or, r_and);
            live = (r_or & col_or) == 0;
            mix = !live && (r_and & col_and) == 0;
        }
        const unsigned live_bits = __ballot_sync(0xffffffffu, live);
        mixed |= __any_sync(0xffffffffu, mix);
        if (lane == 0) {
            const int count = base - m_min + 1;             // valid lanes in this group: j < count
            const unsigned valid = count >= 32 ? 0xffffffffu : ((1u << count) - 1u);
            const unsigned above = (live_bits << 1) | unsigned(prev_live);
            const unsigned starts = live_bits & ~above;
            unsigned events = starts | (~live_bits & above & valid);
            while (events) {
                const int j = __ffs(events) - 1;
                events &= events - 1;
                if ((starts >> j) & 1u) {
                    hi = base - j;
                } else if (nr < kCqsBwdRunSlots - 1) {
                    col_runs[nr++] = make_int2(hi, base - j + 1);
                } else {
                    overflow = true;
                }
            }
            prev_live = (live_bits >> min(31, base - m_min)) & 1u;
        }
    }
    if (lane == 0) {
        if (prev_live) {                   // the last run reaches m_min
            if (nr < kCqsBwdRunSlots - 1) { col_runs[nr++] = make_int2(hi, m_min); } else { overflow = true; }
        }
        col_runs[nr] = make_int2(-1, -1);
        const int why = (mixed ? kCqsBwdRouteMixed : 0) | (overflow ? kCqsBwdRouteOverflow : 0);
        if (why) { atomicOr(route, why); }
    }
}

}  // namespace

void run_cqs_bwd_census(const Flash_bwd_params &params, int block_m, int block_n, int num_sub, cudaStream_t stream) {
    const int num_routes = 1 + (num_sub - 1) * params.cqs_wave_route_stride;
    C10_CUDA_CHECK(cudaMemsetAsync(params.cqs_bwd_route, 0, num_routes * sizeof(int), stream));
    const int num_n = (params.seqlen_k + block_n - 1) / block_n;
    dim3 grid((num_n + kCensusWarps - 1) / kCensusWarps, num_sub);
    cqs_bwd_census_kernel<<<grid, kCensusWarps * 32, 0, stream>>>(params, block_m, block_n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace FLASH_NAMESPACE
