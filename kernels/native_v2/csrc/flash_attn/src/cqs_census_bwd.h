/******************************************************************************
 * next/native_v2: the backward's tile census (per subproblem and column block)
 * and run table: v12b's census (kernels/v12b) on the uniform wave layout.
 *
 * The transpose of the forward census (cqs_census.h): the backward is
 * seqk-parallel, one CTA per n_block column looping over the m blocks from
 * high to low, so the census classifies, per column block, the m tiles the
 * backward visits and lists the live ones as runs of consecutive m blocks in
 * descending m -- the order v11's backward visits them in. grid.y walks the
 * subproblems; every table sits at bidb * stride and each subproblem has its
 * own route word.
 ******************************************************************************/

#pragma once

#include "namespace_config.h"
#include "flash.h"

namespace FLASH_NAMESPACE {

// Run-table slots per column block: up to kCqsBwdRunSlots - 1 runs and the
// {-1, -1} sentinel. A 128-aligned CQS task has 1 run per column block at
// itr=1 (every row block below the column's own chunk, or all of them for the
// owner chunk's columns) and a few at itr=2; more runs than slots is recorded
// as an overflow and the subproblem goes to LEGACY (still exact).
constexpr int kCqsBwdRunSlots = 16;

// Route bits written by the census (0 = ALIGNED).
constexpr int kCqsBwdRouteMixed = 1;      // a visited tile is neither fully live nor fully masked
constexpr int kCqsBwdRouteOverflow = 2;   // a column block has more live runs than slots

// Classify every (m tile, column block) pair the backward visits in each of
// the `num_sub` subproblems, from the 64-token OR/AND summaries (exact over
// the valid tokens: the ragged last summary block is recomputed from the group
// bits), and write params.cqs_bwd_route[w * route_stride] and
// params.cqs_bwd_runs for tiles of block_m x block_n (the backward's
// Kernel_traits). Zeroes the route words first. Stream-ordered, no host sync.
void run_cqs_bwd_census(const Flash_bwd_params &params, int block_m, int block_n, int num_sub, cudaStream_t stream);

}  // namespace FLASH_NAMESPACE
