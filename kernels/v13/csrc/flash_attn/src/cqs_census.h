/******************************************************************************
 * next/ v12f: the forward's tile census and the ALIGNED instantiation's tile shape.
 ******************************************************************************/

#pragma once

#include "namespace_config.h"
#include "flash.h"

namespace FLASH_NAMESPACE {

// Run-table slots per row block: up to kCqsRunSlots - 1 runs and the {-1, -1}
// sentinel. A 128-aligned CQS task has 1 run per causal row block at itr=1
// (everything left of the row's own chunk) and a few at itr=2; more runs than
// slots is recorded as an overflow and the call goes to LEGACY (still exact).
constexpr int kCqsRunSlots = 16;

// Route bits written by the census (0 = ALIGNED).
constexpr int kCqsRouteMixed = 1;      // a visited tile is neither fully live nor fully masked
constexpr int kCqsRouteOverflow = 2;   // a row block has more live runs than slots

// The ALIGNED forward's tile shape per head dim: the sm80 choice of
// run_mha_fwd_hdim64/128, i.e. the tiles v11 runs on an A100, so ALIGNED and
// LEGACY/v11 visit the same tiles and can be compared bit for bit. The census
// is run with exactly this shape (flash_fwd_aligned.h and flash_api.cpp).
template<int Headdim> struct CqsAlignedTile;
template<> struct CqsAlignedTile<64>  { static constexpr int kBlockM = 128, kBlockN = 128; };
template<> struct CqsAlignedTile<128> { static constexpr int kBlockM = 128, kBlockN = 64; };

// Classify every (row block, column tile) pair the forward visits, from the
// 64-token OR/AND summaries (exact over the valid tokens: the ragged last
// summary block is recomputed from the group bits), and write params.cqs_route
// and params.cqs_runs. Zeroes the route first. Stream-ordered, no host sync.
void run_cqs_fwd_census(const Flash_fwd_params &params, int block_m, int block_n, cudaStream_t stream);

}  // namespace FLASH_NAMESPACE
