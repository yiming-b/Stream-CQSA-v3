// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// next/ v12f: the CQS-off forward and the ALIGNED CQS-on forward; v11's CQS-on
// forward (LEGACY) is in flash_fwd_hdim64_bf16_legacy_sm80.cu.
#include "namespace_config.h"
#include "flash_fwd_aligned.h"

namespace FLASH_NAMESPACE {

template<>
void run_mha_fwd_<cutlass::bfloat16_t, 64, false>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_plain_or_aligned<cutlass::bfloat16_t, 64, false>(params, stream);
}

} // namespace FLASH_NAMESPACE
