// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// native_v2: the CQS-off forward and the ALIGNED CQS-on wave forward; v11's CQS-on
// forward (LEGACY) is in flash_fwd_hdim128_fp16_causal_legacy_sm80.cu.
#include "namespace_config.h"
#include "flash_fwd_aligned.h"

namespace FLASH_NAMESPACE {

template<>
void run_mha_fwd_<cutlass::half_t, 128, true>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_plain_or_aligned<cutlass::half_t, 128, true>(params, stream);
}

} // namespace FLASH_NAMESPACE
