// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// native_v2: the whole backward of a call (preprocess, census + ALIGNED wave backward
// for CQS-on calls, the same loop without a run table for CQS-off calls,
// convert_dQ); v11's dq/dk/dv kernel (LEGACY) is in flash_bwd_hdim128_fp16_legacy_sm80.cu.
#include "namespace_config.h"
#include "flash_bwd_aligned.h"

namespace FLASH_NAMESPACE {

template<>
void run_mha_bwd_<cutlass::half_t, 128, false>(Flash_bwd_params &params, cudaStream_t stream) {
    run_mha_bwd_hdim128<cutlass::half_t, false, BwdPart::Main>(params, stream);
}

template<>
void run_mha_bwd_<cutlass::half_t, 128, true>(Flash_bwd_params &params, cudaStream_t stream) {
    run_mha_bwd_hdim128<cutlass::half_t, true, BwdPart::Main>(params, stream);
}

} // namespace FLASH_NAMESPACE
