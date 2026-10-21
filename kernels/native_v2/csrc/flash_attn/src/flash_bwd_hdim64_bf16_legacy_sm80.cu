// native_v2: v11's CQS backward dq/dk/dv kernel (LEGACY: per-element mask path,
// every feature switch) with the wave layout's per-subproblem offsets, alone in
// its translation unit so that its long compile is paid only when it changes.
// Launched by run_mha_bwd_ (flash_bwd_aligned.h).
#include "namespace_config.h"
#include "flash_bwd_launch_template.h"

namespace FLASH_NAMESPACE {

template<>
void run_mha_bwd_legacy_<cutlass::bfloat16_t, 64, false>(Flash_bwd_params &params, cudaStream_t stream) {
    run_mha_bwd_hdim64<cutlass::bfloat16_t, false, BwdPart::Legacy>(params, stream);
}

template<>
void run_mha_bwd_legacy_<cutlass::bfloat16_t, 64, true>(Flash_bwd_params &params, cudaStream_t stream) {
    run_mha_bwd_hdim64<cutlass::bfloat16_t, true, BwdPart::Legacy>(params, stream);
}

} // namespace FLASH_NAMESPACE
