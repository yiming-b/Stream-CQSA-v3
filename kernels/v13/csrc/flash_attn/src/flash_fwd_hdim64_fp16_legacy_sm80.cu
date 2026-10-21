// next/ v12f: v11's CQS-on forward (LEGACY: per-element mask path), alone in
// its translation unit so that its long ptxas run is paid only when it changes.
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template<>
void run_mha_fwd_legacy_<cutlass::half_t, 64, false>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim64<cutlass::half_t, false, /*Cqs_on=*/true>(params, stream);
}

} // namespace FLASH_NAMESPACE
