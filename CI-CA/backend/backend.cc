#include <backend/backend.h>

namespace Kernel {
namespace backend {

void CpuBackend::execute(LayerSlice* ls, ThreadCtx_t* ctx) {
    Layer* l = ls->layer();
    const SliceDesc_t& desc = ls->desc();


    return;
}

void CpuBackend_TEE::execute(LayerSlice* ls, ThreadCtx_t* ctx) {
    Layer* l = ls->layer();
    const SliceDesc_t& desc = ls->desc();


    return;
}


} // namespace end of backend
} // namespace end of Kernel 