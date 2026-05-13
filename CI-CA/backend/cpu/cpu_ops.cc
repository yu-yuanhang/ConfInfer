#include "cpu_ops.h"

namespace Kernel {
namespace backend {
namespace cpu {

WorkspaceView shared_workspace(core::LayerSlice *ls, ThreadCtx_t *ctx) {
    EXIT_ERROR_CHECK_EQ(nullptr, ls, "LayerSlice is nullptr");
    EXIT_ERROR_CHECK_EQ(nullptr, ctx, "ThreadCtx is nullptr");
    EXIT_ERROR_CHECK_EQ(nullptr, ctx->shared, "SharedContext is nullptr");

    const core::SliceDesc_t& desc = ls->desc();
    if (0 == desc.workspaceSize) {
        return WorkspaceView(nullptr, 0);
    }

    EXIT_ERROR_CHECK_EQ(nullptr, ctx->shared->workspace, "Network workspace is nullptr");
    EXIT_ERROR_CHECK_NE(true,
        desc.workspaceOffset + desc.workspaceSize <= ctx->shared->wsSize,
        "LayerSlice workspace range overflow");

    char* base = static_cast<char*>(ctx->shared->workspace);
    return WorkspaceView(base + desc.workspaceOffset, desc.workspaceSize);
}

void* require_workspace(core::LayerSlice *ls, ThreadCtx_t *ctx, UINT bytes) {
    if (0 == bytes) {
        return nullptr;
    }

    WorkspaceView ws = shared_workspace(ls, ctx);
    EXIT_ERROR_CHECK_EQ(nullptr, ws.base, "Workspace base is nullptr");
    EXIT_ERROR_CHECK_NE(true, ws.size >= bytes, "LayerSlice workspace size is not enough");
    return static_cast<void*>(ws.base);
}

void execute_graph_input(core::LayerSlice *ls, ThreadCtx_t *ctx) {
    (void)ls;
    (void)ctx;
}

void execute_graph_output(core::LayerSlice *ls, ThreadCtx_t *ctx) {
    (void)ls;
    (void)ctx;
}

} // namespace cpu
} // namespace backend
} // namespace Kernel
