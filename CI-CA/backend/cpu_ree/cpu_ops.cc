#include <core/Network.h>
#include "cpu_ops.h"

namespace Kernel {
namespace backend {
namespace cpu {

void* require_workspace(core::Layer *layer, core::ExecContext_t *ctx, UINT bytes) {
    EXIT_ERROR_CHECK_EQ(nullptr, layer, "Layer is nullptr");
    EXIT_ERROR_CHECK_EQ(nullptr, ctx, "ExecContext is nullptr");

    if (0 == bytes) {
        return nullptr;
    }

    EXIT_ERROR_CHECK_EQ(nullptr, ctx->workspace, "Network workspace is nullptr");
    EXIT_ERROR_CHECK_NE(true,
        layer->workspaceSize() <= ctx->wsSize,
        "Layer workspace range overflow");
    EXIT_ERROR_CHECK_NE(true, bytes <= layer->workspaceSize(),
        "Layer workspace size is not enough");

    return ctx->workspace;
}

void prepare_graph_input(core::Layer *layer) {
    EXIT_ERROR_CHECK_EQ(nullptr, layer, "Layer is nullptr");
}

void prepare_graph_output(core::Layer *layer) {
    EXIT_ERROR_CHECK_EQ(nullptr, layer, "Layer is nullptr");
}

void execute_graph_input(core::Layer *layer, core::ExecContext_t *ctx) {
    (void)layer;
    (void)ctx;
}

void execute_graph_output(core::Layer *layer, core::ExecContext_t *ctx) {
    (void)layer;
    (void)ctx;
}

} // namespace cpu
} // namespace backend
} // namespace Kernel
