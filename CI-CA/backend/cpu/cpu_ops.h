#ifndef __CPU_OPS_H_CA__
#define __CPU_OPS_H_CA__

#include <core/Layer.h>
#include <core/threads.h>

namespace Kernel {
namespace backend {
namespace cpu {

struct WorkspaceView {
    char* base;
    UINT size;

    WorkspaceView(): base(nullptr), size(0) {}
    WorkspaceView(char* ws_base, UINT ws_size): base(ws_base), size(ws_size) {}
};

WorkspaceView shared_workspace(core::LayerSlice *ls, ThreadCtx_t *ctx);
void* require_workspace(core::LayerSlice *ls, ThreadCtx_t *ctx, UINT bytes);

void execute_graph_input(core::LayerSlice *ls, ThreadCtx_t *ctx);
void execute_graph_output(core::LayerSlice *ls, ThreadCtx_t *ctx);
void execute_relu(core::LayerSlice *ls, ThreadCtx_t *ctx);
void execute_sigmoid(core::LayerSlice *ls, ThreadCtx_t *ctx);
void execute_dropout(core::LayerSlice *ls, ThreadCtx_t *ctx);
void execute_softmax(core::LayerSlice *ls, ThreadCtx_t *ctx);
void execute_add(core::LayerSlice *ls, ThreadCtx_t *ctx);
void execute_biasadd(core::LayerSlice *ls, ThreadCtx_t *ctx);
void execute_concat(core::LayerSlice *ls, ThreadCtx_t *ctx);
void execute_matmul(core::LayerSlice *ls, ThreadCtx_t *ctx);
void execute_linear(core::LayerSlice *ls, ThreadCtx_t *ctx);
void execute_conv2d(core::LayerSlice *ls, ThreadCtx_t *ctx);
void execute_maxpool2d(core::LayerSlice *ls, ThreadCtx_t *ctx);
void execute_avgpool2d(core::LayerSlice *ls, ThreadCtx_t *ctx);
void execute_adaptiveavgpool2d(core::LayerSlice *ls, ThreadCtx_t *ctx);
void execute_adaptivemaxpool2d(core::LayerSlice *ls, ThreadCtx_t *ctx);
void execute_batchnorm2d(core::LayerSlice *ls, ThreadCtx_t *ctx);
void execute_layernorm(core::LayerSlice *ls, ThreadCtx_t *ctx);
void execute_groupnorm(core::LayerSlice *ls, ThreadCtx_t *ctx);

} // namespace cpu
} // namespace backend
} // namespace Kernel

#endif
