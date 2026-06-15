#ifndef __CPU_REF_OPS_H_CA__
#define __CPU_REF_OPS_H_CA__

#include <core/Layer.h>
#include <core/threads.h>

namespace Kernel {
namespace backend {
namespace cpu_ref {

void prepare_graph_input(core::LayerSlice *ls);
void prepare_graph_output(core::LayerSlice *ls);
void prepare_relu(core::LayerSlice *ls);
void prepare_dropout(core::LayerSlice *ls);
void prepare_flatten(core::LayerSlice *ls);
void prepare_linear(core::LayerSlice *ls);
void prepare_conv2d(core::LayerSlice *ls);
void prepare_adaptiveavgpool2d(core::LayerSlice *ls);
void prepare_batchnorm2d(core::LayerSlice *ls);

void execute_graph_input(core::LayerSlice *ls, ThreadCtx_t *ctx);
void execute_graph_output(core::LayerSlice *ls, ThreadCtx_t *ctx);
void execute_relu(core::LayerSlice *ls, ThreadCtx_t *ctx);
void execute_dropout(core::LayerSlice *ls, ThreadCtx_t *ctx);
void execute_flatten(core::LayerSlice *ls, ThreadCtx_t *ctx);
void execute_linear(core::LayerSlice *ls, ThreadCtx_t *ctx);
void execute_conv2d(core::LayerSlice *ls, ThreadCtx_t *ctx);
void execute_adaptiveavgpool2d(core::LayerSlice *ls, ThreadCtx_t *ctx);
void execute_batchnorm2d(core::LayerSlice *ls, ThreadCtx_t *ctx);

} // namespace cpu_ref
} // namespace backend
} // namespace Kernel

#endif
