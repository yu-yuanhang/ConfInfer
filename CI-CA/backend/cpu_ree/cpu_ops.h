#ifndef __CPU_OPS_H_CA__
#define __CPU_OPS_H_CA__

#include <core/Layer.h>

namespace Kernel {
namespace backend {
namespace cpu {

void* require_workspace(core::Layer *layer, core::ExecContext_t *ctx, UINT bytes);

void prepare_graph_input(core::Layer *layer);
void prepare_graph_output(core::Layer *layer);
void prepare_relu(core::Layer *layer);
void prepare_sigmoid(core::Layer *layer);
void prepare_dropout(core::Layer *layer);
void prepare_flatten(core::Layer *layer);
void prepare_softmax(core::Layer *layer);
void prepare_add(core::Layer *layer);
void prepare_biasadd(core::Layer *layer);
void prepare_concat(core::Layer *layer);
void prepare_matmul(core::Layer *layer);
void prepare_linear(core::Layer *layer);
void prepare_conv2d(core::Layer *layer);
void prepare_maxpool2d(core::Layer *layer);
void prepare_avgpool2d(core::Layer *layer);
void prepare_adaptiveavgpool2d(core::Layer *layer);
void prepare_adaptivemaxpool2d(core::Layer *layer);
void prepare_batchnorm2d(core::Layer *layer);
void prepare_layernorm(core::Layer *layer);
void prepare_groupnorm(core::Layer *layer);

void execute_graph_input(core::Layer *layer, core::ExecContext_t *ctx);
void execute_graph_output(core::Layer *layer, core::ExecContext_t *ctx);
void execute_relu(core::Layer *layer, core::ExecContext_t *ctx);
void execute_sigmoid(core::Layer *layer, core::ExecContext_t *ctx);
void execute_dropout(core::Layer *layer, core::ExecContext_t *ctx);
void execute_flatten(core::Layer *layer, core::ExecContext_t *ctx);
void execute_softmax(core::Layer *layer, core::ExecContext_t *ctx);
void execute_add(core::Layer *layer, core::ExecContext_t *ctx);
void execute_biasadd(core::Layer *layer, core::ExecContext_t *ctx);
void execute_concat(core::Layer *layer, core::ExecContext_t *ctx);
void execute_matmul(core::Layer *layer, core::ExecContext_t *ctx);
void execute_linear(core::Layer *layer, core::ExecContext_t *ctx);
void execute_conv2d(core::Layer *layer, core::ExecContext_t *ctx);
void execute_maxpool2d(core::Layer *layer, core::ExecContext_t *ctx);
void execute_avgpool2d(core::Layer *layer, core::ExecContext_t *ctx);
void execute_adaptiveavgpool2d(core::Layer *layer, core::ExecContext_t *ctx);
void execute_adaptivemaxpool2d(core::Layer *layer, core::ExecContext_t *ctx);
void execute_batchnorm2d(core::Layer *layer, core::ExecContext_t *ctx);
void execute_layernorm(core::Layer *layer, core::ExecContext_t *ctx);
void execute_groupnorm(core::Layer *layer, core::ExecContext_t *ctx);

} // namespace cpu
} // namespace backend
} // namespace Kernel

#endif
