#ifndef __CPU_REF_OPS_H_CA__
#define __CPU_REF_OPS_H_CA__

#include <core/Layer.h>

namespace Kernel {
namespace backend {
namespace cpu_ref {

void prepare_graph_input(core::Layer *layer);
void prepare_graph_output(core::Layer *layer);
void prepare_relu(core::Layer *layer);
void prepare_dropout(core::Layer *layer);
void prepare_flatten(core::Layer *layer);
void prepare_linear(core::Layer *layer);
void prepare_conv2d(core::Layer *layer);
void prepare_adaptiveavgpool2d(core::Layer *layer);
void prepare_batchnorm2d(core::Layer *layer);

void execute_graph_input(core::Layer *layer, core::ExecContext_t *ctx);
void execute_graph_output(core::Layer *layer, core::ExecContext_t *ctx);
void execute_relu(core::Layer *layer, core::ExecContext_t *ctx);
void execute_dropout(core::Layer *layer, core::ExecContext_t *ctx);
void execute_flatten(core::Layer *layer, core::ExecContext_t *ctx);
void execute_linear(core::Layer *layer, core::ExecContext_t *ctx);
void execute_conv2d(core::Layer *layer, core::ExecContext_t *ctx);
void execute_adaptiveavgpool2d(core::Layer *layer, core::ExecContext_t *ctx);
void execute_batchnorm2d(core::Layer *layer, core::ExecContext_t *ctx);

} // namespace cpu_ref
} // namespace backend
} // namespace Kernel

#endif
