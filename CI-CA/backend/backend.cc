#include <backend/backend.h>
#include "cpu/cpu_ops.h"

namespace Kernel {
namespace backend {

void CpuBackend::execute(LayerSlice* ls, ThreadCtx_t* ctx) {
    EXIT_ERROR_CHECK_EQ(nullptr, ls, "LayerSlice is nullptr");
    Layer* l = ls->layer();
    EXIT_ERROR_CHECK_EQ(nullptr, l, "Layer is nullptr");

    switch (l->type()) {
        case LayerType::GRAPH_INPUT:
            cpu::execute_graph_input(ls, ctx);
            break;
        case LayerType::GRAPH_OUTPUT:
            cpu::execute_graph_output(ls, ctx);
            break;
        case LayerType::RELU:
            cpu::execute_relu(ls, ctx);
            break;
        case LayerType::SIGMOID:
            cpu::execute_sigmoid(ls, ctx);
            break;
        case LayerType::DROPOUT:
            cpu::execute_dropout(ls, ctx);
            break;
        case LayerType::SOFTMAX:
            cpu::execute_softmax(ls, ctx);
            break;
        case LayerType::ADD:
            cpu::execute_add(ls, ctx);
            break;
        case LayerType::BIASADD:
            cpu::execute_biasadd(ls, ctx);
            break;
        case LayerType::CONCAT:
            cpu::execute_concat(ls, ctx);
            break;
        case LayerType::MATMUL:
            cpu::execute_matmul(ls, ctx);
            break;
        case LayerType::LINEAR:
            cpu::execute_linear(ls, ctx);
            break;
        case LayerType::CONV2D:
            cpu::execute_conv2d(ls, ctx);
            break;
        case LayerType::MAXPOOL2D:
            cpu::execute_maxpool2d(ls, ctx);
            break;
        case LayerType::AVGPOOL2D:
            cpu::execute_avgpool2d(ls, ctx);
            break;
        case LayerType::ADAPTIVEAVGPOOL2D:
            cpu::execute_adaptiveavgpool2d(ls, ctx);
            break;
        case LayerType::ADAPTIVEMAXPOOL2D:
            cpu::execute_adaptivemaxpool2d(ls, ctx);
            break;
        case LayerType::BATCHNORM2D:
            cpu::execute_batchnorm2d(ls, ctx);
            break;
        case LayerType::LAYERNORM:
            cpu::execute_layernorm(ls, ctx);
            break;
        case LayerType::GROUPNORM:
            cpu::execute_groupnorm(ls, ctx);
            break;
        default:
            LogDebug("CpuBackend::execute skip unsupported layer type");
            break;
    }
    return;
}

void CpuBackend_TEE::execute(LayerSlice* ls, ThreadCtx_t* ctx) {
    (void)ls;
    (void)ctx;
    return;
}


} // namespace end of backend
} // namespace end of Kernel 
