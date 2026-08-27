#include <backend/backend.h>
#include "cpu_ree/cpu_ops.h"
#include "cpu_ree_ref/cpu_ref_ops.h"

namespace Kernel {
namespace backend {

namespace {
void bind_cpu_exec(Layer* layer, LayerType type) {
    EXIT_ERROR_CHECK_EQ(nullptr, layer, "Layer is nullptr");

    switch (type) {
        case LayerType::GRAPH_INPUT:
            cpu::prepare_graph_input(layer);
            layer->setExec(cpu::execute_graph_input);
            break;
        case LayerType::GRAPH_OUTPUT:
            cpu::prepare_graph_output(layer);
            layer->setExec(cpu::execute_graph_output);
            break;
        case LayerType::RELU:
            cpu::prepare_relu(layer);
            layer->setExec(cpu::execute_relu);
            break;
        case LayerType::SIGMOID:
            cpu::prepare_sigmoid(layer);
            layer->setExec(cpu::execute_sigmoid);
            break;
        case LayerType::DROPOUT:
            cpu::prepare_dropout(layer);
            layer->setExec(cpu::execute_dropout);
            break;
        case LayerType::FLATTEN:
            cpu::prepare_flatten(layer);
            layer->setExec(cpu::execute_flatten);
            break;
        case LayerType::SOFTMAX:
            cpu::prepare_softmax(layer);
            layer->setExec(cpu::execute_softmax);
            break;
        case LayerType::ADD:
            cpu::prepare_add(layer);
            layer->setExec(cpu::execute_add);
            break;
        case LayerType::BIASADD:
            cpu::prepare_biasadd(layer);
            layer->setExec(cpu::execute_biasadd);
            break;
        case LayerType::CONCAT:
            cpu::prepare_concat(layer);
            layer->setExec(cpu::execute_concat);
            break;
        case LayerType::MATMUL:
            cpu::prepare_matmul(layer);
            layer->setExec(cpu::execute_matmul);
            break;
        case LayerType::LINEAR:
            cpu::prepare_linear(layer);
            layer->setExec(cpu::execute_linear);
            break;
        case LayerType::CONV2D:
            cpu::prepare_conv2d(layer);
            layer->setExec(cpu::execute_conv2d);
            break;
        case LayerType::MAXPOOL2D:
            cpu::prepare_maxpool2d(layer);
            layer->setExec(cpu::execute_maxpool2d);
            break;
        case LayerType::AVGPOOL2D:
            cpu::prepare_avgpool2d(layer);
            layer->setExec(cpu::execute_avgpool2d);
            break;
        case LayerType::ADAPTIVEAVGPOOL2D:
            cpu::prepare_adaptiveavgpool2d(layer);
            layer->setExec(cpu::execute_adaptiveavgpool2d);
            break;
        case LayerType::ADAPTIVEMAXPOOL2D:
            cpu::prepare_adaptivemaxpool2d(layer);
            layer->setExec(cpu::execute_adaptivemaxpool2d);
            break;
        case LayerType::BATCHNORM2D:
            cpu::prepare_batchnorm2d(layer);
            layer->setExec(cpu::execute_batchnorm2d);
            break;
        case LayerType::LAYERNORM:
            cpu::prepare_layernorm(layer);
            layer->setExec(cpu::execute_layernorm);
            break;
        case LayerType::GROUPNORM:
            cpu::prepare_groupnorm(layer);
            layer->setExec(cpu::execute_groupnorm);
            break;
        default:
            EXIT_ERROR("Backend_CPU_REE unsupported layer type: %u", static_cast<UINT>(type));
    }
}

void bind_cpu_ref_exec(Layer* layer, LayerType type) {
    EXIT_ERROR_CHECK_EQ(nullptr, layer, "Layer is nullptr");

    switch (type) {
        case LayerType::GRAPH_INPUT:
            cpu_ref::prepare_graph_input(layer);
            layer->setExec(cpu_ref::execute_graph_input);
            break;
        case LayerType::GRAPH_OUTPUT:
            cpu_ref::prepare_graph_output(layer);
            layer->setExec(cpu_ref::execute_graph_output);
            break;
        case LayerType::RELU:
            cpu_ref::prepare_relu(layer);
            layer->setExec(cpu_ref::execute_relu);
            break;
        case LayerType::DROPOUT:
            cpu_ref::prepare_dropout(layer);
            layer->setExec(cpu_ref::execute_dropout);
            break;
        case LayerType::FLATTEN:
            cpu_ref::prepare_flatten(layer);
            layer->setExec(cpu_ref::execute_flatten);
            break;
        case LayerType::LINEAR:
            cpu_ref::prepare_linear(layer);
            layer->setExec(cpu_ref::execute_linear);
            break;
        case LayerType::CONV2D:
            cpu_ref::prepare_conv2d(layer);
            layer->setExec(cpu_ref::execute_conv2d);
            break;
        case LayerType::ADAPTIVEAVGPOOL2D:
            cpu_ref::prepare_adaptiveavgpool2d(layer);
            layer->setExec(cpu_ref::execute_adaptiveavgpool2d);
            break;
        case LayerType::BATCHNORM2D:
            cpu_ref::prepare_batchnorm2d(layer);
            layer->setExec(cpu_ref::execute_batchnorm2d);
            break;
        default:
            EXIT_ERROR("Backend_CPU_REE_REF unsupported layer type: %u", static_cast<UINT>(type));
    }
}

} // namespace

void Backend::prepare(const ExecPartition& part, ExecContext_t* ctx) {
    (void)ctx;
    for (auto it = part.topo().begin(); it != part.topo().end(); ++it) {
        prepare(*it);
    }
}

void Backend::execute(const ExecPartition& part, ExecContext_t* ctx) {
    for (auto it = part.topo().begin(); it != part.topo().end(); ++it) {
        execute(*it, ctx);
    }
}

void Backend::resetRuntime(ExecContext_t* ctx, bool strict) {
    (void)ctx;
    (void)strict;
}

void Backend_CPU_REE::prepare(Layer* layer) {
    EXIT_ERROR_CHECK_EQ(nullptr, layer, "Layer is nullptr");
    // prepare 阶段绑定 后端 和 运行时 后续 execute 就不需要重复这些操作
    layer->setBackend(this);
    bind_cpu_exec(layer, layer->type());
}

void Backend_CPU_REE::execute(Layer* layer, ExecContext_t* ctx) {
    EXIT_ERROR_CHECK_EQ(nullptr, layer, "Layer is nullptr");
    if (nullptr == layer->exec()) {
        prepare(layer);
    }
    layer->execute(ctx);
}

void Backend_CPU_REE_REF::prepare(Layer* layer) {
    EXIT_ERROR_CHECK_EQ(nullptr, layer, "Layer is nullptr");
    layer->setBackend(this);
    bind_cpu_ref_exec(layer, layer->type());
}

void Backend_CPU_REE_REF::execute(Layer* layer, ExecContext_t* ctx) {
    EXIT_ERROR_CHECK_EQ(nullptr, layer, "Layer is nullptr");
    if (nullptr == layer->exec()) {
        prepare(layer);
    }
    layer->execute(ctx);
}


} // namespace end of backend
} // namespace end of Kernel 
