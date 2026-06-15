#include <backend/backend.h>
#include "cpu_ree/cpu_ops.h"
#include "cpu_ree_ref/cpu_ref_ops.h"

namespace Kernel {
namespace backend {

namespace {
void bind_cpu_exec(LayerSlice* ls, LayerType type) {
    EXIT_ERROR_CHECK_EQ(nullptr, ls, "LayerSlice is nullptr");

    switch (type) {
        case LayerType::GRAPH_INPUT:
            cpu::prepare_graph_input(ls);
            ls->setExec(cpu::execute_graph_input);
            break;
        case LayerType::GRAPH_OUTPUT:
            cpu::prepare_graph_output(ls);
            ls->setExec(cpu::execute_graph_output);
            break;
        case LayerType::RELU:
            cpu::prepare_relu(ls);
            ls->setExec(cpu::execute_relu);
            break;
        case LayerType::SIGMOID:
            cpu::prepare_sigmoid(ls);
            ls->setExec(cpu::execute_sigmoid);
            break;
        case LayerType::DROPOUT:
            cpu::prepare_dropout(ls);
            ls->setExec(cpu::execute_dropout);
            break;
        case LayerType::FLATTEN:
            cpu::prepare_flatten(ls);
            ls->setExec(cpu::execute_flatten);
            break;
        case LayerType::SOFTMAX:
            cpu::prepare_softmax(ls);
            ls->setExec(cpu::execute_softmax);
            break;
        case LayerType::ADD:
            cpu::prepare_add(ls);
            ls->setExec(cpu::execute_add);
            break;
        case LayerType::BIASADD:
            cpu::prepare_biasadd(ls);
            ls->setExec(cpu::execute_biasadd);
            break;
        case LayerType::CONCAT:
            cpu::prepare_concat(ls);
            ls->setExec(cpu::execute_concat);
            break;
        case LayerType::MATMUL:
            cpu::prepare_matmul(ls);
            ls->setExec(cpu::execute_matmul);
            break;
        case LayerType::LINEAR:
            cpu::prepare_linear(ls);
            ls->setExec(cpu::execute_linear);
            break;
        case LayerType::CONV2D:
            cpu::prepare_conv2d(ls);
            ls->setExec(cpu::execute_conv2d);
            break;
        case LayerType::MAXPOOL2D:
            cpu::prepare_maxpool2d(ls);
            ls->setExec(cpu::execute_maxpool2d);
            break;
        case LayerType::AVGPOOL2D:
            cpu::prepare_avgpool2d(ls);
            ls->setExec(cpu::execute_avgpool2d);
            break;
        case LayerType::ADAPTIVEAVGPOOL2D:
            cpu::prepare_adaptiveavgpool2d(ls);
            ls->setExec(cpu::execute_adaptiveavgpool2d);
            break;
        case LayerType::ADAPTIVEMAXPOOL2D:
            cpu::prepare_adaptivemaxpool2d(ls);
            ls->setExec(cpu::execute_adaptivemaxpool2d);
            break;
        case LayerType::BATCHNORM2D:
            cpu::prepare_batchnorm2d(ls);
            ls->setExec(cpu::execute_batchnorm2d);
            break;
        case LayerType::LAYERNORM:
            cpu::prepare_layernorm(ls);
            ls->setExec(cpu::execute_layernorm);
            break;
        case LayerType::GROUPNORM:
            cpu::prepare_groupnorm(ls);
            ls->setExec(cpu::execute_groupnorm);
            break;
        default:
            EXIT_ERROR("Backend_CPU_REE unsupported layer type: %u", static_cast<UINT>(type));
    }
}

void bind_cpu_ref_exec(LayerSlice* ls, LayerType type) {
    EXIT_ERROR_CHECK_EQ(nullptr, ls, "LayerSlice is nullptr");

    switch (type) {
        case LayerType::GRAPH_INPUT:
            cpu_ref::prepare_graph_input(ls);
            ls->setExec(cpu_ref::execute_graph_input);
            break;
        case LayerType::GRAPH_OUTPUT:
            cpu_ref::prepare_graph_output(ls);
            ls->setExec(cpu_ref::execute_graph_output);
            break;
        case LayerType::RELU:
            cpu_ref::prepare_relu(ls);
            ls->setExec(cpu_ref::execute_relu);
            break;
        case LayerType::DROPOUT:
            cpu_ref::prepare_dropout(ls);
            ls->setExec(cpu_ref::execute_dropout);
            break;
        case LayerType::FLATTEN:
            cpu_ref::prepare_flatten(ls);
            ls->setExec(cpu_ref::execute_flatten);
            break;
        case LayerType::LINEAR:
            cpu_ref::prepare_linear(ls);
            ls->setExec(cpu_ref::execute_linear);
            break;
        case LayerType::CONV2D:
            cpu_ref::prepare_conv2d(ls);
            ls->setExec(cpu_ref::execute_conv2d);
            break;
        case LayerType::ADAPTIVEAVGPOOL2D:
            cpu_ref::prepare_adaptiveavgpool2d(ls);
            ls->setExec(cpu_ref::execute_adaptiveavgpool2d);
            break;
        case LayerType::BATCHNORM2D:
            cpu_ref::prepare_batchnorm2d(ls);
            ls->setExec(cpu_ref::execute_batchnorm2d);
            break;
        default:
            EXIT_ERROR("Backend_CPU_REE_REF unsupported layer type: %u", static_cast<UINT>(type));
    }
}
} // namespace

void Backend_CPU_REE::prepare(LayerSlice* ls) {
    EXIT_ERROR_CHECK_EQ(nullptr, ls, "LayerSlice is nullptr");
    Layer* l = ls->layer();
    EXIT_ERROR_CHECK_EQ(nullptr, l, "Layer is nullptr");
    // prepare 阶段绑定 后端 和 运行时 后续 execute 就不需要重复这些操作
    ls->setBackend(this);
    bind_cpu_exec(ls, l->type());
}

void Backend_CPU_REE::execute(LayerSlice* ls, ThreadCtx_t* ctx) {
    EXIT_ERROR_CHECK_EQ(nullptr, ls, "LayerSlice is nullptr");
    if (nullptr == ls->exec()) {
        prepare(ls);
    }
    ls->execute(ctx);
}

void Backend_CPU_REE_REF::prepare(LayerSlice* ls) {
    EXIT_ERROR_CHECK_EQ(nullptr, ls, "LayerSlice is nullptr");
    Layer* l = ls->layer();
    EXIT_ERROR_CHECK_EQ(nullptr, l, "Layer is nullptr");
    ls->setBackend(this);
    bind_cpu_ref_exec(ls, l->type());
}

void Backend_CPU_REE_REF::execute(LayerSlice* ls, ThreadCtx_t* ctx) {
    EXIT_ERROR_CHECK_EQ(nullptr, ls, "LayerSlice is nullptr");
    if (nullptr == ls->exec()) {
        prepare(ls);
    }
    ls->execute(ctx);
}

void Backend_CPU_TEE::prepare(LayerSlice* ls) {
    EXIT_ERROR_CHECK_EQ(nullptr, ls, "LayerSlice is nullptr");
    ls->setBackend(this);
    ls->setExec(nullptr);
}

void Backend_CPU_TEE::execute(LayerSlice* ls, ThreadCtx_t* ctx) {
    (void)ls;
    (void)ctx;
    return;
}


} // namespace end of backend
} // namespace end of Kernel 
