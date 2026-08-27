#include <core/Network.h>
#include <core/BoundaryLayer.h>

namespace Kernel {
namespace core {

namespace {

BackendKind backend_kind_for_domain(ExecutionDomain domain, BackendKind preferred_kind) {
    switch (domain) {
    case ExecutionDomain::ED_CPU_REE:
        return BackendKind::BK_CPU_REE;
    case ExecutionDomain::ED_CPU_TEE:
        return BackendKind::BK_CPU_TEE;
    case ExecutionDomain::ED_DEFAULT:
    default:
        return preferred_kind;
    }
}

ExecutionDomain execution_domain_from_flags(uint32_t lf) {
    if (lf & LF_REQUIRE_TEE) {
        return ExecutionDomain::ED_CPU_TEE;
    }
    return ExecutionDomain::ED_DEFAULT;
}

BackendKind fallback_backend_kind_for_domain(ExecutionDomain domain, BackendKind preferred_kind) {
    switch (domain) {
    case ExecutionDomain::ED_CPU_TEE:
        // 图语义上仍然保留 TEE 执行域。
        // 但当 bridge 不存在时，运行时执行实现需要降级到本地参考后端，
        // 否则会错误地落到 Backend_CPU_TEE 这个占位实现里。
        return BackendKind::BK_CPU_REE_REF;
    case ExecutionDomain::ED_CPU_REE:
    case ExecutionDomain::ED_DEFAULT:
    default:
        return preferred_kind;
    }
}

bool is_tee_partition(const ExecPartition& part) {
    return part.domain() == ExecutionDomain::ED_CPU_TEE;
}

Backend_CPU_TEE *tee_backend(Executor *exec) {
    Backend *backend = nullptr;

    EXIT_ERROR_CHECK_EQ(nullptr, exec, "Executor is nullptr");
    backend = exec->backend(BackendKind::BK_CPU_TEE);
    if (nullptr == backend) {
        return nullptr;
    }
    return dynamic_cast<Backend_CPU_TEE *>(backend);
}

Backend *resolve_backend_for_domain(const Executor *exec,
                                    ExecutionDomain domain,
                                    BackendKind preferred_kind) {
    BackendKind kind = backend_kind_for_domain(domain, preferred_kind);

    EXIT_ERROR_CHECK_EQ(nullptr, exec, "Executor is nullptr");

    if (domain == ExecutionDomain::ED_CPU_TEE) {
#if ENABLE_TEE_BRIDGE
        kind = BackendKind::BK_CPU_TEE;
#else
        kind = fallback_backend_kind_for_domain(ExecutionDomain::ED_CPU_TEE, preferred_kind);
#endif
    }

    if (Backend *selected = exec->backend(kind)) {
        return selected;
    }

    if (kind == BackendKind::BK_CPU_REE_REF) {
        return exec->backend(BackendKind::BK_CPU_REE);
    }
    return nullptr;
}

} // namespace

std::atomic<confinfer_model_id_t> Network::_modelCounter{1};

Executor::Executor(): _by_kind(), _preferred_kind(BackendKind::BK_CPU_REE) {
    static Backend_CPU_REE cpu_backend;
    static Backend_CPU_REE_REF cpu_ref_backend;
    static Backend_CPU_TEE cpu_tee_backend;
    setBackends({&cpu_backend, &cpu_ref_backend, &cpu_tee_backend});
}

Executor::~Executor() {
    _by_kind.clear();
}

void Executor::setBackends(std::vector<Backend *> backends) {
    _by_kind.clear();
    for (auto it = backends.begin(); it != backends.end(); ++it) {
        _by_kind[(*it)->kind()].push_back(*it);
    }
}

void Executor::setBackendKind(BackendKind kind) {
    _preferred_kind = kind;
}

Backend *Executor::backend(BackendKind kind) const {
    auto it = _by_kind.find(kind);
    if (it == _by_kind.end() || it->second.empty()) {
        return nullptr;
    }
    return it->second.front();
}

bool Executor::supports(BackendKind kind) const {
    return nullptr != backend(kind);
}

bool Executor::supports(ExecutionDomain domain) const {
    if (domain == ExecutionDomain::ED_DEFAULT) {
        return supports(BackendKind::BK_CPU_REE);
    }
    if (domain == ExecutionDomain::ED_CPU_TEE) {
#if ENABLE_TEE_BRIDGE
        return supports(BackendKind::BK_CPU_TEE);
#else
        return false;
#endif
    }
    return supports(backend_kind_for_domain(domain, _preferred_kind));
}

void Executor::prepare_layer(Layer* layer)
{
    EXIT_ERROR_CHECK_EQ(nullptr, layer, "Layer is nullptr");
    Backend *backend = route(layer->flags());
    EXIT_ERROR_CHECK_EQ(nullptr, backend, "No available backend for layer");
    layer->setBackend(backend);
    backend->prepare(layer);
}
void Executor::execute_layer(Layer* layer, ExecContext_t* ctx)
{
    EXIT_ERROR_CHECK_EQ(nullptr, layer, "Layer is nullptr");
    Backend *backend = layer->backend();
    if (nullptr == backend) {
        prepare_layer(layer);
        backend = layer->backend();
    }
    backend->execute(layer, ctx);
    return;
}

void Executor::prepare_partition(ExecPartition& part, ExecContext_t* ctx) {
    Backend *backend = route(part);

    EXIT_ERROR_CHECK_EQ(nullptr, backend, "No available backend for partition");
    part.setBackend(backend);
    backend->prepare(part, ctx);
}

void Executor::execute_partition(ExecPartition& part, ExecContext_t* ctx) {
    Backend *backend = part.backend();

    if (nullptr == backend) {
        backend = route(part);
        EXIT_ERROR_CHECK_EQ(nullptr, backend, "No available backend for partition");
        part.setBackend(backend);
    }
    backend->execute(part, ctx);
}

void Executor::reset_runtime(ExecContext_t* ctx, bool strict) {
    for (auto it = _by_kind.begin(); it != _by_kind.end(); ++it) {
        for (auto bit = it->second.begin(); bit != it->second.end(); ++bit) {
            EXIT_ERROR_CHECK_EQ(nullptr, *bit, "Backend is nullptr");
            (*bit)->resetRuntime(ctx, strict);
        }
    }
}

Backend *Executor::route(uint32_t lf) {
    return resolve_backend_for_domain(this,
                                      execution_domain_from_flags(lf),
                                      _preferred_kind);
}

Backend *Executor::route(const ExecPartition& part) {
    return resolve_backend_for_domain(this, part.domain(), _preferred_kind);
}

bool is_exec_domain_registered(ExecutionDomain domain) {
    Executor *exec = EXECUTOR;
    EXIT_ERROR_CHECK_EQ(nullptr, exec, "Executor singleton is nullptr");
    return exec->supports(domain);
}

// ...... todo 整一个 Network::Network(Graph &graph, Executor *exec): 版本
// 直接构造 prepare 一步到位
Network::Network(Graph &graph):
    _fullGraph(&graph),
    _modelId(_modelCounter.fetch_add(1, std::memory_order_relaxed))
{
    _execCtx.wsSize = graph.WorkspaceSize();
    _execCtx.modelId = _modelId;
    _execCtx.parts = nullptr;
    if (_execCtx.wsSize) {
        _execCtx.workspace = static_cast<void*>(new char[_execCtx.wsSize]);
    }
}

Network::~Network() {
    EXECUTOR->reset_runtime(&_execCtx, false);
    if (_execCtx.workspace && _execCtx.wsSize) {
        delete[] static_cast<char*>(_execCtx.workspace);
        _execCtx.workspace = nullptr;
        _execCtx.wsSize = 0;
    }
    _execCtx.parts = nullptr;
}

void Network::buildPartGraph() {
    EXIT_ERROR_CHECK_EQ(nullptr, _fullGraph, "_fullGraph is nullptr");
    _partGraph.build(*_fullGraph);
}

void Network::prepare(Executor *exec) {
    EXIT_ERROR_CHECK_EQ(nullptr, exec, "Executor is nullptr");
    exec->reset_runtime(&_execCtx, false);

    buildPartGraph();
    _execCtx.modelId = _modelId;
    _execCtx.parts = &_partGraph.parts();

    if (!_partGraph.empty()) {
        const std::vector<UINT>& order = _partGraph.topoOrder();
        for (auto it = order.begin(); it != order.end(); ++it) {
            exec->prepare_partition(_partGraph.part(*it), &_execCtx);
        }
        return;
    }

    // 一般不应该走到这里
    for (auto it = _fullGraph->execOrder().begin(); it != _fullGraph->execOrder().end(); ++it) {
        exec->prepare_layer(*it);
    }
}

// 清空 TEE 内数据
void Network::teardown(Executor *exec) {
    EXIT_ERROR_CHECK_EQ(nullptr, exec, "Executor is nullptr");
    exec->reset_runtime(&_execCtx, true);
}

void Network::runNet(Executor *exec) {
    ExecContext_t *ctx = &_execCtx;
    if (!_partGraph.empty()) {
        const std::vector<UINT>& order = _partGraph.topoOrder();
        for (auto it = order.begin(); it != order.end(); ++it) {
            exec->execute_partition(_partGraph.part(*it), ctx);
        }
        return;
    }
    for (auto it = _fullGraph->execOrder().begin(); it != _fullGraph->execOrder().end(); ++it) {
        exec->execute_layer(*it, ctx);
    }
}

void Network::run(std::initializer_list<Value_t*> inputs,
                  std::initializer_list<Value_t*> outputs,
                  Executor *exec) {
    std::vector<Value_t*> input_vec(inputs.begin(), inputs.end());
    std::vector<Value_t*> output_vec(outputs.begin(), outputs.end());
    run(input_vec, output_vec, exec);
}

void Network::run(const std::vector<Value_t*>& inputs, std::vector<Value_t*>& outputs,
                  Executor *exec) {
    const GraphSignature& sig = _fullGraph->signature();
    EXIT_ERROR_CHECK_NE(inputs.size(), sig.inputs.size(), "Network input size mismatch");
    EXIT_ERROR_CHECK_NE(outputs.size(), sig.outputs.size(), "Network output size mismatch");

    Layer* in_boundary = _fullGraph->inputBoundary();
    EXIT_ERROR_CHECK_EQ(nullptr, in_boundary, "Graph input boundary is nullptr");

    for (UINT i = 0; i < sig.inputs.size(); ++i) {
        EXIT_ERROR_CHECK_EQ(nullptr, inputs[i], "Network input value is nullptr");
        EXIT_ERROR_CHECK_EQ(nullptr, inputs[i]->data.ptr, "Network input data ptr is nullptr");
        const Value_t* expected = sig.inputs[i].value;
        EXIT_ERROR_CHECK_EQ(nullptr, expected, "Graph signature input is nullptr");
        EXIT_ERROR_CHECK_NE(expected->data.dtype, inputs[i]->data.dtype,
            "Network input dtype mismatch");
        EXIT_ERROR_CHECK_NE(expected->data.shape.ndim, inputs[i]->data.shape.ndim,
            "Network input ndim mismatch");
        for (UINT d = 0; d < expected->data.shape.ndim; ++d) {
            EXIT_ERROR_CHECK_NE(expected->data.shape.dims[d], inputs[i]->data.shape.dims[d],
                "Network input shape mismatch");
        }
        Value_t& v = in_boundary->output(i);
        v.borrowFrom(*inputs[i], PARAM_INPUT);
    }

    runNet(exec);

    Layer* out_boundary = _fullGraph->outputBoundary();
    EXIT_ERROR_CHECK_EQ(nullptr, out_boundary, "Graph output boundary is nullptr");
    GraphOutputLayer* out_layer = static_cast<GraphOutputLayer*>(out_boundary);

    for (UINT i = 0; i < sig.outputs.size(); ++i) {
        EXIT_ERROR_CHECK_EQ(nullptr, outputs[i], "Network output value is nullptr");
        Value_t& value = out_layer->input(i);
        outputs[i]->deepCopyFrom(value, PARAM_OUTPUT);
    }
}


} // namespace end of core
} // namespace end of Kernel 
