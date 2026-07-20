#include <core/Network.h>
#include <core/BoundaryLayer.h>
#include <core/ExecBridgeProto.h>
#if ENABLE_TEE_BRIDGE
#include <bridges/TeeExecBridge.h>
#endif

namespace Kernel {
namespace core {

namespace {

BackendKind backend_kind_from_execution_domain(ExecutionDomain domain, BackendKind preferred_kind) {
    switch (domain) {
    // 这里目前的设计就是 一个执行区域 对应一种 BackendKind
    case ExecutionDomain::ED_CPU_REE:
        return BackendKind::BK_CPU_REE;
    case ExecutionDomain::ED_CPU_TEE:
        return BackendKind::BK_CPU_TEE;
    case ExecutionDomain::ED_DEFAULT:
    default:
        return preferred_kind;
    }
}

BackendKind fallback_backend_kind_for_domain(ExecutionDomain domain, BackendKind preferred_kind) {
    switch (domain) {
    case ExecutionDomain::ED_CPU_TEE:
        // 当 TEE bridge 还没挂上时，先回退到本地参考实现，保证 CI-CA 可独立运行。
        // return BackendKind::BK_CPU_REE_REF;
        return BackendKind::BK_CPU_TEE;
    case ExecutionDomain::ED_CPU_REE:
    case ExecutionDomain::ED_DEFAULT:
    default:
        return preferred_kind;
    }
}

// unordered_map 哈希表型键值对容器 
// key-value 存储 无序
using SliceMap = std::unordered_map<const Layer *, LayerSlice *>;

SliceMap make_slice_map(const std::vector<LayerSlice *>& slices) {
    SliceMap smap;
    smap.reserve(slices.size());
    for (auto it = slices.begin(); it != slices.end(); ++it) {
        LayerSlice *slice = *it;
        EXIT_ERROR_CHECK_EQ(nullptr, slice, "LayerSlice is nullptr");
        smap[slice->layer()] = slice;
    }
    return smap;
}

void fill_unit_io_from_part(ExecUnit& unit, const ExecutionPartition& part) {
    for (auto it = part.inputs().begin(); it != part.inputs().end(); ++it) {
        unit.addInput(*it);
    }
    for (auto it = part.outputs().begin(); it != part.outputs().end(); ++it) {
        unit.addOutput(*it);
    }
}

void append_partition_unit(ExecutionPlan& plan,
                           const ExecutionPartition& part,
                           const SliceMap& slices) {
    ExecUnit unit;
    unit.setType(ExecUnitType::EU_PARTITION);
    unit.setDomain(part.domain());
    unit.setPart(&part);
    fill_unit_io_from_part(unit, part);

    for (auto it = part.topo().begin(); it != part.topo().end(); ++it) {
        auto smap_it = slices.find(*it);
        EXIT_ERROR_CHECK_EQ(smap_it, slices.end(), "Partition layer slice not found");
        unit.addSlice(smap_it->second);
    }
    plan.addUnit(unit);
}

void append_layer_units(ExecutionPlan& plan,
                        const ExecutionPartition& part,
                        const SliceMap& slices) {
    for (auto it = part.topo().begin(); it != part.topo().end(); ++it) {
        auto smap_it = slices.find(*it);
        EXIT_ERROR_CHECK_EQ(smap_it, slices.end(), "Layer slice not found");

        ExecUnit unit;
        unit.setType(ExecUnitType::EU_LAYER);
        unit.setDomain(part.domain());
        unit.setPart(&part);
        unit.addSlice(smap_it->second);
        fill_unit_io_from_part(unit, part);
        plan.addUnit(unit);
    }
}

void execute_unit_local(const ExecUnit& unit, Executor *exec, ThreadCtx_t *ctx) {
    for (auto it = unit.slices().begin(); it != unit.slices().end(); ++it) {
        exec->execute_layer(*it, ctx);
    }
}

bool execute_unit_bridge(const ExecUnit& unit, Executor *exec, ThreadCtx_t *ctx) {
    // ExecutionPlan 里拿到一个 ExecUnit
    // 看它的 domain
    // 去 Executor 里找这个域对应的 bridge
    // 找到就交给它执行
    // 找不到就本地执行
    EXIT_ERROR_CHECK_EQ(nullptr, exec, "Executor is nullptr");
    ExecDomainBridge *bridge = exec->execBridge(unit.domain());
    if (nullptr == bridge) {
        return false;
    }
    if (bridge->domain() != unit.domain()) {
        return false;
    }
    return bridge->execute(unit, exec, ctx);
}

void execute_unit(const ExecUnit& unit, Executor *exec, ThreadCtx_t *ctx) {
    if (execute_unit_bridge(unit, exec, ctx)) {
        return;
    }
    execute_unit_local(unit, exec, ctx);
}

uint32_t data_type_to_proto(DataType dtype) {
    switch (dtype) {
    case DataType::FP16:
        return CONFINFER_DTYPE_FP16;
    case DataType::INT8:
        return CONFINFER_DTYPE_INT8;
    case DataType::INT32:
        return CONFINFER_DTYPE_INT32;
    case DataType::FP32:
    default:
        return CONFINFER_DTYPE_FP32;
    }
}

uint32_t data_location_to_proto(DataLocation location) {
    switch (location) {
    case DataLocation::TEE:
        return CONFINFER_DATA_TEE;
    case DataLocation::CPU:
    default:
        return CONFINFER_DATA_CPU;
    }
}

uint32_t param_role_to_proto(ParamRole role) {
    switch (role) {
    case ParamRole::WEIGHT:
        return CONFINFER_PARAM_ROLE_WEIGHT;
    case ParamRole::BIAS:
        return CONFINFER_PARAM_ROLE_BIAS;
    case ParamRole::RUNNING_MEAN:
        return CONFINFER_PARAM_ROLE_RUNNING_MEAN;
    case ParamRole::RUNNING_VAR:
        return CONFINFER_PARAM_ROLE_RUNNING_VAR;
    case ParamRole::UNKNOWN:
    default:
        return CONFINFER_PARAM_ROLE_UNKNOWN;
    }
}

bool is_tee_partition_unit(const ExecUnit& unit) {
    return unit.domain() == ExecutionDomain::ED_CPU_TEE &&
           unit.type() == ExecUnitType::EU_PARTITION;
}

ExecBridgeProto *tee_exec_bridge_proto(Executor *exec) {
    ExecDomainBridge *bridge = exec->execBridge(ExecutionDomain::ED_CPU_TEE);
    if (nullptr == bridge) {
        return nullptr;
    }
    return dynamic_cast<ExecBridgeProto *>(bridge);
}

UINT count_tee_partition_units(const ExecutionPlan& plan) {
    UINT count = 0;
    for (auto it = plan.units().begin(); it != plan.units().end(); ++it) {
        if (is_tee_partition_unit(*it)) {
            ++count;
        }
    }
    return count;
}

// 某一个 layer 的某一个参数槽位转成协议里的 confinfer_param_desc_t
// 并把它的原始字节追加到 blob 里
void append_param_desc(std::vector<confinfer_param_desc_t>& descs,
                       std::vector<uint8_t>& blob,
                       std::unordered_set<confinfer_param_id_t>& seen,
                       confinfer_partition_id_t partition_id,
                       const Layer *layer,
                       ParamRole role,
                       const Data_t *param) {
    confinfer_param_desc_t desc{};
    const uint32_t role_id = param_role_to_proto(role);
    const UINT param_id = layer ? layer->paramId(role) : INVALID_VALUE_U;
    const uint32_t byte_size = param->shape.size * param->getTypeSize();

    EXIT_ERROR_CHECK_EQ(nullptr, layer, "Layer is nullptr");
    EXIT_ERROR_CHECK_EQ(nullptr, param, "Param is nullptr");
    EXIT_ERROR_CHECK_EQ(nullptr, param->ptr, "Param data.ptr is nullptr");
    EXIT_ERROR_CHECK_EQ(INVALID_VALUE_U, param_id, "Layer param_id is invalid");

    // 同一个共享参数 可能被多个 layer 或多个 slice 间接访问到
    // 但 TA 里只应该加载一份常驻参数
    desc.param_id = static_cast<confinfer_param_id_t>(param_id);
    if (seen.find(desc.param_id) != seen.end()) {
        return;
    }

    desc.owner_layer_id = layer->id();
    // 参数槽位所属的分区身份也直接沿用 REE 侧 ExecutionPartition.id。
    desc.owner_partition_id = partition_id;
    desc.role = role_id;
    desc.dtype = data_type_to_proto(param->dtype);
    desc.location = data_location_to_proto(param->location);
    desc.flags = param->flags;
    desc.elem_count = param->shape.size;
    desc.byte_size = byte_size;
    desc.data_offset = static_cast<uint32_t>(blob.size());
    desc.ndim = param->shape.ndim;
    EXIT_ERROR_CHECK_EQ(true, desc.ndim > CONFINFER_VALUE_MAX_DIMS,
                        "Param ndim exceeds protocol max dims");
    for (uint32_t i = 0; i < desc.ndim; ++i) {
        desc.dims[i] = param->shape.dims[i];
    }

    const uint8_t *src = static_cast<const uint8_t *>(param->ptr);
    blob.insert(blob.end(), src, src + byte_size);
    descs.push_back(desc);
    seen.insert(desc.param_id);
}

// 把所有 TEE 分区会用到的常驻参数抽取出来 整理成：
// 参数描述表 param_descs
// 参数字节流 param_blob
void collect_tee_params(const ExecutionPlan& plan,
                        std::vector<confinfer_param_desc_t>& descs,
                        std::vector<uint8_t>& blob) {
    std::unordered_set<confinfer_param_id_t> seen;
    const ParamRole roles[] = {
        ParamRole::WEIGHT,
        ParamRole::BIAS,
        ParamRole::RUNNING_MEAN,
        ParamRole::RUNNING_VAR,
    };

    // 先遍历 exec_unit 中的 TEE 分区
    for (auto it = plan.units().begin(); it != plan.units().end(); ++it) {
        if (!is_tee_partition_unit(*it)) {
            continue;
        }
        const ExecutionPartition *part = it->part();
        EXIT_ERROR_CHECK_EQ(nullptr, part, "TEE ExecUnit partition is nullptr");
        // 遍历这个分区单元里的每个 LayerSlice 取出 slice->layer()
        for (auto sit = it->slices().begin(); sit != it->slices().end(); ++sit) {
            LayerSlice *slice = *sit;
            Layer *layer = slice ? slice->layer() : nullptr;
            EXIT_ERROR_CHECK_EQ(nullptr, layer, "TEE ExecUnit layer is nullptr");
            for (ParamRole role : roles) {
                // 对固定的参数角色集合逐个检查
                const Data_t *param = layer->param(role);
                if (nullptr == param) {
                    continue;
                }
                append_param_desc(descs, blob, seen,
                                  static_cast<confinfer_partition_id_t>(part->id()),
                                  layer, role, param);
            }
        }
    }
}

// prepare 需要做的事
// 1. 找出当前执行计划里哪些执行单元属于 TEE
// 2. 收集这些 TEE 单元涉及到的常驻参数
// 3. 在 TA 里注册一个模型上下文
// 4. 把参数加载到这个模型上下文里
// 5. 把每个 TEE 分区注册到这个模型上下文里
void prepare_tee_runtime(const Network *network, Executor *exec) {
    ExecBridgeProto *bridge = nullptr;
    const ExecutionPlan& plan = network->execPlan();
    // 统计当前执行计划中有多少个 TEE 分区执行单元
    const UINT tee_part_count = count_tee_partition_units(plan);
    // param 描述信息 和 buffer
    std::vector<confinfer_param_desc_t> param_descs;
    std::vector<uint8_t> param_blob;
    // 协议头
    confinfer_model_desc_t model_desc{};
    confinfer_load_params_req_t params_req{};

    EXIT_ERROR_CHECK_EQ(nullptr, network, "Network is nullptr");
    EXIT_ERROR_CHECK_EQ(nullptr, exec, "Executor is nullptr");
    if (0 == tee_part_count) {
        return;
    }

    bridge = tee_exec_bridge_proto(exec);
    EXIT_ERROR_CHECK_EQ(nullptr, bridge, "TEE ExecBridgeProto is not installed");
    EXIT_ERROR_CHECK_EQ(false, bridge->lifecycleReady(),
                        "TEE ExecBridgeProto lifecycle callbacks are not ready");

    // 当前的设计还是 为了最简化的考量 绑定 TEE 的使用场景
    collect_tee_params(plan, param_descs, param_blob);

    // 先构造 model_desc 再调用 registerModel
    // 后面 分别回调 bridge 中的 
    // _register_model 
    // _load_params 
    // _register_partition
    model_desc.version = CONFINFER_PROTOCOL_VERSION;
    model_desc.model_id = network->modelId();
    model_desc.flags = 0;
    model_desc.expected_partition_count = tee_part_count;
    model_desc.expected_param_count = static_cast<uint32_t>(param_descs.size());
    model_desc.reserved0 = 0;
    model_desc.reserved1 = 0;
    EXIT_ERROR_CHECK_EQ(false, bridge->registerModel(model_desc),
                        "TEE registerModel failed");

    params_req.version = CONFINFER_PROTOCOL_VERSION;
    params_req.model_id = network->modelId();
    params_req.param_count = static_cast<uint32_t>(param_descs.size());
    params_req.total_param_bytes = static_cast<uint32_t>(param_blob.size());
    params_req.flags = 0;
    params_req.reserved0 = 0;
    params_req.reserved1 = 0;
    EXIT_ERROR_CHECK_EQ(false,
                        bridge->loadParams(params_req,
                                           param_descs.empty() ? nullptr : param_descs.data(),
                                           static_cast<UINT>(param_descs.size()),
                                           param_blob.empty() ? nullptr : param_blob.data(),
                                           static_cast<UINT>(param_blob.size())),
                        "TEE loadParams failed");

    for (auto it = plan.units().begin(); it != plan.units().end(); ++it) {
        if (!is_tee_partition_unit(*it)) {
            continue;
        }
        EXIT_ERROR_CHECK_EQ(false,
                            bridge->registerPartition(*it, network->modelId()),
                            "TEE registerPartition failed");
    }
}

void teardown_tee_runtime(Network *network, Executor *exec, bool strict) {
    ExecBridgeProto *bridge = nullptr;
    confinfer_unload_model_req_t req{};
    confinfer_unload_model_rsp_t rsp{};
    bool ok = false;

    if (nullptr == network || nullptr == exec || !network->teeRuntimeRegistered()) {
        return;
    }

    bridge = tee_exec_bridge_proto(exec);
    if (nullptr == bridge || !bridge->lifecycleReady()) {
        if (strict) {
            EXIT_ERROR("TEE ExecBridgeProto is not ready for unload");
        }
        return;
    }

    req.version = CONFINFER_PROTOCOL_VERSION;
    req.model_id = network->modelId();
    req.flags = 0;
    req.reserved0 = 0;
    req.reserved1 = 0;

    ok = bridge->unloadModel(req, &rsp);
    if (strict) {
        EXIT_ERROR_CHECK_EQ(false, ok, "TEE unloadModel invoke failed");
        EXIT_ERROR_CHECK_NE(CONFINFER_PROTOCOL_VERSION, rsp.version,
                            "TEE unloadModel response version mismatch");
        EXIT_ERROR_CHECK_NE(network->modelId(), rsp.model_id,
                            "TEE unloadModel response model_id mismatch");
        EXIT_ERROR_CHECK_NE(CONFINFER_UNLOAD_MODEL_OK, rsp.status,
                            "TEE unloadModel remote failed");
    }
}

} // namespace

std::atomic<confinfer_model_id_t> Network::_modelCounter{1};

Executor::Executor(): _by_kind(), _preferred_kind(BackendKind::BK_CPU_REE), _exec_bridges() {
    static Backend_CPU_REE cpu_backend;
    static Backend_CPU_REE_REF cpu_ref_backend;
    // 再次强调一下 这里的 TEE 后端作为特殊占位符
    // 其存在逻辑 和 TEE 执行桥的存在逻辑是分开的
    static Backend_CPU_TEE cpu_tee_backend;
    setBackends({&cpu_backend, &cpu_ref_backend, &cpu_tee_backend});
#if ENABLE_TEE_BRIDGE
    {
        static Kernel::bridges::TeeExecBridge tee_exec_bridge;
        uint32_t err_origin = 0;
        EXIT_ERROR_CHECK_NE(true,
                            tee_exec_bridge.install(this, &err_origin),
                            "Failed to install default TEE execution bridge");
    }
#endif
}

Executor::~Executor() {
    _by_kind.clear();
    _exec_bridges.clear();
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

// 注册执行区域
void Executor::setExecBridge(ExecutionDomain domain, ExecDomainBridge *bridge) {
    EXIT_ERROR_CHECK_EQ(nullptr, bridge, "ExecDomainBridge is nullptr");
    EXIT_ERROR_CHECK_NE(domain, bridge->domain(), "ExecDomainBridge domain mismatch");
    _exec_bridges[static_cast<uint32_t>(domain)] = bridge;
}

void Executor::clearExecBridge(ExecutionDomain domain) {
    _exec_bridges.erase(static_cast<uint32_t>(domain));
}

ExecDomainBridge *Executor::execBridge(ExecutionDomain domain) const {
    auto it = _exec_bridges.find(static_cast<uint32_t>(domain));
    if (it == _exec_bridges.end()) {
        return nullptr;
    }
    return it->second;
}

bool Executor::supports(BackendKind kind) const {
    auto it = _by_kind.find(kind);
    return it != _by_kind.end() && !it->second.empty();
}

bool Executor::supports(ExecutionDomain domain) const {
    if (nullptr != execBridge(domain)) {
        return true;
    }
    const BackendKind kind = backend_kind_from_execution_domain(domain, _preferred_kind);
    if (supports(kind)) {
        return true;
    }
    if (domain == ExecutionDomain::ED_DEFAULT) {
        return supports(BackendKind::BK_CPU_REE);
    }
    return false;
}

void Executor::prepare_layer(LayerSlice* ls)
{
    EXIT_ERROR_CHECK_EQ(nullptr, ls, "LayerSlice is nullptr");
    Backend *backend = route(ls->layer()->flags());
    EXIT_ERROR_CHECK_EQ(nullptr, backend, "No available backend for layer");
    ls->setBackend(backend);
    backend->prepare(ls);
}
void Executor::execute_layer(LayerSlice* ls, ThreadCtx_t* ctx)
{
    EXIT_ERROR_CHECK_EQ(nullptr, ls, "LayerSlice is nullptr");
    Backend *backend = ls->backend();
    if (nullptr == backend) {
        prepare_layer(ls);
        backend = ls->backend();
    }
    backend->execute(ls, ctx);
    return;
}
Backend *Executor::route(uint32_t lf) {
    BackendKind kind = _preferred_kind;

    if (lf & LF_REQUIRE_TEE) {
        if (nullptr != execBridge(ExecutionDomain::ED_CPU_TEE)) {
            kind = BackendKind::BK_CPU_TEE;
        } else {
            kind = fallback_backend_kind_for_domain(ExecutionDomain::ED_CPU_TEE, _preferred_kind);
        }
    }
    auto it = _by_kind.find(kind);
    if (it != _by_kind.end() && !it->second.empty()) {
        // 简单策略: 取第一个后端
        return it->second.front();
    }
    if (kind == BackendKind::BK_CPU_TEE) {
        return nullptr;
    }
    // 如果没有找到首选后端，回退到默认 CPU
    auto cpu_it = _by_kind.find(BackendKind::BK_CPU_REE);
    if (cpu_it != _by_kind.end() && !cpu_it->second.empty()) {
        return cpu_it->second.front();
    }
    // EXIT_ERROR("backends error");
    return nullptr;
}

bool is_exec_domain_registered(ExecutionDomain domain) {
    Executor *exec = EXECUTOR;
    EXIT_ERROR_CHECK_EQ(nullptr, exec, "Executor singleton is nullptr");
    return exec->supports(domain);
}

Network::Network(Graph &graph):
    _fullGraph(&graph),
    _workspace(nullptr),
    _wsSize(graph.WorkspaceSize()),
    _modelId(_modelCounter.fetch_add(1, std::memory_order_relaxed)),
    _teeRuntimeRegistered(false),
    _nets{}, _netNum(1) 
{
    if (_wsSize) {
        _workspace = static_cast<void*>(new char[_wsSize]);
    }
}

Network::Network(Graph &graph, const ThreadContextManager *tcm):
    _fullGraph(&graph),
    _workspace(nullptr),
    _wsSize(graph.WorkspaceSize()),
    _modelId(_modelCounter.fetch_add(1, std::memory_order_relaxed)),
    _teeRuntimeRegistered(false),
    _nets{}, _netNum(tcm->size())
{
    if (_wsSize) {
        _workspace = static_cast<void*>(new char[_wsSize]);
    }
    // 这里只记录默认的线程数视图
    // 真正的切片构造统一延后到 prepare() 中完成
}
Network::~Network() {
    teardown_tee_runtime(this, EXECUTOR, false);
    _teeRuntimeRegistered = false;
    if (_workspace && _wsSize) { 
        delete[] static_cast<char*>(_workspace);
        _workspace = nullptr;
        _wsSize = 0;
    }
}

void Network::buildExecPartitions() {
    EXIT_ERROR_CHECK_EQ(nullptr, _fullGraph, "_fullGraph is nullptr");
    _execPartitions = _partBuilder.build(*_fullGraph);
}

void Network::buildPartGraph() {
    _partGraph.build(_execPartitions);
}

// 把前面分析得到的 ExecutionPartition[] 转成当前运行时真正要消费的 ExecutionPlan
// 如果 network 被切成多个 Net_t
// 那每个 Net_t 都要有自己的 execPlan
void Network::buildExecPlans() {
    for (UINT net_id = 0; net_id < _netNum; ++net_id) {
        Net_t &net = _nets[net_id];
        net.execPlan.clear();

        // ExecutionPartition 里面保存的是 Layer*
        // 但真正运行时执行的是 LayerSlice*
        const SliceMap slices = make_slice_map(net.sliceExecOrder);
        for (auto it = _execPartitions.begin(); it != _execPartitions.end(); ++it) {
            const ExecutionPartition& part = *it;
            if (ExecutionDomain::ED_CPU_TEE == part.domain()) {
                append_partition_unit(net.execPlan, part, slices);
            } else {
                append_layer_units(net.execPlan, part, slices);
            }
        }
    }
}

void Network::split(UINT netNum) {
    if (MAX_CORES_NUM < netNum) EXIT_ERROR("error split netNum = %u", netNum);
    unsigned int coreNum = getCoreCount();
    if (!coreNum || !netNum || coreNum < netNum) 
        EXIT_ERROR("error split netNum = %u : coreNum = %u", netNum, coreNum);

    // Layer 决定 如何 切 (how to shard)
    // Graph 决定 能不能 切 (is it legal) (维度是否支持等等)
    // Network 决定 切多少份 (how many shards)
    if (!_fullGraph->splittable(netNum)) 
        EXIT_ERROR("Error: vertical split unsupported (num = %u)", netNum);

    for (UINT sliceId = 0; sliceId < netNum; ++sliceId) {
        _nets[sliceId].clear(); 
        _nets[sliceId].sliceExecOrder = _fullGraph->getLayerSlices(sliceId, netNum);
    }
    return;
}


// Network 有 (Graph &graph, const ThreadContextManager *tcm) 版本的析构函数 
// 原则上 这里传入的 tcm 应该和 析构函数中用的同一个 
void Network::prepare(ThreadContextManager *tcm, Executor *exec) {
    EXIT_ERROR_CHECK_EQ(nullptr, tcm, "ThreadContextManager is nullptr");
    EXIT_ERROR_CHECK_EQ(nullptr, exec, "Executor is nullptr");

    if (_teeRuntimeRegistered) {
        teardown(exec);
    }

    _netNum = tcm->size();
    EXIT_ERROR_CHECK_EQ(0, _netNum, "_netNum == 0");
    split(_netNum);

    // std::vector<ExecutionPartition> build by PartitionBuilder
    buildExecPartitions();
    buildPartGraph();// PartitionGraph build
    // net_t 内部 ExecutionPlan 初始化 by sliceExecOrder
    buildExecPlans();
    // 这里针对 TEE 内的情况 开始建立 TEE 内的上下文
    // 但是 目前这个设计只考虑了 单核的情况
    prepare_tee_runtime(this, exec);
    _teeRuntimeRegistered = (count_tee_partition_units(execPlan()) > 0);

    // 为每个 Net_s 对应绑定到 ThreadCtx_s
    // 在启动线程之前做必要的初始化操作 
    for (UINT i = 0; i < _netNum; ++i) {
        _nets[i].ctx = tcm->ctx(i);
        tcm->ctx(i)->shared->workspace = _workspace;
        tcm->ctx(i)->shared->wsSize = _wsSize;
        for (auto it = _nets[i].sliceExecOrder.begin(); it != _nets[i].sliceExecOrder.end(); ++it) {
            // bind Backend 并执行 Executor::perpare
            exec->prepare_layer(*it);
        }
    }

    ThreadCtx_t *ctx = tcm->caller_ctx();
    ctx->shared->start_flag.store(true, std::memory_order_relaxed);
    ctx->shared->stop_flag.store(false, std::memory_order_relaxed);
    for (UINT i = 1; i < _netNum; ++i) {
        tcm->launch_workers(
            i,
            static_cast<void *>(&_nets[i]),
            [this](ThreadCtx_t &ctx, Executor *exec, void *Args) -> void
            { this->worker_loop(ctx, exec, Args); }
        );
    }
    // 检测通信
    tcm->broadcast_task(make_event(ThreadMsg::PING));
    tcm->wait_all_done();
    for (UINT i = 1; i < _netNum; ++i) {
        LogDebug("Worker[%u] thread_id=%zu started and responded.", 
                i, tcm->ctx(i)->thread_id);
    }
    LogDebug("All %u workers created and communication verified.", _netNum);
    return;
}

// 清空 TEE 内数据
void Network::teardown(Executor *exec) {
    EXIT_ERROR_CHECK_EQ(nullptr, exec, "Executor is nullptr");
    teardown_tee_runtime(this, exec, true);
    _teeRuntimeRegistered = false;
}

void Network::runNet(ThreadContextManager *tcm, Executor *exec) {
    Net_t &net = _nets[0];

    ThreadCtx_t &ctx = *(tcm->caller_ctx());
    SharedContext_t *shared = ctx.shared;
    unsigned total_threads = tcm->size();
    UINT curr = INVALID_UINT_MAX;

    // 目前还是单线程执行的版本 直接按顺序执行就好了
    // 至于 多线程并行的问题 其实也不是必须的 但是框架写都写了还是留着了
    if (1 == total_threads) {
        if (!net.execPlan.empty()) {
            for (auto it = net.execPlan.units().begin(); it != net.execPlan.units().end(); ++it) {
                execute_unit(*it, exec, &ctx);
            }
            return;
        }
        for (auto it = net.sliceExecOrder.begin(); it != net.sliceExecOrder.end(); ++it) {
            exec->execute_layer(*it, &ctx);
        }
        return;
    }

#if 0
    // 主线程进入默认是持有锁的状态
    // 下面的代码目前是 不再用了 但是先保留着 以免后续多线程版本的开发需要
    curr = 0;
    shared->mtx.unlock();
    for (auto it = net.sliceExecOrder.begin(); it != net.sliceExecOrder.end(); 
            ++it, ++curr) {
        
        {   
            std::unique_lock<std::mutex> lk(shared->mtx); 
            shared->finished_cnt.store(0, std::memory_order_relaxed);
            shared->current_layer.store(curr, std::memory_order_release);
        }
        shared->cv.notify_all();

        exec->execute_layer(*it, &ctx);
        if (shared->finished_cnt.fetch_add(1, std::memory_order_acq_rel)
                == total_threads - 1) {
            std::lock_guard<std::mutex> lk(shared->mtx);
            shared->cv.notify_all();
        }

        { 
            std::unique_lock<std::mutex> lk(shared->mtx); 
            shared->cv.wait(lk, 
                [&]() -> bool { return shared->finished_cnt.load(std::memory_order_acquire) == total_threads; }); 
        }
        // prev = curr;
    }
    shared->mtx.lock();
#endif
}

void Network::run(std::initializer_list<Value_t*> inputs,
                  std::initializer_list<Value_t*> outputs,
                  ThreadContextManager *tcm, Executor *exec) {
    std::vector<Value_t*> input_vec(inputs.begin(), inputs.end());
    std::vector<Value_t*> output_vec(outputs.begin(), outputs.end());
    run(input_vec, output_vec, tcm, exec);
}

void Network::run(const std::vector<Value_t*>& inputs, std::vector<Value_t*>& outputs,
                  ThreadContextManager *tcm, Executor *exec) {
    const GraphSignature& sig = _fullGraph->signature();
    EXIT_ERROR_CHECK_NE(inputs.size(), sig.inputs.size(), "Network input size mismatch");
    EXIT_ERROR_CHECK_NE(outputs.size(), sig.outputs.size(), "Network output size mismatch");

    Layer* in_boundary = _fullGraph->inputBoundary();
    EXIT_ERROR_CHECK_EQ(nullptr, in_boundary, "Graph input boundary is nullptr");

    // 这里还是重新把 输入数据的一些 属性信息复制了一遍
    // 但是理论上目前 基本上输入的结构大小等等都是固定的
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
        // 这里用的是默认的 PARAM_NONE 不拥有数据
        v.borrowFrom(*inputs[i], PARAM_INPUT);
    }

    runNet(tcm, exec);

    Layer* out_boundary = _fullGraph->outputBoundary();
    EXIT_ERROR_CHECK_EQ(nullptr, out_boundary, "Graph output boundary is nullptr");
    GraphOutputLayer* out_layer = static_cast<GraphOutputLayer*>(out_boundary);

    // 目前这里的 output 都还默认保持 深拷贝
    for (UINT i = 0; i < sig.outputs.size(); ++i) {
        EXIT_ERROR_CHECK_EQ(nullptr, outputs[i], "Network output value is nullptr");
        Value_t& value = out_layer->input(i);
        outputs[i]->deepCopyFrom(value, PARAM_OUTPUT);
    }
}


/*
 * 对于运行模式的设计原理
 * 首先处于 threadsCtx 和 Network 语义上解耦的考量 我并不希望 threadsCtx 直接管理 Net_t
 * 我更加希望 Net_t 是由 Network 在 run() 时一次性注入的执行视图
 *
 * 这里有个确定的前提 
 *      : 每个 worker 在整个 run() 生命周期内 应该参与同一个 Net 的 layer 执行
 */
void Network::worker_loop(ThreadCtx_t &ctx, Executor *exec, void *Args) {
    Net_t &net = *static_cast<Net_t *>(Args);
    LayerSlice *ls = nullptr;

    SharedContext_t *shared = ctx.shared;
    unsigned total_threads = RUNTIME->size();
    UINT prev = INVALID_UINT_MAX;
    UINT curr = INVALID_UINT_MAX;
    
    if (shared->start_flag.load(std::memory_order_relaxed)) {
        ctx.read();
        ctx.write(make_event(ThreadMsg::PONG));
    }

    while (shared->start_flag.load(std::memory_order_relaxed))
    {
        // 等待主线程发布新的 Layer idx
        // shared->cv.wait(lk, predicate)
        // 检查条件: 先执行 predicate 如果返回 true 直接继续 不阻塞
        // 阻塞等待: 返回 false 线程会阻塞 并自动释放 lk 持有的锁 
        // 被唤醒后: 当主线程调用 notify_one() 或 notify_all() 时 wait 会重新加锁再次检查 predicate
        {
            std::unique_lock<std::mutex> lk(shared->mtx);
            shared->cv.wait(lk, [&]() -> bool {
                return shared->current_layer.load(std::memory_order_acquire) != prev
                    || shared->stop_flag.load(std::memory_order_relaxed);
            });
        }
        // 主线程在结束时至少需要 
        // shared->stop_flag.store(true, std::memory_order_relaxed);
        // shared->cv.notify_all();
        if (shared->stop_flag.load(std::memory_order_relaxed)) break;

        curr = shared->current_layer.load(std::memory_order_acquire);
        ls = net.sliceExecOrder.at(curr);
        exec->execute_layer(ls, &ctx);

        // 更新 prev 避免重复执行
        prev = curr;
        if (shared->finished_cnt.fetch_add(1, std::memory_order_acq_rel) 
                == total_threads -1) {
            std::lock_guard<std::mutex> lk(shared->mtx);
            shared->cv.notify_all();
        }
    }
}


} // namespace end of core
} // namespace end of Kernel 
