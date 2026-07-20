#ifndef __NETWORK_H_CA__
#define __NETWORK_H_CA__

#include <All.h>
#include <core/ExecUnitProto.h>
#include <core/ExecutionPlan.h>
#include <core/ExecutionPartition.h>
#include <core/Graph.h>
#include <core/PartitionBuilder.h>
#include <core/PartitionGraph.h>
#include <generic/utils.h>
#include <core/threads.h>
#include <backend/backend.h>

using namespace Kernel::backend;

namespace Kernel {
namespace core {

class Executor;

// 用于承接“非当前进程内直接逐层执行”的执行域调用。
// 当前最主要的场景是 TEE 分区调用 后续也可以扩展到其他外部执行域。
// 用来提供具体的执行接口 为了通用化设计 不同执行区域之间 Bridge 应该有一个共同的抽象约束
// 
// ExecDomainBridge 规定了所有 bridge 至少要能回答两件事
// 服务于哪个执行域
// 获得一个 ExecUnit 怎么执行
// ExecBridgeProto 继承自 ExecDomainBridge 因此
// ExecBridgeProto 才是真正注册到 Executor 里的 bridge 对象
class ExecDomainBridge {
public:
    virtual ~ExecDomainBridge() = default;
    // 说明这个 bridge 负责哪个执行区域
    virtual ExecutionDomain domain() const = 0;
    virtual bool execute(const ExecUnit& unit, Executor *exec, ThreadCtx_t *ctx) = 0;
};

// Executor 对线程不可见 仅仅负责 
// 执行哪个 Layer 使用哪个 Backend 需要哪些 runtime context
// 
// Executor 负责拿到一个 ExecUnit 决定交给本地 backend 还是交给 bridge
using BackendList = std::vector<Backend *>;
// Executor 设计为一种通用的执行器 
// 本地执行 跨域执行 对上衔接 Network 都统一通过这个就行
class Executor {
friend class Singleton<Executor>; 
private:
    Executor();
    ~Executor();
public:
    // setBackends 用来注册 哪些 后端能用 
    void setBackends(std::vector<Backend *> backends);
    void setBackendKind(BackendKind kind);
    BackendKind backendKind() const { return _preferred_kind; }
    // 把某个执行域对应的 bridge 注册进去 / 或是删除
    void setExecBridge(ExecutionDomain domain, ExecDomainBridge *bridge);
    void clearExecBridge(ExecutionDomain domain);
    // 按执行域在 _exec_bridges 中查 bridge 找不到返回 nullptr
    ExecDomainBridge *execBridge(ExecutionDomain domain) const;
    bool supports(BackendKind kind) const;
    bool supports(ExecutionDomain domain) const;

    void prepare_layer(LayerSlice* ls);
    void execute_layer(LayerSlice* ls, ThreadCtx_t* ctx);
    Backend *route(uint32_t lf);
private:
    // 这里的注册的后端都是本地的
    std::unordered_map<BackendKind, BackendList> _by_kind;
    // 标识本地当前使用的 backend 类型 默认的是 BK_CPU_REE
    BackendKind _preferred_kind;
    // 执行区域到 bridge 的映射表 
    // ED_CPU_TEE -> 某个 ExecDomainBridge*
    std::unordered_map<uint32_t, ExecDomainBridge *> _exec_bridges;
};
#define EXECUTOR (Singleton<Executor>::getInstance())

// Net 用于表示子网络 一个 Net 绑定对应一个 threadCtx
// INT id == 0 保留给首个 子网络
typedef struct Net_s {
    UINT    id; // 这里暂时用不到
    ThreadCtx_t *ctx;
    // Graph   *graph;
    // 我认为对于 Net 自网络来说无需再感知完整的计算图结构 
    // 其次 Nets 也不该有独立的 新计算图或是网络 (为了设计上的内存安全和简化 用偏移的方式) 
    // 但是 LayerSlice 的生命周期是由 Net_s 管理
    std::vector<LayerSlice *> sliceExecOrder;
    ExecutionPlan execPlan;

    // 每 Net 私有 workspace
    // 当前阶段暂不参与主执行路径，保留给后续分片 / pipeline / 私有缓存。
    void* workspace;
    UINT  workspace_size;

    // 从设计上思考 TEE 内模型初次运行 
    // 模型初始化 和 运行流水式运行依然有些困难 (原理上可行 实现上困难)
    // struct Net_s *next;     // 用于 pipeline

    Net_s():
        id(INVALID_VALUE_U),
        ctx(nullptr),
        sliceExecOrder(),
        execPlan(),
        workspace(nullptr), workspace_size(INVALID_VALUE_U) {}
    ~Net_s() {
        clear();
    }
    void clear() {
        // Network::split 调用内可以调用 clean 清理确保内存安全
        // 用于清理 ctx sliceExecOrder 
        if (workspace && workspace_size) { 
            delete[] static_cast<char*>(workspace);
            workspace = nullptr;
            workspace_size = 0;
        }
        // clear() 是服务于 Network
        for (auto it = sliceExecOrder.begin(); it != sliceExecOrder.end(); ++it) 
        { delete (*it); }
        sliceExecOrder.clear();
        execPlan.clear();
    }
} Net_t;
 
// 在设计逻辑上 Network/Net 作为网络推理执行的基本单位
// Net_t 用于与线程环境绑定 以支持模型 (或者说是计算图) 语义上的切分
// 一个 Metwork 对应 一个 _fullGraph
// 因此 :
//      单个 Network 内部并行化语义应该是统一的 (这样在逻辑上便于管理和理解)
//      单个 Network 内部 Value 的位置语义不需要相等 (太过笨重)
class Network {
public:
    Network(Graph &graph);
    Network(Graph &graph, const ThreadContextManager *tcm = RUNTIME);
    ~Network();
    const std::vector<ExecutionPartition>& execPartitions() const { return _execPartitions; }
    const PartitionGraph& partGraph() const { return _partGraph; }
    const ExecutionPlan& execPlan(UINT netId = 0) const { return _nets[netId].execPlan; }
    // 这是 Network 作为“模型执行对象”的全局唯一身份，不是 TEE 专用字段。
    // 当前 TEE 协议会直接复用它作为 protocol model_id，后续其他外部执行域也应复用同一身份。
    confinfer_model_id_t modelId() const { return _modelId; }
    bool teeRuntimeRegistered() const { return _teeRuntimeRegistered; }
    const PartitionBuildOptions& partOpts() const { return _partBuilder.opts(); }
    void setPartOpts(const PartitionBuildOptions& opts) { _partBuilder.setOpts(opts); }

    void split(UINT netNum);
    // Network 基于指定的上下文运行环境 设置内部的计算图和子网络
    void prepare(ThreadContextManager *tcm = RUNTIME, Executor *exec = EXECUTOR);
    void teardown(Executor *exec = EXECUTOR);
    void run(std::initializer_list<Value_t*> inputs,
             std::initializer_list<Value_t*> outputs,
             ThreadContextManager *tcm = RUNTIME, Executor *exec = EXECUTOR);
    void run(const std::vector<Value_t*>& inputs, std::vector<Value_t*>& outputs,
             ThreadContextManager *tcm = RUNTIME, Executor *exec = EXECUTOR);
    // void print() const;

private:
    void buildExecPartitions();
    void buildPartGraph();
    void buildExecPlans();
    void runNet(ThreadContextManager *tcm = RUNTIME, Executor *exec = EXECUTOR);
    void worker_loop(ThreadCtx_t &ctx, 
            Executor *exec = EXECUTOR, void *Args = nullptr);
private:
    // 框架级模型对象 id 生成器
    // 其实目前用不太到
    static std::atomic<confinfer_model_id_t> _modelCounter;
/*
 * Network              ---> [模型语义]
 * Executor             ---> [执行语义]
 * ThreadContextManager ---> [并发资源]
 * Backend              ---> [算子实现]
 * 处于上述的设计初衷 这里的 NetWork 
 * 拥有 _fullGraph 
 * 决定 是否切分 / 如何切分
 * 持有 Net (逻辑子图)
 * 决定 执行策略 (单 Net / 多 Net)
 */
    Graph   *_fullGraph;   // 完整图 (用于分析/切分)
    // Network 级共享 scratch workspace
    // 当前阶段默认单线程顺序执行，不做切片，任一时刻只服务当前执行层。
    // _wsSize 单位 / B
    void    *_workspace;
    UINT    _wsSize;    
    PartitionBuilder _partBuilder;
    std::vector<ExecutionPartition> _execPartitions;
    // 存储分区的图结构 依赖关系
    PartitionGraph _partGraph;
    // uint32_t
    confinfer_model_id_t _modelId;
    bool _teeRuntimeRegistered;

    Net_t   _nets[MAX_CORES_NUM];
    UINT    _netNum;   // 1: 不切分; >1: 统一纵向切分

    // network 视角下不应该有 Executor / ThreadPool / backend 的概念
    // 但是计算图语义上的 切割/优化 应该是可见的 所以持有 _nets
    // Executor    *_executor;
};

// NETWORK 不应该存在单例限制
// #define NETWORK (Singleton<Network>::getInstance())

} // namespace end of core
} // namespace end of Kernel 

#endif
