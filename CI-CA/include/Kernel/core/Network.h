#ifndef __NETWORK_H_CA__
#define __NETWORK_H_CA__

#include <All.h>
#include <confinfer_protocol.h>
#include <generic/utils.h>
#include <core/Graph.h>
#include <core/PartitionGraph.h>
#include <backend/backend.h>

using namespace Kernel::backend;

namespace Kernel {
namespace core {

class Executor;

// 数据可用但不持有
typedef struct ExecContext_s {
    void *workspace;
    UINT wsSize;
    confinfer_model_id_t modelId;
    const std::vector<ExecPartition> *parts;

    ExecContext_s()
        : workspace(nullptr),
          wsSize(0),
          modelId(CONFINFER_INVALID_MODEL_ID),
          parts(nullptr) {}
} ExecContext_t;

// Executor 对线程不可见 仅仅负责 
// 执行哪个 Layer 使用哪个 Backend 需要哪些 runtime context
// 
// Executor 负责拿到一个 Layer 或分区 决定交给哪个 backend
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
    Backend *backend(BackendKind kind) const;
    bool supports(BackendKind kind) const;
    bool supports(ExecutionDomain domain) const;

    void prepare_layer(Layer* layer);
    void execute_layer(Layer* layer, ExecContext_t* ctx);
    void prepare_partition(ExecPartition& part, ExecContext_t* ctx);
    void execute_partition(ExecPartition& part, ExecContext_t* ctx);
    void reset_runtime(ExecContext_t* ctx, bool strict);
    Backend *route(uint32_t lf);
    Backend *route(const ExecPartition& part);
private:
    // 这里的注册的后端都是本地的
    std::unordered_map<BackendKind, BackendList> _by_kind;
    // 标识本地当前使用的 backend 类型 默认的是 BK_CPU_REE
    BackendKind _preferred_kind;
};
#define EXECUTOR (Singleton<Executor>::getInstance())

class Network {
public:
    Network(Graph &graph);
    ~Network();
    const PartitionGraph& partGraph() const { return _partGraph; }
    // 这是 Network 作为“模型执行对象”的全局唯一身份，不是 TEE 专用字段。
    // 当前 TEE 协议会直接复用它作为 protocol model_id，后续其他外部执行域也应复用同一身份。
    confinfer_model_id_t modelId() const { return _modelId; }
    const PartitionBuildOptions& partOpts() const { return _partGraph.opts(); }
    void setPartOpts(const PartitionBuildOptions& opts) { _partGraph.setOpts(opts); }

    // Network 基于指定的上下文运行环境 设置内部的执行视图
    void prepare(Executor *exec = EXECUTOR);
    void teardown(Executor *exec = EXECUTOR);
    void run(std::initializer_list<Value_t*> inputs,
             std::initializer_list<Value_t*> outputs,
             Executor *exec = EXECUTOR);
    void run(const std::vector<Value_t*>& inputs, std::vector<Value_t*>& outputs,
             Executor *exec = EXECUTOR);
    // void print() const;

private:
    void buildPartGraph();
    void runNet(Executor *exec = EXECUTOR);
private:
    // 框架级模型对象 id 生成器
    // 其实目前用不太到
    static std::atomic<confinfer_model_id_t> _modelCounter;
/*
 * Network              ---> [模型语义]
 * Executor             ---> [执行语义]
 * ExecContext          ---> [执行上下文]
 * Backend              ---> [算子实现]
 * 处于上述的设计初衷 这里的 Network
 * 拥有 _fullGraph
 * 持有分区后的执行图
 * 决定执行路径
 */
    Graph   *_fullGraph;   // 完整图 (用于分析/切分)
    PartitionGraph _partGraph;
    // 当前执行阶段只保留一份统一的运行时上下文
    // 目前里面承载的核心内容就是 Network 级共享 workspace
    // ...... todo 
    // 但是这个逻辑后续还是 需要 优化的 
    // 需要基于 不同 ExecutionDomain 有不同的 ExecContext
    // 这个 _execCtx 应该仅仅对应 本地 ExecutionDomain 
    ExecContext_t _execCtx;

    // uint32_t
    confinfer_model_id_t _modelId;
};

// NETWORK 不应该存在单例限制
// #define NETWORK (Singleton<Network>::getInstance())

} // namespace end of core
} // namespace end of Kernel 

#endif
