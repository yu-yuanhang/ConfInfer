#ifndef __NETWORK_H_CA__
#define __NETWORK_H_CA__

#include <All.h>
#include <core/Graph.h>
#include <generic/utils.h>
#include <core/threads.h>
#include <backend/backend.h>

using namespace Kernel::backend;

namespace Kernel {
namespace core {

// Executor 对线程不可见 仅仅负责 
// 执行哪个 Layer 使用哪个 Backend 需要哪些 runtime context
// 不在 Executor 里 loop topo_order
using BackendList = std::vector<Backend *>;
class Executor {
friend class Singleton<Executor>; 
private:
    Executor();
    ~Executor();
public:
    void setBackends(std::vector<Backend *> backends);
    void execute_layer(LayerSlice* ls, ThreadCtx_t* ctx);
    Backend *route(uint32_t lf);
private:
    std::unordered_map<BackendKind, BackendList> _by_kind;
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

    // 每 Net 私有 workspace (这里目前用不到)
    void* workspace;
    UINT  workspace_size;

    // 从设计上思考 TEE 内模型初次运行 
    // 模型初始化 和 运行流水式运行依然有些困难 (原理上可行 实现上困难)
    // struct Net_s *next;     // 用于 pipeline

    Net_s():
        id(INVALID_VALUE_U),
        ctx(nullptr),
        sliceExecOrder(),
        workspace(nullptr), workspace_size(INVALID_VALUE_U) {}
    ~Net_s() {
        // ...... 关于 execOrder 内 Layer * 所有权生命周期的问题
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

    void split(UINT netNum);
    // Network 基于指定的上下文运行环境 设置内部的计算图和子网络
    void prepare(ThreadContextManager *tcm = RUNTIME, Executor *exec = EXECUTOR);
    void run(Value_t &value, ThreadContextManager *tcm = RUNTIME, Executor *exec = EXECUTOR);
    // void print() const;

private:
    void worker_loop(ThreadCtx_t &ctx, 
            Executor *exec = EXECUTOR, void *Args = nullptr);
private:
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
    // 这里有一个关于 数据类型的问题
    // 为了实现不同 Layer 之间可能的不同数据精度的统一 
    // _wsSize 单位 1 字节 (包括子网络)
    void    *_workspace;
    UINT    _wsSize;    

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