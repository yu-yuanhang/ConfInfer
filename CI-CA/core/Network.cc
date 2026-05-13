#include <core/Network.h>
#include <core/BoundaryLayer.h>

namespace Kernel {
namespace core {

Executor::Executor(): _by_kind() {
    static CpuBackend cpu_backend;
    setBackends({&cpu_backend});
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
void Executor::execute_layer(LayerSlice* ls, ThreadCtx_t* ctx)
{
    EXIT_ERROR_CHECK_EQ(nullptr, ls, "LayerSlice is nullptr");
    uint32_t lf = ls->layer()->flags();
    Backend *backend = route(lf);
    EXIT_ERROR_CHECK_EQ(nullptr, backend, "No available backend for layer");
    backend->execute(ls, ctx);
    return;
}
Backend *Executor::route(uint32_t lf) {
    BackendKind kind = BackendKind::CPU_REE; // 默认回退 CPU

    if (lf & LF_REQUIRE_TEE) { kind = BackendKind::CPU_TEE; }
    auto it = _by_kind.find(kind);
    if (it != _by_kind.end() && !it->second.empty()) {
        // 简单策略: 取第一个后端
        return it->second.front();
    }
    // 如果没有找到合适的后端，回退到 CPU
    auto cpu_it = _by_kind.find(BackendKind::CPU_REE);
    if (cpu_it != _by_kind.end() && !cpu_it->second.empty()) {
        return cpu_it->second.front();
    }
    // EXIT_ERROR("backends error");
    return nullptr;
}

Network::Network(Graph &graph):
    _fullGraph(&graph),
    _workspace(nullptr),
    _wsSize(graph.WorkspaceSize()),
    _nets{}, _netNum(1) 
{
    if (_wsSize) {
        _workspace = static_cast<void*>(new char[_wsSize]);
    }
    split(_netNum);
}

Network::Network(Graph &graph, const ThreadContextManager *tcm):
    _fullGraph(&graph),
    _workspace(nullptr),
    _wsSize(graph.WorkspaceSize()),
    _nets{}, _netNum(tcm->size())
{
    if (_wsSize) {
        _workspace = static_cast<void*>(new char[_wsSize]);
    }
    // 这个版本的 构造函数 在这里绑定 ThreadContextManager
    // 因此可以提前初始化 net[MAX_CORES_NUM] 
    // _netNum = tcm->size();
    split(_netNum);

    // 绑定 Net 与 threadCtx
}
Network::~Network() {
    if (_workspace && _wsSize) { 
        delete[] static_cast<char*>(_workspace);
        _workspace = nullptr;
        _wsSize = 0;
    }
}

void Network::split(UINT netNum) {
    if (MAX_CORES_NUM < netNum) EXIT_ERROR("error split netNum = %u", netNum);
    unsigned int coreNum = getCoreCount();
    if (!coreNum || !netNum || coreNum < netNum) 
        EXIT_ERROR("error split netNum = %u : coreNum = ", netNum, coreNum);

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
    EXIT_ERROR_CHECK_EQ(0, _netNum, "_netNum == 0");

    // 为每个 Net_s 对应绑定到 ThreadCtx_s
    // 在启动线程之前做必要的初始化操作 
    for (UINT i = 0; i < _netNum; ++i) {
        _nets[i].ctx = tcm->ctx(i);
        tcm->ctx(i)->shared->workspace = _workspace;
        tcm->ctx(i)->shared->wsSize = _wsSize;
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

void Network::runNet(ThreadContextManager *tcm, Executor *exec) {
    Net_t &net = _nets[0];

    ThreadCtx_t &ctx = *(tcm->caller_ctx());
    SharedContext_t *shared = ctx.shared;
    unsigned total_threads = tcm->size();
    UINT curr = INVALID_UINT_MAX;

    // 主线程进入默认是持有锁的状态
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
        Value_t& v = in_boundary->output(i);
        v.borrowFrom(*inputs[i], PARAM_INPUT);
    }

    runNet(tcm, exec);

    Layer* out_boundary = _fullGraph->outputBoundary();
    EXIT_ERROR_CHECK_EQ(nullptr, out_boundary, "Graph output boundary is nullptr");
    GraphOutputLayer* out_layer = static_cast<GraphOutputLayer*>(out_boundary);

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
