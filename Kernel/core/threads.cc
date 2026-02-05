#include <core/threads.h>
#include <core/Network.h>

namespace Kernel {

ThreadContextManager::ThreadContextManager(UINT num)
    : _num(num), _shared(), _ctxs(), _workers() 
{
    _ctxs.resize(_num);
    setCallerCtx();
}
ThreadContextManager::~ThreadContextManager() { clear(); }
void ThreadContextManager::setThreadsNum(UINT num) {
    if (MAX_CORES_NUM < num) EXIT_ERROR("error threads num = %u", num);
    unsigned int coreNum = getCoreCount();
    if (!coreNum || !num || coreNum < num) 
        EXIT_ERROR("error split num = %u : coreNum = ", num, coreNum);

    // 对于重置的情况需要进行 clean
    clear();

    _num = num;
    _ctxs.resize(_num);
    for (UINT i = 0; i < _num; ++i) { 
        _ctxs.at(i).id = i;
        _ctxs.at(i).shared = &_shared; 
    }
    setCallerCtx();
    // 设计上 这里还没有获取 执行器 Executor 因此还无需启动线程
}

void ThreadContextManager::setCallerCtx() {
    if (_ctxs.empty()) EXIT_ERROR("_ctxs is empty");

    ThreadCtx_t &caller = _ctxs.at(0);
    // 获取当前线程的真实 ID
    caller.thread_id = std::this_thread::get_id();
    caller.is_caller_thread = true;
    caller.is_worker_thread = true;
    caller.worker = nullptr; // 调用者线程不是 std::thread 创建的

    return;
}
void ThreadContextManager::clear() {
    join_all();
    _shared.clear();
    _ctxs.clear();
    return;
}
void ThreadContextManager::join_all() {
    _shared.start_flag.store(false, std::memory_order_relaxed);
    _shared.stop_flag.store(true, std::memory_order_relaxed);
    _shared.cv.notify_all();
    _shared.mtx.unlock();
    // broadcast_task(make_event(ThreadMsg::EXIT));
    for (auto it = _workers.begin(); it != _workers.end(); ++it) {
        if ((*it).joinable()) { (*it).join(); }
    }
    _workers.clear();
    _shared.mtx.lock();
}

void ThreadContextManager::launch_workers(UINT id, void *Args,
    std::function<void(ThreadCtx_t &, Executor *, void *)> entry
) {
    // 启动 worker 线程 (不包含 caller)
    EXIT_ERROR_CHECK_EQ(0, id, "Start worker threads (excluding the caller)");
    _workers.emplace_back([this, id, Args, entry]() -> void {
        ThreadCtx_t& ctx = _ctxs.at(id);
        ctx.thread_id = std::this_thread::get_id();
        ctx.is_worker_thread = true;
        LogDebug("launch_workers: id=%u, thread_id=%zu",
                ctx.id,
                std::hash<std::thread::id>{}(ctx.thread_id));
        entry(ctx, EXECUTOR, Args);
    });
    _ctxs[id].worker = &_workers.back();    // 在主线程里保存指针
    return;
}
ThreadCtx_t *ThreadContextManager::ctx(UINT id) {
    if (_ctxs.empty()) return nullptr;
    return &(_ctxs.at(id));
}
ThreadCtx_t *ThreadContextManager::caller_ctx() {
    return ctx(0);
}
void ThreadContextManager::broadcast_task(EventPayload ev) {
    // EventPayload ev = make_event(ThreadMsg::PING);
    for (UINT i = 1; i < _num; ++i) {
        ThreadCtx_t &ctx = _ctxs[i];
        if (!ctx.is_worker_thread) continue;

        write(ctx.task_event_fd, &ev, sizeof(ev));
    }
}
void ThreadContextManager::wait_all_done()
{
    EventPayload sig = 0;
    for (UINT i = 1; i < _num; ++i) {
        ThreadCtx_t &ctx = _ctxs[i];
        if (!ctx.is_worker_thread) continue;

        read(ctx.done_event_fd, &sig, sizeof(sig));
        // ...... 检查返回信息 (这里目前也不考虑)
    }
}

} // namespace end of Kernel 