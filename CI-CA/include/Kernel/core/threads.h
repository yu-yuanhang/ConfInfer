#ifndef __THREADS_H_CA__
#define __THREADS_H_CA__

#include <thread>
#include <mutex>
#include <functional>
#include <condition_variable>
#include <All.h>
#include <generic/utils.h>

// =================== POSIX/Linux ===================
#include <sys/eventfd.h>
#include <unistd.h>
// ===================================================

namespace Kernel {
namespace core {
class Executor;
struct Net_s;
} // namespace end of core

using core::Executor;
using core::Net_s;

// 线程间控制消息类型
enum class ThreadMsg : uint8_t {
    NONE = 0,
    RUN_TASK,  
    EXIT,   
    SYNC_POINT,  
    // ......

    PING,
    PONG,
};
/*
 * 高 8 bit  : ThreadMsg
 * 低 56 bit : 参数 / 序号 / 保留   (这里目前还用不到)
 */
typedef uint64_t EventPayload;
inline EventPayload make_event(ThreadMsg msg, uint64_t param = 0)
{
    return (static_cast<uint64_t>(msg) << 56) | (param & 0x00FFFFFFFFFFFFFFULL);
}

inline ThreadMsg event_msg(EventPayload v)
{
    return static_cast<ThreadMsg>(v >> 56);
}

inline uint64_t event_param(EventPayload v)
{
    return v & 0x00FFFFFFFFFFFFFFULL;
}

// 多 Net 共享数据 Network 生命周期内唯一
typedef struct SharedContext_s {

    std::atomic<bool> start_flag;   // 主线程置 true worker 
    std::atomic<bool> stop_flag;    // 子线程终止 (直接跳出 不在等待)

    // memory_order_relaxed 只保证这个原子操作本身是原子的
    // memory_order_release / memory_order_acquire
    // store    release 保证在这次原子写入之前的所有普通读写操作 对其他线程来说都"先于"这次写入完成
    // load     acquire 保证在这次原子读取之后的所有普通读写操作 都"后于"这次读取执行
    // memory_order_acq_rel  (同时具备 acquire 和 release 的语义)
    // memory_order_seq_cst 
    std::atomic<UINT> finished_cnt;     // 当前 Layer 已完成的线程数
    std::atomic<UINT> current_layer;    // 当前要执行的 Layer idx

    // 从子线程创建起
    // 除了主线程调用子线程 run 以外的任何时候 主线程都持有该锁
    std::mutex mtx; // std::lock_guard<std::mutex> lk(mtx);
    std::condition_variable cv;

    // 不管理其生命周期
    void    *workspace;
    UINT    wsSize;

    SharedContext_s():
        start_flag(false),
        stop_flag(true),
        finished_cnt(INVALID_VALUE_U),
        current_layer(INVALID_UINT_MAX),
        mtx(), cv(),
        workspace(nullptr), wsSize(INVALID_VALUE_U) {
            mtx.lock();
        }
    ~SharedContext_s() {
        mtx.unlock();
    }
    void clear() {
        // 原子变量恢复初始状态
        start_flag.store(false, std::memory_order_relaxed);
        stop_flag.store(true, std::memory_order_relaxed);
        finished_cnt.store(INVALID_VALUE_U, std::memory_order_relaxed);
        current_layer.store(INVALID_UINT_MAX, std::memory_order_relaxed);

        // 不管理 workspace 生命周期 workspace 释放并置空
        // if (workspace) {
        //     delete[] static_cast<char*>(workspace);
        //     workspace = nullptr;
        //     wsSize = INVALID_VALUE_U;
        // }
        // 注意: std::mutex 和 std::condition_variable 本身没有 clear 方法
        // 只能通过重新构造来保证干净状态
        // 但是这里这里无需处理 mtx
    }
} SharedContext_t;

// ThreadCtx 我认为还是需要一个独立的结构来表示线程环境
typedef struct ThreadCtx_s {
    UINT id;
    std::thread::id thread_id;
    std::thread *worker;
    bool is_caller_thread;  // 是否为调用线程
    bool is_worker_thread;  // 是否有真实 worker

    int task_event_fd;   // 主线程 -> worker: 发布任务
    int done_event_fd; 

    // 线程私有资源 目前作为保留 
    void *tls_base;
    UINT tls_size;


    // std::atomic<bool> running;
    // std::atomic<bool> finished;
    // 指向 Network 级共享上下文
    // 不实际拥有其生命周期
    SharedContext_t *shared;

    ThreadCtx_s(): 
        id(INVALID_VALUE_U),
        thread_id(), 
        worker(nullptr), 
        is_caller_thread(false), is_worker_thread(false), 
        task_event_fd(-1), done_event_fd(-1),
        tls_base(nullptr), tls_size(INVALID_VALUE_U), 
        shared(nullptr) {
            task_event_fd = eventfd(0, EFD_CLOEXEC);
            done_event_fd = eventfd(0, EFD_CLOEXEC);
        }
    ~ThreadCtx_s() {
        if (tls_base) { delete static_cast<char*>(tls_base);tls_base = nullptr; }
        if (task_event_fd >= 0) close(task_event_fd);
        if (done_event_fd >= 0) close(done_event_fd);
    }
    EventPayload read() {
        EventPayload sig = 0;
        ::read(task_event_fd, &sig, sizeof(sig));
        return sig;
    }
    void write(EventPayload sig) {
        ::write(done_event_fd, &sig, sizeof(sig));
        return;
    }
} ThreadCtx_t;

// 创建 / 销毁 worker threads 
// 初始化 ThreadCtx
// 实际管理 threads 生命周期
class ThreadContextManager {
friend class Singleton<ThreadContextManager>; 
private:
    ThreadContextManager(UINT num = 1);
    ~ThreadContextManager();
public:
    inline UINT size() const { return _num; }
    void setThreadsNum(UINT num);
    void setCallerCtx();
    void clear();
    void join_all();
    void launch_workers(UINT id, void *Args,
        std::function<void(ThreadCtx_t &, Executor *, void *)> entry
    );
    ThreadCtx_t *ctx(UINT id = 0);
    ThreadCtx_t *caller_ctx();
    // 向所有 worker 广播一次任务
    void broadcast_task(EventPayload ev);
    // 等待所有 worker 完成 (一次 barrier)
    void wait_all_done();
private:
    // 总线程数量
    UINT _num;  // 表示 threadCtx 的数量 (_workersNum + 1)
    SharedContext_t _shared;
    std::vector<ThreadCtx_t> _ctxs;
    std::vector<std::thread> _workers;
};
#define RUNTIME (Singleton<ThreadContextManager>::getInstance())

} // namespace end of Kernel 
#endif 