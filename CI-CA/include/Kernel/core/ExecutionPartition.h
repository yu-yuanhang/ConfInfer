#ifndef __EXECUTION_PARTITION_H_CA__
#define __EXECUTION_PARTITION_H_CA__

#include <algorithm>
#include <core/Layer.h>

namespace Kernel {
namespace core {

// 用来表示一个执行分区 一些 Layer 的聚合
// 但是在本项目的背景下 PartitionBuilder 决定了 Layer 执行分区的 划分准则
// 默认就是代表了同一执行域且 拓扑连续的 Layers
class ExecutionPartition {
public:
    ExecutionPartition()
        : _id(_counter.fetch_add(1, std::memory_order_relaxed)),
          _domain(ExecutionDomain::ED_DEFAULT),
          _layers(),
          _inputs(),
          _outputs(),
          _internals(),
          _topo() {}

    // 这里的 id 就是 REE 侧 partition_id
    // 后续发往 TA 时 直接作为协议里的 partition_id 传递
    UINT id() const { return _id; }
    bool empty() const { return _layers.empty(); }
    UINT size() const { return static_cast<UINT>(_layers.size()); }

    ExecutionDomain domain() const { return _domain; }
    void setDomain(ExecutionDomain domain) { _domain = domain; }

    const std::vector<Layer *>& layers() const { return _layers; }
    const std::vector<Value_t *>& inputs() const { return _inputs; }
    const std::vector<Value_t *>& outputs() const { return _outputs; }
    const std::vector<Value_t *>& internals() const { return _internals; }
    const std::vector<Layer *>& topo() const { return _topo; }
    Layer *front() const { return _layers.empty() ? nullptr : _layers.front(); }
    Layer *back() const { return _layers.empty() ? nullptr : _layers.back(); }

    bool contains(const Layer *layer) const {
        return _layers.end() != std::find(_layers.begin(), _layers.end(), layer);
    }
    bool contains(const Value_t *value) const {
        return _inputs.end() != std::find(_inputs.begin(), _inputs.end(), value) ||
               _outputs.end() != std::find(_outputs.begin(), _outputs.end(), value) ||
               _internals.end() != std::find(_internals.begin(), _internals.end(), value);
    }

    void addLayer(Layer *layer) {
        pushUnique(_layers, layer);
        pushUnique(_topo, layer);
    }

    void addInput(Value_t *value) { pushUnique(_inputs, value); }
    void addOutput(Value_t *value) { pushUnique(_outputs, value); }
    void addInternal(Value_t *value) { pushUnique(_internals, value); }

    void clear() {
        _id = _counter.fetch_add(1, std::memory_order_relaxed);
        _domain = ExecutionDomain::ED_DEFAULT;
        _layers.clear();
        _inputs.clear();
        _outputs.clear();
        _internals.clear();
        _topo.clear();
    }

private:
    template <typename T>
    static void pushUnique(std::vector<T>& vec, T item) {
        if (std::find(vec.begin(), vec.end(), item) == vec.end()) {
            vec.push_back(item);
        }
    }

private:
    static std::atomic<UINT> _counter;
    UINT _id;
    ExecutionDomain _domain;
    // 这里在设计原则上 ExecutionPartition 不应该直接拥有数据的管理权
    // ExecutionPartition 不负责以下数据生命周期的管理
    // 只作为 Graph 生命周期内的只读观察者
    std::vector<Layer *> _layers;
    std::vector<Value_t *> _inputs;
    std::vector<Value_t *> _outputs;
    std::vector<Value_t *> _internals;
    // partition 内部节点的执行顺序
    std::vector<Layer *> _topo;
};

} // namespace core
} // namespace Kernel

#endif
