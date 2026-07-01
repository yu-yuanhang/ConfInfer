#ifndef __PARTITION_GRAPH_H_CA__
#define __PARTITION_GRAPH_H_CA__

#include <unordered_map>
#include <core/ExecutionPartition.h>

namespace Kernel {
namespace core {

struct PartitionEdge {
    PartitionEdge()
        : from(INVALID_VALUE_U),
          to(INVALID_VALUE_U),
          values() {}

    // from / to 是 PartitionGraph 内部的局部下标
    // 不是 ExecutionPartition::id()
    UINT from;
    UINT to;
    std::vector<Value_t *> values;
};

// ExecutionPartition 也只是用来表示一个具体的执行分区
// 分区之间的需要有个图结构来表示它们之间的依赖关系
class PartitionGraph {
public:
    PartitionGraph()
        : _parts(),
          _edges(),
          _outs(),
          _ins() {}

    void clear();
    void build(const std::vector<ExecutionPartition>& parts);

    bool empty() const { return _parts.empty(); }
    UINT size() const { return static_cast<UINT>(_parts.size()); }

    const std::vector<const ExecutionPartition *>& parts() const { return _parts; }
    const std::vector<PartitionEdge>& edges() const { return _edges; }
    // 这里的 id 也是 PartitionGraph 内部局部下标
    const std::vector<UINT>& outs(UINT id) const;
    const std::vector<UINT>& ins(UINT id) const;

private:
    void addEdge(UINT from, UINT to, Value_t *value);

private:
    // PartitionGraph 不拥有 partition 生命周期
    // 这里只是引用 Network / Graph 分析阶段已经稳定存在的 partition 结果
    std::vector<const ExecutionPartition *> _parts;
    std::vector<PartitionEdge> _edges;
    std::vector<std::vector<UINT>> _outs;
    std::vector<std::vector<UINT>> _ins;
};

} // namespace core
} // namespace Kernel

#endif
