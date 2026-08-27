#ifndef __PARTITION_GRAPH_H_CA__
#define __PARTITION_GRAPH_H_CA__

#include <core/Graph.h>
#include <unordered_map>
#include <core/ExecutionPartition.h>

namespace Kernel {
namespace core {

struct PartitionBuildOptions {
    PartitionBuildOptions()
        : allowBranchMerge(false),
          allowMultiIO(true),
          singleDomainOnly(true),
          maxLayers(INVALID_VALUE_U),
          maxInputs(INVALID_VALUE_U),
          maxOutputs(INVALID_VALUE_U) {}

    bool allowBranchMerge;
    bool allowMultiIO;
    bool singleDomainOnly;
    UINT maxLayers;
    UINT maxInputs;
    UINT maxOutputs;
};

struct PartitionEdge {
    PartitionEdge()
        : from(INVALID_VALUE_U),
          to(INVALID_VALUE_U),
          values() {}

    // from / to 是 PartitionGraph 内部的局部下标
    // 不是 ExecPartition::id()
    UINT from;
    UINT to;
    std::vector<Value_t *> values;
};

// ExecPartition 也只是用来表示一个具体的执行分区
// 分区之间的需要有个图结构来表示它们之间的依赖关系
class PartitionGraph {
public:
    PartitionGraph()
        : _parts(),
          _edges(),
          _outs(),
          _ins(),
          _topo_order(),
          _opts() {}

    void clear();
    void build(const Graph& graph);
    void build(const Graph& graph, const PartitionBuildOptions& opts);

    bool empty() const { return _parts.empty(); }
    UINT size() const { return static_cast<UINT>(_parts.size()); }

    const std::vector<ExecPartition>& parts() const { return _parts; }
    const std::vector<PartitionEdge>& edges() const { return _edges; }
    // 这里的 id 也是 PartitionGraph 内部局部下标
    const std::vector<UINT>& outs(UINT id) const;
    const std::vector<UINT>& ins(UINT id) const;
    const std::vector<UINT>& topoOrder() const { return _topo_order; }
    const PartitionBuildOptions& opts() const { return _opts; }
    void setOpts(const PartitionBuildOptions& opts) { _opts = opts; }
    ExecPartition& part(UINT id);
    const ExecPartition& part(UINT id) const;

private:
    void addEdge(UINT from, UINT to, Value_t *value);
    void buildTopoOrder();
    void buildParts(const Graph& graph, const PartitionBuildOptions& opts);
    void buildEdges();

private:
    std::vector<ExecPartition> _parts;
    std::vector<PartitionEdge> _edges;
    std::vector<std::vector<UINT>> _outs;
    std::vector<std::vector<UINT>> _ins;
    std::vector<UINT> _topo_order;
    PartitionBuildOptions _opts;
};

} // namespace core
} // namespace Kernel

#endif
