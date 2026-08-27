#ifndef __PARTITION_BUILDER_H_CA__
#define __PARTITION_BUILDER_H_CA__

#include <core/PartitionGraph.h>

namespace Kernel {
namespace core {

class PartitionBuilder {
public:
    PartitionBuilder() : _opts() {}
    ~PartitionBuilder() = default;

    const PartitionBuildOptions& opts() const { return _opts; }
    void setOpts(const PartitionBuildOptions& opts) { _opts = opts; }

    // 这里的计划就是遍历 graph.execOrder() 按 layer->execDomain() 划分 partition
    std::vector<ExecPartition> build(const Graph& graph) const;
    std::vector<ExecPartition> build(const Graph& graph, const PartitionBuildOptions& opts) const;

private:
    PartitionBuildOptions _opts;
};

} // namespace core
} // namespace Kernel

#endif
