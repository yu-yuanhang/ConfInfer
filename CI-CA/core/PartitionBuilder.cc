#include <core/PartitionBuilder.h>

namespace Kernel {
namespace core {

std::vector<ExecPartition> PartitionBuilder::build(const Graph& graph) const {
    return build(graph, _opts);
}

std::vector<ExecPartition> PartitionBuilder::build(const Graph& graph,
                                                   const PartitionBuildOptions& opts) const {
    PartitionGraph part_graph;
    part_graph.build(graph, opts);
    return part_graph.parts();
}

} // namespace core
} // namespace Kernel
