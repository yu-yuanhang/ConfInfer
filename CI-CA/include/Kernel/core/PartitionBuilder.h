#ifndef __PARTITION_BUILDER_H_CA__
#define __PARTITION_BUILDER_H_CA__

#include <core/ExecutionPartition.h>
#include <core/Graph.h>

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

    // 这里主要是为了来规范一下 划分策略
    // 虽然设计上 具体的划分方案是依赖用户自定义的
    // 这里只是给一个限制 预留一个和用来检测是否支持的接口 
    bool allowBranchMerge;  // 表示是否允许跨分支结构继续合并
    bool allowMultiIO;  // 允许一个 partition 具有多个输入或多个输出
    bool singleDomainOnly;  // 表示一个 partition 内是否只允许存在同一个 ExecutionDomain
    // 一个 partition 中的 Layer、Input、Output 的上限数量
    // 但目前就直接给个最大值就行
    UINT maxLayers;
    UINT maxInputs;
    UINT maxOutputs;
};

class PartitionBuilder {
public:
    PartitionBuilder() : _opts() {}
    ~PartitionBuilder() = default;

    const PartitionBuildOptions& opts() const { return _opts; }
    void setOpts(const PartitionBuildOptions& opts) { _opts = opts; }

    // 这里的计划就是遍历 graph.execOrder() 按 layer->execDomain() 划分 partition
    std::vector<ExecutionPartition> build(const Graph& graph) const;
    std::vector<ExecutionPartition> build(const Graph& graph, const PartitionBuildOptions& opts) const;

private:
    PartitionBuildOptions _opts;
};

} // namespace core
} // namespace Kernel

#endif
