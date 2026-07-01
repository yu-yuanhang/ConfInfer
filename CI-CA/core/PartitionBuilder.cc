#include <core/PartitionBuilder.h>

namespace Kernel {
namespace core {

namespace {

using LayerSet = std::unordered_set<Layer *>;

LayerSet make_layer_set(const ExecutionPartition& part) {
    LayerSet layers;
    layers.reserve(part.layers().size());
    for (auto it = part.layers().begin(); it != part.layers().end(); ++it) {
        layers.insert(*it);
    }
    return layers;
}

bool is_partition_input(const LayerSet& layers, Value_t *value) {
    if (nullptr == value) {
        return false;
    }
    return nullptr == value->producer || 0 == layers.count(value->producer);
}

bool is_partition_output(const Graph& graph, const LayerSet& layers, Value_t *value) {
    if (nullptr == value) {
        return false;
    }
    if (graph.isGraphOutputValue(value)) {
        return true;
    }
    for (auto it = value->consumers.begin(); it != value->consumers.end(); ++it) {
        if (0 == layers.count(*it)) {
            return true;
        }
    }
    return false;
}

bool is_partition_internal(const Graph& graph, const LayerSet& layers, Value_t *value) {
    if (nullptr == value || graph.isGraphOutputValue(value)) {
        return false;
    }
    if (nullptr == value->producer || 0 == layers.count(value->producer)) {
        return false;
    }
    if (value->consumers.empty()) {
        return false;
    }
    for (auto it = value->consumers.begin(); it != value->consumers.end(); ++it) {
        if (0 == layers.count(*it)) {
            return false;
        }
    }
    return true;
}

void finalize_partition(const Graph& graph, ExecutionPartition& part) {
    const LayerSet layers = make_layer_set(part);

    for (auto lit = part.layers().begin(); lit != part.layers().end(); ++lit) {
        Layer *layer = *lit;
        std::vector<Value_t *> inputs = graph.ins(layer);
        for (auto vit = inputs.begin(); vit != inputs.end(); ++vit) {
            if (is_partition_input(layers, *vit)) {
                part.addInput(*vit);
            }
        }

        std::vector<Value_t *> outputs = graph.outs(layer);
        for (auto vit = outputs.begin(); vit != outputs.end(); ++vit) {
            if (is_partition_output(graph, layers, *vit)) {
                part.addOutput(*vit);
            } else if (is_partition_internal(graph, layers, *vit)) {
                part.addInternal(*vit);
            }
        }
    }
}

bool would_cross_branch_boundary(const Graph& graph,
                                 const ExecutionPartition& current,
                                 Layer *layer) {
    if (current.empty()) {
        return false;
    }

    Layer *tail = current.back();
    EXIT_ERROR_CHECK_EQ(nullptr, tail, "ExecutionPartition tail is nullptr");

    const std::vector<Layer *> prevs = graph.prevs(layer);
    const std::vector<Layer *> nexts = graph.nexts(tail);

    if (prevs.size() > 1 || nexts.size() > 1) {
        return true;
    }

    UINT inside_prev_num = 0;
    for (auto it = prevs.begin(); it != prevs.end(); ++it) {
        if (current.contains(*it)) {
            ++inside_prev_num;
        }
    }
    if (inside_prev_num > 1) {
        return true;
    }

    return false;
}

bool exceeds_limits(const ExecutionPartition& part, const PartitionBuildOptions& opts) {
    if (opts.maxLayers != INVALID_VALUE_U && part.layers().size() > opts.maxLayers) {
        return true;
    }
    if (opts.maxInputs != INVALID_VALUE_U && part.inputs().size() > opts.maxInputs) {
        return true;
    }
    if (opts.maxOutputs != INVALID_VALUE_U && part.outputs().size() > opts.maxOutputs) {
        return true;
    }
    return false;
}

bool can_append_layer(const Graph& graph,
                      const ExecutionPartition& current,
                      Layer *layer,
                      const PartitionBuildOptions& opts) {
    if (nullptr == layer) {
        return false;
    }
    if (current.empty()) {
        return true;
    }
    if (opts.singleDomainOnly && current.domain() != layer->execDomain()) {
        return false;
    }
    if (opts.maxLayers != INVALID_VALUE_U && current.layers().size() >= opts.maxLayers) {
        return false;
    }
    if (!opts.allowBranchMerge && would_cross_branch_boundary(graph, current, layer)) {
        return false;
    }
    return true;
}

void validate_partition(const ExecutionPartition& part, const PartitionBuildOptions& opts) {
    EXIT_ERROR_CHECK_EQ(true, part.empty(), "ExecutionPartition must not be empty");

    if (opts.singleDomainOnly) {
        for (auto it = part.layers().begin(); it != part.layers().end(); ++it) {
            EXIT_ERROR_CHECK_NE((*it)->execDomain(), part.domain(),
                "ExecutionPartition contains mixed execution domains");
        }
    }

    if (!opts.allowMultiIO) {
        EXIT_ERROR_CHECK_NE(part.inputs().size() <= 1, true,
            "ExecutionPartition multi-input is disabled");
        EXIT_ERROR_CHECK_NE(part.outputs().size() <= 1, true,
            "ExecutionPartition multi-output is disabled");
    }

    EXIT_ERROR_CHECK_EQ(true, exceeds_limits(part, opts),
        "ExecutionPartition exceeds build option limits");
}

} // namespace

std::vector<ExecutionPartition> PartitionBuilder::build(const Graph& graph) const {
    return build(graph, _opts);
}

std::vector<ExecutionPartition> PartitionBuilder::build(const Graph& graph,
                                                        const PartitionBuildOptions& opts) const {
    std::vector<ExecutionPartition> parts;
    const std::vector<Layer *>& order = graph.execOrder();
    if (order.empty()) {
        return parts;
    }

    ExecutionPartition current;
    bool has_current = false;

    for (auto it = order.begin(); it != order.end(); ++it) {
        Layer *layer = *it;
        EXIT_ERROR_CHECK_EQ(nullptr, layer, "Graph execOrder contains nullptr layer");

        // 目前我也不想添加太复杂的 分区划分逻辑 
        // 就直接按照 执行区域来划分就行
        const ExecutionDomain domain = layer->execDomain();
        if (!has_current || !can_append_layer(graph, current, layer, opts)) {
            if (has_current) {
                finalize_partition(graph, current);
                validate_partition(current, opts);
                parts.push_back(current);
            }
            current.clear();
            current.setDomain(domain);
            has_current = true;
        }
        current.addLayer(layer);
    }

    if (has_current) {
        finalize_partition(graph, current);
        validate_partition(current, opts);
        parts.push_back(current);
    }

    return parts;
}

} // namespace core
} // namespace Kernel
