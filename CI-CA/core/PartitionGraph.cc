#include <core/PartitionGraph.h>
#include <queue>

namespace Kernel {
namespace core {

namespace {

using PartIndexMap = std::unordered_map<const Layer *, UINT>;
using LayerSet = std::unordered_set<Layer *>;

void append_unique(std::vector<UINT>& vec, UINT value) {
    if (vec.end() == std::find(vec.begin(), vec.end(), value)) {
        vec.push_back(value);
    }
}

void append_unique(std::vector<Value_t *>& vec, Value_t *value) {
    if (vec.end() == std::find(vec.begin(), vec.end(), value)) {
        vec.push_back(value);
    }
}

PartIndexMap make_part_index_map(const std::vector<ExecPartition>& parts) {
    PartIndexMap ids;
    UINT part_index = 0;
    for (auto it = parts.begin(); it != parts.end(); ++it, ++part_index) {
        const ExecPartition& part = *it;
        for (auto lit = part.layers().begin(); lit != part.layers().end(); ++lit) {
            ids[*lit] = part_index;
        }
    }
    return ids;
}

LayerSet make_layer_set(const ExecPartition& part) {
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

void finalize_partition(const Graph& graph, ExecPartition& part) {
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
                                 const ExecPartition& current,
                                 Layer *layer) {
    if (current.empty()) {
        return false;
    }

    Layer *tail = current.back();
    EXIT_ERROR_CHECK_EQ(nullptr, tail, "ExecPartition tail is nullptr");

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

bool exceeds_limits(const ExecPartition& part, const PartitionBuildOptions& opts) {
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
                      const ExecPartition& current,
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

void validate_partition(const ExecPartition& part, const PartitionBuildOptions& opts) {
    EXIT_ERROR_CHECK_EQ(true, part.empty(), "ExecPartition must not be empty");

    if (opts.singleDomainOnly) {
        for (auto it = part.layers().begin(); it != part.layers().end(); ++it) {
            EXIT_ERROR_CHECK_NE((*it)->execDomain(), part.domain(),
                "ExecPartition contains mixed execution domains");
        }
    }

    if (!opts.allowMultiIO) {
        EXIT_ERROR_CHECK_NE(part.inputs().size() <= 1, true,
            "ExecPartition multi-input is disabled");
        EXIT_ERROR_CHECK_NE(part.outputs().size() <= 1, true,
            "ExecPartition multi-output is disabled");
    }

    EXIT_ERROR_CHECK_EQ(true, exceeds_limits(part, opts),
        "ExecPartition exceeds build option limits");
}

} // namespace

void PartitionGraph::clear() {
    _parts.clear();
    _edges.clear();
    _outs.clear();
    _ins.clear();
    _topo_order.clear();
}

void PartitionGraph::build(const Graph& graph) {
    build(graph, _opts);
}

void PartitionGraph::build(const Graph& graph, const PartitionBuildOptions& opts) {
    clear();
    _opts = opts;
    buildParts(graph, opts);
    buildEdges();
    buildTopoOrder();
}

ExecPartition& PartitionGraph::part(UINT id) {
    EXIT_ERROR_CHECK_EQ(true, id >= _parts.size(), "PartitionGraph part id out of range");
    return _parts[id];
}

const ExecPartition& PartitionGraph::part(UINT id) const {
    EXIT_ERROR_CHECK_EQ(true, id >= _parts.size(), "PartitionGraph part id out of range");
    return _parts[id];
}

void PartitionGraph::buildParts(const Graph& graph, const PartitionBuildOptions& opts) {
    const std::vector<Layer *>& order = graph.execOrder();
    if (order.empty()) {
        return;
    }

    ExecPartition current;
    bool has_current = false;

    for (auto it = order.begin(); it != order.end(); ++it) {
        Layer *layer = *it;
        EXIT_ERROR_CHECK_EQ(nullptr, layer, "Graph execOrder contains nullptr layer");

        const ExecutionDomain domain = layer->execDomain();
        if (!has_current || !can_append_layer(graph, current, layer, opts)) {
            if (has_current) {
                finalize_partition(graph, current);
                validate_partition(current, opts);
                _parts.push_back(current);
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
        _parts.push_back(current);
    }
}

void PartitionGraph::buildEdges() {
    _outs.resize(_parts.size());
    _ins.resize(_parts.size());

    const PartIndexMap ids = make_part_index_map(_parts);

    UINT from_index = 0;
    for (auto pit = _parts.begin(); pit != _parts.end(); ++pit, ++from_index) {
        const ExecPartition& part = *pit;

        for (auto vit = part.outputs().begin(); vit != part.outputs().end(); ++vit) {
            Value_t *value = *vit;
            if (nullptr == value) {
                continue;
            }

            for (auto cit = value->consumers.begin(); cit != value->consumers.end(); ++cit) {
                const Layer *consumer = *cit;
                if (nullptr == consumer) {
                    continue;
                }

                auto id_it = ids.find(consumer);
                if (id_it == ids.end()) {
                    continue;
                }

                const UINT to_index = id_it->second;
                if (to_index == from_index) {
                    continue;
                }

                addEdge(from_index, to_index, value);
            }
        }
    }
}

const std::vector<UINT>& PartitionGraph::outs(UINT id) const {
    EXIT_ERROR_CHECK_EQ(true, id >= _outs.size(), "PartitionGraph outs id out of range");
    return _outs[id];
}

const std::vector<UINT>& PartitionGraph::ins(UINT id) const {
    EXIT_ERROR_CHECK_EQ(true, id >= _ins.size(), "PartitionGraph ins id out of range");
    return _ins[id];
}

void PartitionGraph::addEdge(UINT from, UINT to, Value_t *value) {
    EXIT_ERROR_CHECK_EQ(true, from >= _outs.size(), "PartitionGraph edge from out of range");
    EXIT_ERROR_CHECK_EQ(true, to >= _ins.size(), "PartitionGraph edge to out of range");

    for (auto it = _edges.begin(); it != _edges.end(); ++it) {
        if (it->from == from && it->to == to) {
            append_unique(it->values, value);
            append_unique(_outs[from], to);
            append_unique(_ins[to], from);
            return;
        }
    }

    PartitionEdge edge;
    edge.from = from;
    edge.to = to;
    edge.values.push_back(value);
    _edges.push_back(edge);

    append_unique(_outs[from], to);
    append_unique(_ins[to], from);
}

void PartitionGraph::buildTopoOrder() {
    std::queue<UINT> ready;
    std::vector<UINT> indegree(_parts.size(), 0);

    _topo_order.clear();
    _topo_order.reserve(_parts.size());

    for (UINT i = 0; i < _parts.size(); ++i) {
        indegree[i] = static_cast<UINT>(_ins[i].size());
        if (0 == indegree[i]) {
            ready.push(i);
        }
    }

    while (!ready.empty()) {
        const UINT id = ready.front();
        ready.pop();
        _topo_order.push_back(id);

        for (UINT out : _outs[id]) {
            EXIT_ERROR_CHECK_EQ(true, out >= indegree.size(),
                                "PartitionGraph topo out id out of range");
            EXIT_ERROR_CHECK_EQ(true, 0 == indegree[out],
                                "PartitionGraph topo indegree underflow");
            --indegree[out];
            if (0 == indegree[out]) {
                ready.push(out);
            }
        }
    }

    EXIT_ERROR_CHECK_NE(_topo_order.size(), _parts.size(),
                        "PartitionGraph contains cycle or disconnected topo state");
}

} // namespace core
} // namespace Kernel
