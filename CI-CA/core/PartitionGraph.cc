#include <core/PartitionGraph.h>

namespace Kernel {
namespace core {

namespace {

using PartIndexMap = std::unordered_map<const Layer *, UINT>;

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

PartIndexMap make_part_index_map(const std::vector<const ExecutionPartition *>& parts) {
    PartIndexMap ids;
    UINT part_index = 0;
    for (auto it = parts.begin(); it != parts.end(); ++it, ++part_index) {
        const ExecutionPartition *part = *it;
        EXIT_ERROR_CHECK_EQ(nullptr, part, "ExecutionPartition is nullptr");
        for (auto lit = part->layers().begin(); lit != part->layers().end(); ++lit) {
            ids[*lit] = part_index;
        }
    }
    return ids;
}

} // namespace

void PartitionGraph::clear() {
    _parts.clear();
    _edges.clear();
    _outs.clear();
    _ins.clear();
}

void PartitionGraph::build(const std::vector<ExecutionPartition>& parts) {
    clear();

    _parts.reserve(parts.size());
    for (auto it = parts.begin(); it != parts.end(); ++it) {
        _parts.push_back(&(*it));
    }

    _outs.resize(parts.size());
    _ins.resize(parts.size());

    const PartIndexMap ids = make_part_index_map(_parts);

    UINT from_index = 0;
    for (auto pit = _parts.begin(); pit != _parts.end(); ++pit, ++from_index) {
        const ExecutionPartition *part = *pit;
        EXIT_ERROR_CHECK_EQ(nullptr, part, "ExecutionPartition is nullptr");

        for (auto vit = part->outputs().begin(); vit != part->outputs().end(); ++vit) {
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

} // namespace core
} // namespace Kernel
