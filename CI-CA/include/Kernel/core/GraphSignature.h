#ifndef __GRAPH_SIGNATURE_H_CA__
#define __GRAPH_SIGNATURE_H_CA__

#include <All.h>
#include <core/Param.h>

namespace Kernel {
namespace core {

struct GraphInputSlot {
    std::string name;
    Value_t* value;

    GraphInputSlot() : name(), value(nullptr) {}
    GraphInputSlot(const std::string& n, Value_t& v)
        : name(n), value(&v) {}
};

struct GraphOutputSlot {
    std::string name;
    Value_t* value;

    GraphOutputSlot() : name(), value(nullptr) {}
    GraphOutputSlot(const std::string& n, Value_t& v)
        : name(n), value(&v) {}
};

struct GraphSignature {
    std::vector<GraphInputSlot> inputs;
    std::vector<GraphOutputSlot> outputs;
};

} // namespace core
} // namespace Kernel

#endif
