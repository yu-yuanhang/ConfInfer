#include <core/BoundaryLayer.h>

namespace Kernel {
namespace core {

namespace {
class BoundaryOpSignature : public OpSignature {
public:
    explicit BoundaryOpSignature(LayerType type) : OpSignature(type) {}
};

OpSignature* input_boundary_sig() {
    static BoundaryOpSignature sig(LayerType::GRAPH_INPUT);
    return &sig;
}

OpSignature* output_boundary_sig() {
    static BoundaryOpSignature sig(LayerType::GRAPH_OUTPUT);
    return &sig;
}
} // namespace

GraphInputLayer::GraphInputLayer(const std::vector<Value_t*>& values)
    : Layer(LayerType::GRAPH_INPUT, input_boundary_sig()) {
    _outputs.clear();
    _outputs.reserve(values.size());

    for (UINT i = 0; i < values.size(); ++i) {
        Value_t* src = values[i];
        EXIT_ERROR_CHECK_EQ(nullptr, src, "GraphInputLayer bound input is nullptr");

        std::unique_ptr<Value_t> out = std::make_unique<Value_t>();
        out->borrowFrom(*src, PARAM_INPUT);
        out->producer = this;
        out->output_index = i;
        out->kind = OutputKind::Default;

        _outputs.push_back(std::move(out));
    }
}

void GraphInputLayer::makeOutputs() { return; }

GraphOutputLayer::GraphOutputLayer(const std::vector<Value_t*>& values)
    : Layer(LayerType::GRAPH_OUTPUT, output_boundary_sig()) {
    bindInputs(values);
}

void GraphOutputLayer::makeOutputs() { return; }

Value_t& GraphOutputLayer::input(UINT idx) {
    EXIT_ERROR_CHECK_EQ(false, idx < _inputs.size(), "GraphOutputLayer input index out of range");
    Value_t* value = _inputs[idx];
    EXIT_ERROR_CHECK_EQ(nullptr, value, "GraphOutputLayer input is nullptr");
    return *value;
}

void GraphOutputLayer::bindInputs(const std::vector<Value_t*>& values) {
    _inputs.clear();
    _inputs.reserve(values.size());

    for (auto it = values.begin(); it != values.end(); ++it) {
        Value_t* value = *it;
        EXIT_ERROR_CHECK_EQ(nullptr, value, "GraphOutputLayer input is nullptr");

        _inputs.push_back(value);
        if (value->producer && !_inputsL.contains(value->producer)) {
            _inputsL.push_back(value->producer);
            ++_inputsLNum;
        }
        value->consumers.push_back(this);
    }
}

} // namespace core
} // namespace Kernel
