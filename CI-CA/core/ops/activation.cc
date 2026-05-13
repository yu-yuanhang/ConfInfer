#include <activation.h>

namespace Kernel {
namespace core {

UnaryOp_L::~UnaryOp_L() {}
ReLU::~ReLU() {}
Sigmoid::~Sigmoid() {}
Dropout::~Dropout() {}
Softmax_L::~Softmax_L() {}
Softmax::~Softmax() {}

UnaryOp_L::UnaryOp_L(LayerType type, OpSignature *opSignature):
    Layer(type, opSignature) {}

void UnaryOp_L::makeParams(Params *params) {
    (void)params;
}

void UnaryOp_L::makeOutputs() {
    EXIT_ERROR_CHECK_NE(1, _inputs.size(), "Unary op only supports one input");
    Value_t* input = _inputs[0];
    EXIT_ERROR_CHECK_EQ(nullptr, input, "Unary op input is nullptr");

    std::unique_ptr<Value_t> out =
        std::make_unique<Value_t>(PARAM_INTERMEDIATE | PARAM_OWN_DATA);
    out->copyDescFrom(*input);
    out->data.flags = PARAM_INTERMEDIATE | PARAM_OWN_DATA;
    out->data.ptr = new(std::nothrow) char[out->data.shape.size * out->data.getTypeSize()];
    EXIT_ERROR_CHECK_EQ(nullptr, out->data.ptr, "Unary op output allocation failed");
    out->producer = this;
    out->output_index = 0;
    out->kind = OutputKind::Default;

    _outputs.clear();
    _outputs.reserve(1);
    _outputs.push_back(std::move(out));
}

UINT UnaryOp_L::calcWorkspaceSize() { return 0; }

Softmax_L::Softmax_L(INT dim, OpSignature *opSignature):
    Layer(LayerType::SOFTMAX, opSignature),
    _dim(dim) {}

void Softmax_L::makeParams(Params *params) {
    (void)params;
}

void Softmax_L::makeOutputs() {
    EXIT_ERROR_CHECK_NE(1, _inputs.size(), "Softmax only supports one input");
    Value_t* input = _inputs[0];
    EXIT_ERROR_CHECK_EQ(nullptr, input, "Softmax input is nullptr");
    EXIT_ERROR_CHECK_EQ(0, input->data.shape.ndim, "Softmax input ndim must be > 0");

    INT axis = _dim;
    if (axis < 0) {
        axis += static_cast<INT>(input->data.shape.ndim);
    }
    EXIT_ERROR_CHECK_EQ(false,
        axis >= 0 && axis < static_cast<INT>(input->data.shape.ndim),
        "Softmax dim out of range");

    std::unique_ptr<Value_t> out =
        std::make_unique<Value_t>(PARAM_INTERMEDIATE | PARAM_OWN_DATA);
    out->copyDescFrom(*input);
    out->data.flags = PARAM_INTERMEDIATE | PARAM_OWN_DATA;
    out->data.ptr = new(std::nothrow) char[out->data.shape.size * out->data.getTypeSize()];
    EXIT_ERROR_CHECK_EQ(nullptr, out->data.ptr, "Softmax output allocation failed");
    out->producer = this;
    out->output_index = 0;
    out->kind = OutputKind::Default;

    _outputs.clear();
    _outputs.reserve(1);
    _outputs.push_back(std::move(out));
}

UINT Softmax_L::calcWorkspaceSize() { return 0; }

ReLU::ReLU(BOOL inplace):
    OpSignature(LayerType::RELU),
    _inplace(inplace) {}

Layer &ReLU::operator()(Value_t &value) {
    EXIT_ERROR_CHECK_EQ(true, _inplace, "ReLU inplace=true is unsupported");
    Layer *l = (Layer *) new(std::nothrow) UnaryOp_L(LayerType::RELU, this);
    EXIT_ERROR_CHECK_EQ(l, nullptr, "(new) heap allocation failed");
    dealParams(l);
    _layers.push_back(l);
    return l->link(value);
}

Sigmoid::Sigmoid(BOOL inplace):
    OpSignature(LayerType::SIGMOID),
    _inplace(inplace) {}

Layer &Sigmoid::operator()(Value_t &value) {
    EXIT_ERROR_CHECK_EQ(true, _inplace, "Sigmoid inplace=true is unsupported");
    Layer *l = (Layer *) new(std::nothrow) UnaryOp_L(LayerType::SIGMOID, this);
    EXIT_ERROR_CHECK_EQ(l, nullptr, "(new) heap allocation failed");
    dealParams(l);
    _layers.push_back(l);
    return l->link(value);
}

Dropout::Dropout(FLOAT p, BOOL inplace):
    OpSignature(LayerType::DROPOUT),
    _p(p),
    _inplace(inplace) {}

Layer &Dropout::operator()(Value_t &value) {
    EXIT_ERROR_CHECK_EQ(true, _inplace, "Dropout inplace=true is unsupported");
    if (!(_p >= 0.0f && _p <= 1.0f)) {
        EXIT_ERROR("Dropout p must be in [0, 1]");
    }
    Layer *l = (Layer *) new(std::nothrow) UnaryOp_L(LayerType::DROPOUT, this);
    EXIT_ERROR_CHECK_EQ(l, nullptr, "(new) heap allocation failed");
    dealParams(l);
    _layers.push_back(l);
    return l->link(value);
}

Softmax::Softmax(INT dim):
    OpSignature(LayerType::SOFTMAX),
    _dim(dim) {}

Layer &Softmax::operator()(Value_t &value) {
    Layer *l = (Layer *) new(std::nothrow) Softmax_L(_dim, this);
    EXIT_ERROR_CHECK_EQ(l, nullptr, "(new) heap allocation failed");
    dealParams(l);
    _layers.push_back(l);
    return l->link(value);
}

} // namespace core
} // namespace Kernel
