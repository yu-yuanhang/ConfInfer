#include <arithmetic.h>

namespace Kernel {
namespace core {

Add_L::~Add_L() {}
BiasAdd_L::~BiasAdd_L() {}
MatMul_L::~MatMul_L() {}
Concat_L::~Concat_L() {}
Add::~Add() {}
BiasAdd::~BiasAdd() {}
MatMul::~MatMul() {}
Concat::~Concat() {}

namespace {
void require_same_shape(const DataShape_t& lhs, const DataShape_t& rhs, const char* msg) {
    EXIT_ERROR_CHECK_NE(lhs.ndim, rhs.ndim, msg);
    EXIT_ERROR_CHECK_NE(lhs.size, rhs.size, msg);
    for (UINT i = 0; i < lhs.ndim; ++i) {
        EXIT_ERROR_CHECK_NE(lhs.dims[i], rhs.dims[i], msg);
    }
}
}

Add_L::Add_L(FLOAT alpha, OpSignature *opSignature):
    Layer(LayerType::ADD, opSignature),
    _alpha(alpha) {}

BiasAdd_L::BiasAdd_L(UINT size, INT dim, OpSignature *opSignature):
    Layer(LayerType::BIASADD, opSignature),
    _size(size),
    _dim(dim) {}

void Add_L::makeParams(Params *params) {
    (void)params;
}

void Add_L::makeOutputs() {
    EXIT_ERROR_CHECK_NE(2, _inputs.size(), "Add expects two inputs");
    Value_t* input = _inputs[0];
    Value_t* other = _inputs[1];
    EXIT_ERROR_CHECK_EQ(nullptr, input, "Add input is nullptr");
    EXIT_ERROR_CHECK_EQ(nullptr, other, "Add other is nullptr");
    require_same_shape(input->data.shape, other->data.shape, "Add input shape mismatch");

    std::unique_ptr<Value_t> out =
        std::make_unique<Value_t>(PARAM_INTERMEDIATE | PARAM_OWN_DATA);
    out->copyDescFrom(*input);
    out->data.flags = PARAM_INTERMEDIATE | PARAM_OWN_DATA;
    out->data.ptr = new(std::nothrow) char[out->data.shape.size * out->data.getTypeSize()];
    EXIT_ERROR_CHECK_EQ(nullptr, out->data.ptr, "Add output allocation failed");
    out->producer = this;
    out->output_index = 0;
    out->kind = OutputKind::Default;

    _outputs.clear();
    _outputs.reserve(1);
    _outputs.push_back(std::move(out));
}

UINT Add_L::calcWorkspaceSize() { return 0; }

void BiasAdd_L::makeParams(Params *params) {
    EXIT_ERROR_CHECK_EQ(nullptr, params, "Params is nullptr");

    Data_t* bias = new(std::nothrow) Data_t(PARAM_CONST | PARAM_OWN_DATA, {_size});
    EXIT_ERROR_CHECK_EQ(nullptr, bias, "BiasAdd bias alloc failed");
    bias->ptr = new(std::nothrow) char[bias->shape.size * bias->getTypeSize()];
    EXIT_ERROR_CHECK_EQ(nullptr, bias->ptr, "BiasAdd bias buffer alloc failed");
    fill_random(bias->ptr, bias->dtype, bias->shape.size, TIMESEED);
    params->insert(ParamRole::BIAS, bias);
}

void BiasAdd_L::makeOutputs() {
    EXIT_ERROR_CHECK_NE(1, _inputs.size(), "BiasAdd only supports one input");
    Value_t* input = _inputs[0];
    EXIT_ERROR_CHECK_EQ(nullptr, input, "BiasAdd input is nullptr");
    const DataShape_t& shape = input->data.shape;
    EXIT_ERROR_CHECK_EQ(0, shape.ndim, "BiasAdd input ndim must be > 0");

    INT axis = _dim;
    if (axis < 0) {
        axis += static_cast<INT>(shape.ndim);
    }
    EXIT_ERROR_CHECK_EQ(false,
        axis >= 0 && axis < static_cast<INT>(shape.ndim),
        "BiasAdd dim out of range");
    EXIT_ERROR_CHECK_NE(_size, shape.dims[static_cast<UINT>(axis)],
        "BiasAdd size must match selected axis");

    std::unique_ptr<Value_t> out =
        std::make_unique<Value_t>(PARAM_INTERMEDIATE | PARAM_OWN_DATA);
    out->copyDescFrom(*input);
    out->data.flags = PARAM_INTERMEDIATE | PARAM_OWN_DATA;
    out->data.ptr = new(std::nothrow) char[out->data.shape.size * out->data.getTypeSize()];
    EXIT_ERROR_CHECK_EQ(nullptr, out->data.ptr, "BiasAdd output allocation failed");
    out->producer = this;
    out->output_index = 0;
    out->kind = OutputKind::Default;

    _outputs.clear();
    _outputs.reserve(1);
    _outputs.push_back(std::move(out));
}

UINT BiasAdd_L::calcWorkspaceSize() { return 0; }

MatMul_L::MatMul_L(OpSignature *opSignature):
    Layer(LayerType::MATMUL, opSignature) {}

Concat_L::Concat_L(INT dim, OpSignature *opSignature):
    Layer(LayerType::CONCAT, opSignature),
    _dim(dim) {}

void MatMul_L::makeParams(Params *params) {
    (void)params;
}

void MatMul_L::makeOutputs() {
    EXIT_ERROR_CHECK_NE(2, _inputs.size(), "MatMul expects two inputs");
    Value_t* input = _inputs[0];
    Value_t* other = _inputs[1];
    EXIT_ERROR_CHECK_EQ(nullptr, input, "MatMul input is nullptr");
    EXIT_ERROR_CHECK_EQ(nullptr, other, "MatMul other is nullptr");
    const DataShape_t& a = input->data.shape;
    const DataShape_t& b = other->data.shape;

    EXIT_ERROR_CHECK_EQ(false, a.ndim >= 2, "MatMul input ndim must be >= 2");
    EXIT_ERROR_CHECK_NE(a.ndim, b.ndim, "MatMul currently requires same-rank inputs");
    for (UINT i = 0; i + 2 < a.ndim; ++i) {
        EXIT_ERROR_CHECK_NE(a.dims[i], b.dims[i], "MatMul batch dims mismatch");
    }
    EXIT_ERROR_CHECK_NE(a.dims[a.ndim - 1], b.dims[b.ndim - 2], "MatMul inner dim mismatch");

    std::unique_ptr<Value_t> out =
        std::make_unique<Value_t>(PARAM_INTERMEDIATE | PARAM_OWN_DATA);
    DataShape_t& shape = out->data.shape;
    shape.ndim = a.ndim;
    shape.size = 1;
    for (UINT i = 0; i + 2 < a.ndim; ++i) {
        shape.dims[i] = a.dims[i];
        shape.size *= shape.dims[i];
    }
    shape.dims[a.ndim - 2] = a.dims[a.ndim - 2];
    shape.dims[a.ndim - 1] = b.dims[b.ndim - 1];
    shape.size *= shape.dims[a.ndim - 2];
    shape.size *= shape.dims[a.ndim - 1];

    out->data.copyTypeFrom(input->data);
    out->data.flags = PARAM_INTERMEDIATE | PARAM_OWN_DATA;
    out->data.ptr = new(std::nothrow) char[shape.size * out->data.getTypeSize()];
    EXIT_ERROR_CHECK_EQ(nullptr, out->data.ptr, "MatMul output allocation failed");
    out->producer = this;
    out->output_index = 0;
    out->kind = OutputKind::Default;

    _outputs.clear();
    _outputs.reserve(1);
    _outputs.push_back(std::move(out));
}

UINT MatMul_L::calcWorkspaceSize() { return 0; }

void Concat_L::makeParams(Params *params) {
    (void)params;
}

void Concat_L::makeOutputs() {
    EXIT_ERROR_CHECK_EQ(true, _inputs.empty(), "Concat expects at least one input");
    Value_t* first = _inputs[0];
    EXIT_ERROR_CHECK_EQ(nullptr, first, "Concat input is nullptr");
    const DataShape_t& base = first->data.shape;
    EXIT_ERROR_CHECK_EQ(0, base.ndim, "Concat input ndim must be > 0");

    INT axis = _dim;
    if (axis < 0) {
        axis += static_cast<INT>(base.ndim);
    }
    EXIT_ERROR_CHECK_EQ(false,
        axis >= 0 && axis < static_cast<INT>(base.ndim),
        "Concat dim out of range");

    std::unique_ptr<Value_t> out =
        std::make_unique<Value_t>(PARAM_INTERMEDIATE | PARAM_OWN_DATA);
    out->copyDescFrom(*first);
    DataShape_t& out_shape = out->data.shape;
    out_shape.size = 1;
    out_shape.dims[static_cast<UINT>(axis)] = 0;

    for (UINT i = 0; i < _inputs.size(); ++i) {
        Value_t* input = _inputs[i];
        EXIT_ERROR_CHECK_EQ(nullptr, input, "Concat input is nullptr");
        EXIT_ERROR_CHECK_NE(base.ndim, input->data.shape.ndim, "Concat input ndim mismatch");
        for (UINT d = 0; d < base.ndim; ++d) {
            if (d == static_cast<UINT>(axis)) {
                continue;
            }
            EXIT_ERROR_CHECK_NE(base.dims[d], input->data.shape.dims[d], "Concat input shape mismatch");
        }
        out_shape.dims[static_cast<UINT>(axis)] += input->data.shape.dims[static_cast<UINT>(axis)];
    }
    for (UINT d = 0; d < out_shape.ndim; ++d) {
        out_shape.size *= out_shape.dims[d];
    }

    out->data.flags = PARAM_INTERMEDIATE | PARAM_OWN_DATA;
    out->data.ptr = new(std::nothrow) char[out_shape.size * out->data.getTypeSize()];
    EXIT_ERROR_CHECK_EQ(nullptr, out->data.ptr, "Concat output allocation failed");
    out->producer = this;
    out->output_index = 0;
    out->kind = OutputKind::Default;

    _outputs.clear();
    _outputs.reserve(1);
    _outputs.push_back(std::move(out));
}

UINT Concat_L::calcWorkspaceSize() { return 0; }

Add::Add(FLOAT alpha):
    OpSignature(LayerType::ADD),
    _alpha(alpha) {}

Layer &Add::operator()(Value_t &input, Value_t &other) {
    Layer *l = (Layer *) new(std::nothrow) Add_L(_alpha, this);
    EXIT_ERROR_CHECK_EQ(l, nullptr, "(new) heap allocation failed");
    dealParams(l);
    _layers.push_back(l);
    return l->link(input, other);
}

BiasAdd::BiasAdd(UINT size, INT dim):
    OpSignature(LayerType::BIASADD),
    _size(size),
    _dim(dim) {}

Layer &BiasAdd::operator()(Value_t &input) {
    Layer *l = (Layer *) new(std::nothrow) BiasAdd_L(_size, _dim, this);
    EXIT_ERROR_CHECK_EQ(l, nullptr, "(new) heap allocation failed");
    dealParams(l);
    _layers.push_back(l);
    return l->link(input);
}

MatMul::MatMul():
    OpSignature(LayerType::MATMUL) {}

Layer &MatMul::operator()(Value_t &input, Value_t &other) {
    Layer *l = (Layer *) new(std::nothrow) MatMul_L(this);
    EXIT_ERROR_CHECK_EQ(l, nullptr, "(new) heap allocation failed");
    dealParams(l);
    _layers.push_back(l);
    return l->link(input, other);
}

Concat::Concat(INT dim):
    OpSignature(LayerType::CONCAT),
    _dim(dim) {}

Layer &Concat::operator()(Value_t &value) {
    Layer *l = (Layer *) new(std::nothrow) Concat_L(_dim, this);
    EXIT_ERROR_CHECK_EQ(l, nullptr, "(new) heap allocation failed");
    dealParams(l);
    _layers.push_back(l);
    return l->link(value);
}

} // namespace core
} // namespace Kernel
