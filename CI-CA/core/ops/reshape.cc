#include <reshape.h>

namespace Kernel {
namespace core {

namespace {
UINT normalize_dim(INT dim, UINT ndim, const char* name) {
    INT normalized = dim;
    if (normalized < 0) {
        normalized += static_cast<INT>(ndim);
    }
    EXIT_ERROR_CHECK_EQ(false,
        normalized >= 0 && normalized < static_cast<INT>(ndim),
        "%s dim out of range", name);
    return static_cast<UINT>(normalized);
}
} // namespace

Flatten_L::~Flatten_L() {}
Flatten::~Flatten() {}

Flatten_L::Flatten_L(INT startDim,
                     INT endDim,
                     OpSignature *opSignature):
    Layer(LayerType::FLATTEN, opSignature),
    _startDim(startDim),
    _endDim(endDim) {}

void Flatten_L::makeParams(Params *params) {
    (void)params;
}

void Flatten_L::makeOutputs() {
    EXIT_ERROR_CHECK_NE(1, _inputs.size(), "Flatten only supports one input");
    Value_t* input = _inputs[0];
    EXIT_ERROR_CHECK_EQ(nullptr, input, "Flatten input is nullptr");

    const DataShape_t& in_shape = input->data.shape;
    EXIT_ERROR_CHECK_EQ(0, in_shape.ndim, "Flatten input ndim must be > 0");

    const UINT start_dim = normalize_dim(_startDim, in_shape.ndim, "Flatten start");
    const UINT end_dim = normalize_dim(_endDim, in_shape.ndim, "Flatten end");
    EXIT_ERROR_CHECK_EQ(false, start_dim <= end_dim, "Flatten start_dim must be <= end_dim");

    std::unique_ptr<Value_t> out =
        std::make_unique<Value_t>(PARAM_INTERMEDIATE | PARAM_OWN_DATA);
    out->copyDescFrom(*input);
    out->data.flags = PARAM_INTERMEDIATE | PARAM_OWN_DATA;

    DataShape_t& out_shape = out->data.shape;
    UINT flattened = 1;
    for (UINT d = start_dim; d <= end_dim; ++d) {
        flattened *= in_shape.dims[d];
    }

    out_shape.ndim = in_shape.ndim - (end_dim - start_dim);
    UINT out_idx = 0;
    for (UINT d = 0; d < start_dim; ++d) {
        out_shape.dims[out_idx++] = in_shape.dims[d];
    }
    out_shape.dims[out_idx++] = flattened;
    for (UINT d = end_dim + 1; d < in_shape.ndim; ++d) {
        out_shape.dims[out_idx++] = in_shape.dims[d];
    }
    out_shape.size = in_shape.size;

    out->data.ptr = new(std::nothrow) char[out_shape.size * out->data.getTypeSize()];
    EXIT_ERROR_CHECK_EQ(nullptr, out->data.ptr, "Flatten output allocation failed");
    out->producer = this;
    out->output_index = 0;
    out->kind = OutputKind::Default;

    _outputs.clear();
    _outputs.reserve(1);
    _outputs.push_back(std::move(out));
}

UINT Flatten_L::calcWorkspaceSize() { return 0; }

Flatten::Flatten(INT start_dim, INT end_dim):
    OpSignature(LayerType::FLATTEN),
    _startDim(start_dim),
    _endDim(end_dim) {}

Layer &Flatten::operator()(Value_t &value) {
    Layer *l = (Layer *) new(std::nothrow) Flatten_L(_startDim, _endDim, this);
    EXIT_ERROR_CHECK_EQ(l, nullptr, "(new) heap allocation failed");
    dealParams(l);
    _layers.push_back(l);
    return l->link(value);
}

} // namespace core
} // namespace Kernel
