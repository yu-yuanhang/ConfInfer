#include <linear.h>

namespace Kernel {
namespace core {

Linear_L::~Linear_L() {}
Linear::~Linear() {}

Linear_L::Linear_L(UINT inFeatures,
                   UINT outFeatures,
                   BOOL bias,
                   OpSignature *opSignature):
    Layer(LayerType::LINEAR, opSignature),
    _inFeatures(inFeatures),
    _outFeatures(outFeatures),
    _bias(bias) {}

void Linear_L::makeParams(Params *params) {
    EXIT_ERROR_CHECK_EQ(nullptr, params, "Params is nullptr");

    Data_t* weight = new(std::nothrow) Data_t(PARAM_CONST | PARAM_OWN_DATA, {_outFeatures, _inFeatures});
    EXIT_ERROR_CHECK_EQ(nullptr, weight, "Linear weight alloc failed");
    weight->ptr = new(std::nothrow) char[weight->shape.size * weight->getTypeSize()];
    EXIT_ERROR_CHECK_EQ(nullptr, weight->ptr, "Linear weight buffer alloc failed");
    fill_random(weight->ptr, weight->dtype, weight->shape.size, TIMESEED);
    params->insert(ParamRole::WEIGHT, weight);

    if (_bias) {
        Data_t* bias = new(std::nothrow) Data_t(PARAM_CONST | PARAM_OWN_DATA, {_outFeatures});
        EXIT_ERROR_CHECK_EQ(nullptr, bias, "Linear bias alloc failed");
        bias->ptr = new(std::nothrow) char[bias->shape.size * bias->getTypeSize()];
        EXIT_ERROR_CHECK_EQ(nullptr, bias->ptr, "Linear bias buffer alloc failed");
        fill_random(bias->ptr, bias->dtype, bias->shape.size, TIMESEED);
        params->insert(ParamRole::BIAS, bias);
    }
}

void Linear_L::makeOutputs() {
    EXIT_ERROR_CHECK_NE(1, _inputs.size(), "Linear only supports one input");
    Value_t* input = _inputs[0];
    EXIT_ERROR_CHECK_EQ(nullptr, input, "Linear input is nullptr");
    EXIT_ERROR_CHECK_EQ(false, input->data.shape.ndim >= 2, "Linear input ndim must be >= 2");
    EXIT_ERROR_CHECK_NE(_inFeatures, input->data.shape.dims[input->data.shape.ndim - 1],
        "Linear in_features mismatch");

    std::unique_ptr<Value_t> out =
        std::make_unique<Value_t>(PARAM_INTERMEDIATE | PARAM_OWN_DATA);
    out->copyDescFrom(*input);
    out->data.shape.dims[out->data.shape.ndim - 1] = _outFeatures;
    out->data.shape.size = input->data.shape.size / _inFeatures * _outFeatures;
    out->data.flags = PARAM_INTERMEDIATE | PARAM_OWN_DATA;
    out->data.ptr = new(std::nothrow) char[out->data.shape.size * out->data.getTypeSize()];
    EXIT_ERROR_CHECK_EQ(nullptr, out->data.ptr, "Linear output allocation failed");
    out->producer = this;
    out->output_index = 0;
    out->kind = OutputKind::Default;

    _outputs.clear();
    _outputs.reserve(1);
    _outputs.push_back(std::move(out));
}

UINT Linear_L::calcWorkspaceSize() { return 0; }

Linear::Linear(UINT in_features,
               UINT out_features,
               BOOL bias):
    OpSignature(LayerType::LINEAR),
    _inFeatures(in_features),
    _outFeatures(out_features),
    _bias(bias) {}

Layer &Linear::operator()(Value_t &value) {
    Layer *l = (Layer *) new(std::nothrow) Linear_L(
        _inFeatures, _outFeatures, _bias, this);
    EXIT_ERROR_CHECK_EQ(l, nullptr, "(new) heap allocation failed");
    dealParams(l);
    _layers.push_back(l);
    return l->link(value);
}

} // namespace core
} // namespace Kernel
