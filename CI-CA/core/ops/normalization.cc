#include <normalization.h>

#include <cstring>

namespace Kernel {
namespace core {

namespace {
void init_fill(Data_t* data, FLOAT value) {
    EXIT_ERROR_CHECK_EQ(nullptr, data, "Data_t is nullptr");
    EXIT_ERROR_CHECK_EQ(0, data->shape.size, "Param shape size is zero");

    data->ptr = new(std::nothrow) char[data->shape.size * data->getTypeSize()];
    EXIT_ERROR_CHECK_EQ(nullptr, data->ptr, "Parameter allocation failed");
    FLOAT* ptr = static_cast<FLOAT*>(data->ptr);
    for (UINT i = 0; i < data->shape.size; ++i) {
        ptr[i] = value;
    }
}
} // namespace

BatchNorm2d_L::~BatchNorm2d_L() {}
LayerNorm_L::~LayerNorm_L() {}
GroupNorm_L::~GroupNorm_L() {}
BatchNorm2d::~BatchNorm2d() {}
LayerNorm::~LayerNorm() {}
GroupNorm::~GroupNorm() {}

BatchNorm2d_L::BatchNorm2d_L(UINT numFeatures,
                             FLOAT eps,
                             FLOAT momentum,
                             BOOL affine,
                             BOOL trackRunningStats,
                             OpSignature *opSignature):
    Layer(LayerType::BATCHNORM2D, opSignature),
    _numFeatures(numFeatures),
    _eps(eps),
    _momentum(momentum),
    _affine(affine),
    _trackRunningStats(trackRunningStats) {}

void BatchNorm2d_L::makeParams(Params *params) {
    EXIT_ERROR_CHECK_EQ(nullptr, params, "Params is nullptr");

    if (_affine) {
        Data_t* weight = new(std::nothrow) Data_t(PARAM_CONST | PARAM_OWN_DATA, {_numFeatures});
        EXIT_ERROR_CHECK_EQ(nullptr, weight, "BatchNorm2d weight alloc failed");
        init_fill(weight, 1.0f);
        params->insert(ParamRole::WEIGHT, weight);

        Data_t* bias = new(std::nothrow) Data_t(PARAM_CONST | PARAM_OWN_DATA, {_numFeatures});
        EXIT_ERROR_CHECK_EQ(nullptr, bias, "BatchNorm2d bias alloc failed");
        init_fill(bias, 0.0f);
        params->insert(ParamRole::BIAS, bias);
    }

    if (_trackRunningStats) {
        Data_t* running_mean = new(std::nothrow) Data_t(PARAM_CONST | PARAM_OWN_DATA, {_numFeatures});
        EXIT_ERROR_CHECK_EQ(nullptr, running_mean, "BatchNorm2d running_mean alloc failed");
        init_fill(running_mean, 0.0f);
        params->insert(ParamRole::RUNNING_MEAN, running_mean);

        Data_t* running_var = new(std::nothrow) Data_t(PARAM_CONST | PARAM_OWN_DATA, {_numFeatures});
        EXIT_ERROR_CHECK_EQ(nullptr, running_var, "BatchNorm2d running_var alloc failed");
        init_fill(running_var, 1.0f);
        params->insert(ParamRole::RUNNING_VAR, running_var);
    }
}

void BatchNorm2d_L::makeOutputs() {
    Value_t* input = _inputs[0];
    EXIT_ERROR_CHECK_EQ(nullptr, input, "BatchNorm2d input is nullptr");
    EXIT_ERROR_CHECK_NE(4, input->data.shape.ndim, "BatchNorm2d expects NCHW input");
    EXIT_ERROR_CHECK_NE(_numFeatures, input->data.shape.dims[1],
        "BatchNorm2d num_features mismatch");

    std::unique_ptr<Value_t> out =
        std::make_unique<Value_t>(PARAM_INTERMEDIATE | PARAM_OWN_DATA);
    out->copyDescFrom(*input);
    out->data.flags = PARAM_INTERMEDIATE | PARAM_OWN_DATA;
    out->data.ptr = new(std::nothrow) char[out->data.shape.size * out->data.getTypeSize()];
    EXIT_ERROR_CHECK_EQ(nullptr, out->data.ptr, "BatchNorm2d output allocation failed");
    out->producer = this;
    out->output_index = 0;
    out->kind = OutputKind::Default;

    _outputs.clear();
    _outputs.reserve(1);
    _outputs.push_back(std::move(out));
}

UINT BatchNorm2d_L::calcWorkspaceSize() { return 0; }

LayerNorm_L::LayerNorm_L(const std::vector<UINT>& normalizedShape,
                         FLOAT eps,
                         BOOL elementwiseAffine,
                         OpSignature *opSignature):
    Layer(LayerType::LAYERNORM, opSignature),
    _normalizedShape(normalizedShape),
    _eps(eps),
    _elementwiseAffine(elementwiseAffine) {}

void LayerNorm_L::makeParams(Params *params) {
    EXIT_ERROR_CHECK_EQ(nullptr, params, "Params is nullptr");
    if (!_elementwiseAffine) { return; }

    Data_t* weight = new(std::nothrow) Data_t(PARAM_CONST | PARAM_OWN_DATA);
    EXIT_ERROR_CHECK_EQ(nullptr, weight, "LayerNorm weight alloc failed");
    weight->shape = DataShape_t{};
    weight->shape.ndim = static_cast<UINT>(_normalizedShape.size());
    weight->shape.size = 1;
    for (UINT i = 0; i < _normalizedShape.size(); ++i) {
        weight->shape.dims[i] = _normalizedShape[i];
        weight->shape.size *= _normalizedShape[i];
    }
    init_fill(weight, 1.0f);
    params->insert(ParamRole::WEIGHT, weight);

    Data_t* bias = new(std::nothrow) Data_t(PARAM_CONST | PARAM_OWN_DATA);
    EXIT_ERROR_CHECK_EQ(nullptr, bias, "LayerNorm bias alloc failed");
    bias->shape = weight->shape;
    init_fill(bias, 0.0f);
    params->insert(ParamRole::BIAS, bias);
}

void LayerNorm_L::makeOutputs() {
    Value_t* input = _inputs[0];
    EXIT_ERROR_CHECK_EQ(nullptr, input, "LayerNorm input is nullptr");
    EXIT_ERROR_CHECK_EQ(true, _normalizedShape.empty(), "LayerNorm normalized_shape must not be empty");
    EXIT_ERROR_CHECK_NE(true, input->data.shape.ndim >= _normalizedShape.size(),
        "LayerNorm input ndim must be >= normalized_shape size");

    UINT offset = input->data.shape.ndim - static_cast<UINT>(_normalizedShape.size());
    for (UINT i = 0; i < _normalizedShape.size(); ++i) {
        EXIT_ERROR_CHECK_NE(_normalizedShape[i], input->data.shape.dims[offset + i],
            "LayerNorm normalized_shape mismatch");
    }

    std::unique_ptr<Value_t> out =
        std::make_unique<Value_t>(PARAM_INTERMEDIATE | PARAM_OWN_DATA);
    out->copyDescFrom(*input);
    out->data.flags = PARAM_INTERMEDIATE | PARAM_OWN_DATA;
    out->data.ptr = new(std::nothrow) char[out->data.shape.size * out->data.getTypeSize()];
    EXIT_ERROR_CHECK_EQ(nullptr, out->data.ptr, "LayerNorm output allocation failed");
    out->producer = this;
    out->output_index = 0;
    out->kind = OutputKind::Default;

    _outputs.clear();
    _outputs.reserve(1);
    _outputs.push_back(std::move(out));
}

UINT LayerNorm_L::calcWorkspaceSize() { return 0; }

GroupNorm_L::GroupNorm_L(UINT numGroups,
                         UINT numChannels,
                         FLOAT eps,
                         BOOL affine,
                         OpSignature *opSignature):
    Layer(LayerType::GROUPNORM, opSignature),
    _numGroups(numGroups),
    _numChannels(numChannels),
    _eps(eps),
    _affine(affine) {}

void GroupNorm_L::makeParams(Params *params) {
    EXIT_ERROR_CHECK_EQ(nullptr, params, "Params is nullptr");
    if (!_affine) { return; }

    Data_t* weight = new(std::nothrow) Data_t(PARAM_CONST | PARAM_OWN_DATA, {_numChannels});
    EXIT_ERROR_CHECK_EQ(nullptr, weight, "GroupNorm weight alloc failed");
    init_fill(weight, 1.0f);
    params->insert(ParamRole::WEIGHT, weight);

    Data_t* bias = new(std::nothrow) Data_t(PARAM_CONST | PARAM_OWN_DATA, {_numChannels});
    EXIT_ERROR_CHECK_EQ(nullptr, bias, "GroupNorm bias alloc failed");
    init_fill(bias, 0.0f);
    params->insert(ParamRole::BIAS, bias);
}

void GroupNorm_L::makeOutputs() {
    Value_t* input = _inputs[0];
    EXIT_ERROR_CHECK_EQ(nullptr, input, "GroupNorm input is nullptr");
    EXIT_ERROR_CHECK_EQ(_numGroups, 0, "GroupNorm num_groups must be > 0");
    EXIT_ERROR_CHECK_EQ(_numChannels, 0, "GroupNorm num_channels must be > 0");
    EXIT_ERROR_CHECK_NE(_numChannels % _numGroups, 0, "GroupNorm num_channels must be divisible by num_groups");
    EXIT_ERROR_CHECK_EQ(false,
        input->data.shape.ndim >= 3,
        "GroupNorm expects input shape [N,C,*]");
    EXIT_ERROR_CHECK_NE(_numChannels, input->data.shape.dims[1],
        "GroupNorm num_channels mismatch");

    std::unique_ptr<Value_t> out =
        std::make_unique<Value_t>(PARAM_INTERMEDIATE | PARAM_OWN_DATA);
    out->copyDescFrom(*input);
    out->data.flags = PARAM_INTERMEDIATE | PARAM_OWN_DATA;
    out->data.ptr = new(std::nothrow) char[out->data.shape.size * out->data.getTypeSize()];
    EXIT_ERROR_CHECK_EQ(nullptr, out->data.ptr, "GroupNorm output allocation failed");
    out->producer = this;
    out->output_index = 0;
    out->kind = OutputKind::Default;

    _outputs.clear();
    _outputs.reserve(1);
    _outputs.push_back(std::move(out));
}

UINT GroupNorm_L::calcWorkspaceSize() { return 0; }

BatchNorm2d::BatchNorm2d(UINT num_features,
                         FLOAT eps,
                         FLOAT momentum,
                         BOOL affine,
                         BOOL track_running_stats):
    OpSignature(LayerType::BATCHNORM2D),
    _numFeatures(num_features),
    _eps(eps),
    _momentum(momentum),
    _affine(affine),
    _trackRunningStats(track_running_stats) {}

Layer &BatchNorm2d::operator()(Value_t &value) {
    Layer *l = (Layer *) new(std::nothrow) BatchNorm2d_L(
        _numFeatures, _eps, _momentum, _affine, _trackRunningStats, this);
    EXIT_ERROR_CHECK_EQ(l, nullptr, "(new) heap allocation failed");
    dealParams(l);
    _layers.push_back(l);
    return l->link(value);
}

LayerNorm::LayerNorm(const std::vector<UINT>& normalized_shape,
                     FLOAT eps,
                     BOOL elementwise_affine):
    OpSignature(LayerType::LAYERNORM),
    _normalizedShape(normalized_shape),
    _eps(eps),
    _elementwiseAffine(elementwise_affine) {}

Layer &LayerNorm::operator()(Value_t &value) {
    Layer *l = (Layer *) new(std::nothrow) LayerNorm_L(
        _normalizedShape, _eps, _elementwiseAffine, this);
    EXIT_ERROR_CHECK_EQ(l, nullptr, "(new) heap allocation failed");
    dealParams(l);
    _layers.push_back(l);
    return l->link(value);
}

GroupNorm::GroupNorm(UINT num_groups,
                     UINT num_channels,
                     FLOAT eps,
                     BOOL affine):
    OpSignature(LayerType::GROUPNORM),
    _numGroups(num_groups),
    _numChannels(num_channels),
    _eps(eps),
    _affine(affine) {}

Layer &GroupNorm::operator()(Value_t &value) {
    Layer *l = (Layer *) new(std::nothrow) GroupNorm_L(
        _numGroups, _numChannels, _eps, _affine, this);
    EXIT_ERROR_CHECK_EQ(l, nullptr, "(new) heap allocation failed");
    dealParams(l);
    _layers.push_back(l);
    return l->link(value);
}

} // namespace core
} // namespace Kernel
