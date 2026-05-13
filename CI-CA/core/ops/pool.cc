#include <pool.h>

namespace Kernel {
namespace core {

PoolNd_L::~PoolNd_L() {}
MaxPool2d::~MaxPool2d() {}
AvgPool2d::~AvgPool2d() {}
AdaptivePool2d_L::~AdaptivePool2d_L() {}
AdaptiveAvgPool2d::~AdaptiveAvgPool2d() {}
AdaptiveMaxPool2d::~AdaptiveMaxPool2d() {}

PoolNd_L::PoolNd_L(LayerType type, 
            const std::vector<UINT>& size, 
            const std::vector<UINT>& stride, 
            const std::vector<INT>&  padding, 
            const std::vector<UINT>& dilation, 
            BOOL returnIndices, BOOL ceilMode,
            BOOL countIncludePad, UINT divisorOverride,
            UINT SpatialDim,
            OpSignature *opSignature):
    Layer(type, opSignature),
    _inChannels(INVALID_VALUE_U),
    _outChannels(INVALID_VALUE_U),
    _kernelSize(size), _stride(stride), _padding(padding), _dilation(dilation),
    _returnIndices(returnIndices), _ceilMode(ceilMode),
    _countIncludePad(countIncludePad), _divisorOverride(divisorOverride),
    _SpatialDim(SpatialDim)
{
    LogDebug("PoolNd_L(LayerType type, ...) : _id = %u", _id);
}

void PoolNd_L::makeOutputs() {
    EXIT_ERROR_CHECK_NE(
        1, _inputs.size(),
        "PoolNd only supports a single input Value"
    );
    Value_t* input = _inputs[0];
    EXIT_ERROR_CHECK_EQ(
        nullptr, input,
        "PoolNd input is nullptr"
    );

    const DataShape_t& inShape = input->data.shape;
    EXIT_ERROR_CHECK_NE(
        inShape.ndim,
        static_cast<uint32_t>(_SpatialDim + 2),
        "PoolNd expects NCHW input shape"
    );

    const UINT batch_axis = 0u;
    const UINT channel_axis = 1u;
    const UINT spatial_axis = 2u;

    EXIT_ERROR_CHECK_NE(
        _kernelSize.size(),
        static_cast<size_t>(_SpatialDim),
        "PoolNd kernel size dim mismatch"
    );

    const std::vector<UINT>& stride = _stride.empty() ? _kernelSize : _stride;
    EXIT_ERROR_CHECK_NE(
        stride.size(),
        static_cast<size_t>(_SpatialDim),
        "PoolNd stride dim mismatch"
    );

    EXIT_ERROR_CHECK_EQ(
        false,
        (_padding.size() == static_cast<size_t>(_SpatialDim)
            || _padding.size() == static_cast<size_t>(2 * _SpatialDim)),
        "PoolNd padding size must be SpatialDim or 2*SpatialDim"
    );

    DataShape_t outShape;
    outShape.ndim = inShape.ndim;
    outShape.size = 1;
    outShape.dims[batch_axis] = inShape.dims[batch_axis];
    outShape.size *= outShape.dims[batch_axis];
    outShape.dims[channel_axis] = inShape.dims[channel_axis];
    outShape.size *= outShape.dims[channel_axis];

    for (UINT d = 0; d < _SpatialDim; ++d) {
        const UINT I = inShape.dims[spatial_axis + d];
        const UINT K = _kernelSize[d];
        const UINT S = stride[d];
        const UINT D = (_dilation.empty() ? 1 : _dilation[d]);

        EXIT_ERROR_CHECK_EQ(K, 0, "PoolNd kernel size must be > 0");
        EXIT_ERROR_CHECK_EQ(S, 0, "PoolNd stride must be > 0");

        INT pad_l = 0;
        INT pad_r = 0;
        if (_padding.size() == static_cast<size_t>(_SpatialDim)) {
            pad_l = _padding[d];
            pad_r = _padding[d];
        } else {
            pad_l = _padding[2 * d + 0];
            pad_r = _padding[2 * d + 1];
        }

        const INT numer = static_cast<INT>(I) + pad_l + pad_r
            - static_cast<INT>(D) * (static_cast<INT>(K) - 1) - 1;
        EXIT_ERROR_CHECK_NE(true, numer >= 0, "PoolNd invalid output numerator");

        INT O = numer / static_cast<INT>(S) + 1;
        if (_ceilMode) {
            O = (numer + static_cast<INT>(S) - 1) / static_cast<INT>(S) + 1;
        }
        EXIT_ERROR_CHECK_NE(true, O > 0, "PoolNd output size <= 0");

        outShape.dims[spatial_axis + d] = static_cast<UINT>(O);
        outShape.size *= static_cast<UINT>(O);
    }

    _outputs.clear();
    _outputs.reserve(_returnIndices ? 2 : 1);

    std::unique_ptr<Value_t> out =
        std::make_unique<Value_t>(PARAM_INTERMEDIATE | PARAM_OWN_DATA);
    out->data.shape = outShape;
    out->data.copyTypeFrom(input->data);
    out->data.flags = PARAM_INTERMEDIATE | PARAM_OWN_DATA;
    out->data.ptr = new(std::nothrow) char[out->data.shape.size * out->data.getTypeSize()];
    EXIT_ERROR_CHECK_EQ(nullptr, out->data.ptr, "PoolNd output allocation failed");
    out->producer = this;
    out->output_index = 0;
    out->kind = OutputKind::Default;
    _outputs.push_back(std::move(out));

    if (_returnIndices) {
        std::unique_ptr<Value_t> indices =
            std::make_unique<Value_t>(PARAM_INTERMEDIATE | PARAM_OWN_DATA);
        indices->data.shape = outShape;
        indices->data.dtype = DataType::INT32;
        indices->data.location = input->data.location;
        indices->data.flags = PARAM_INTERMEDIATE | PARAM_OWN_DATA;
        indices->data.ptr = new(std::nothrow) char[indices->data.shape.size * indices->data.getTypeSize()];
        EXIT_ERROR_CHECK_EQ(nullptr, indices->data.ptr, "PoolNd indices allocation failed");
        indices->producer = this;
        indices->output_index = 1;
        indices->kind = OutputKind::Indices;
        _outputs.push_back(std::move(indices));
    }
}

UINT PoolNd_L::calcWorkspaceSize() { return 0; }
void PoolNd_L::makeParams(Params *params) { return; }

AdaptivePool2d_L::AdaptivePool2d_L(LayerType type,
            const std::vector<UINT>& outputSize,
            BOOL returnIndices,
            OpSignature *opSignature):
    Layer(type, opSignature),
    _outputSize(outputSize),
    _returnIndices(returnIndices) {}

void AdaptivePool2d_L::makeParams(Params *params) { (void)params; }

void AdaptivePool2d_L::makeOutputs() {
    EXIT_ERROR_CHECK_NE(1, _inputs.size(), "AdaptivePool2d only supports a single input Value");
    Value_t* input = _inputs[0];
    EXIT_ERROR_CHECK_EQ(nullptr, input, "AdaptivePool2d input is nullptr");
    const DataShape_t& inShape = input->data.shape;
    EXIT_ERROR_CHECK_NE(4, inShape.ndim, "AdaptivePool2d expects NCHW input shape");
    EXIT_ERROR_CHECK_EQ(false, _outputSize.size() == 1 || _outputSize.size() == 2,
        "AdaptivePool2d output_size must have 1 or 2 elements");

    _outputs.clear();
    _outputs.reserve(_returnIndices ? 2 : 1);

    std::unique_ptr<Value_t> out =
        std::make_unique<Value_t>(PARAM_INTERMEDIATE | PARAM_OWN_DATA);
    out->copyDescFrom(*input);
    DataShape_t& shape = out->data.shape;
    shape.dims[2] = _outputSize[0];
    shape.dims[3] = (_outputSize.size() == 1) ? _outputSize[0] : _outputSize[1];
    shape.size = shape.dims[0] * shape.dims[1] * shape.dims[2] * shape.dims[3];
    out->data.flags = PARAM_INTERMEDIATE | PARAM_OWN_DATA;
    out->data.ptr = new(std::nothrow) char[shape.size * out->data.getTypeSize()];
    EXIT_ERROR_CHECK_EQ(nullptr, out->data.ptr, "AdaptivePool2d output allocation failed");
    out->producer = this;
    out->output_index = 0;
    out->kind = OutputKind::Default;
    _outputs.push_back(std::move(out));

    if (_returnIndices) {
        std::unique_ptr<Value_t> indices =
            std::make_unique<Value_t>(PARAM_INTERMEDIATE | PARAM_OWN_DATA);
        indices->data.shape = shape;
        indices->data.dtype = DataType::INT32;
        indices->data.location = input->data.location;
        indices->data.flags = PARAM_INTERMEDIATE | PARAM_OWN_DATA;
        indices->data.ptr = new(std::nothrow) char[shape.size * indices->data.getTypeSize()];
        EXIT_ERROR_CHECK_EQ(nullptr, indices->data.ptr, "AdaptivePool2d indices allocation failed");
        indices->producer = this;
        indices->output_index = 1;
        indices->kind = OutputKind::Indices;
        _outputs.push_back(std::move(indices));
    }
}

UINT AdaptivePool2d_L::calcWorkspaceSize() { return 0; }

MaxPool2d::MaxPool2d(const vector<UINT> &size,
                     const vector<UINT> &stride,
                     const vector<INT>  &padding,
                     const vector<UINT> &dilation,
                     BOOL return_indices,
                     BOOL ceil_mode):
    OpSignature(LayerType::MAXPOOL2D),
    _size(size), _stride(stride), _padding(padding), _dilation(dilation),
    _returnIndices(return_indices), _ceilMode(ceil_mode)
{
    LogDebug("MaxPool2d(&size, ...)");
}

Layer &MaxPool2d::operator()(Value_t &value) {
    // 调用具体的计算 表示一个具体的计算节点
    // 创建 Layer
    Layer *l = (Layer *) new(std::nothrow) PoolNd_L(_type, 
            _size, _stride, _padding, _dilation,
            _returnIndices, _ceilMode,
            false, 0,
            _SpatialDim, this);
    EXIT_ERROR_CHECK_EQ(l, nullptr, "(new) heap allocation failed");
    dealParams(l);
    _layers.push_back(l);
    return l->link(value);
}

AvgPool2d::AvgPool2d(const vector<UINT> &size,
                     const vector<UINT> &stride,
                     const vector<INT>  &padding,
                     BOOL ceil_mode,
                     BOOL count_include_pad,
                     UINT divisor_override):
    OpSignature(LayerType::AVGPOOL2D),
    _size(size), _stride(stride), _padding(padding), _dilation({1, 1}),
    _ceilMode(ceil_mode), _countIncludePad(count_include_pad),
    _divisorOverride(divisor_override)
{
    LogDebug("AvgPool2d(&size, ...)");
}

Layer &AvgPool2d::operator()(Value_t &value) {
    Layer *l = (Layer *) new(std::nothrow) PoolNd_L(_type,
            _size, _stride, _padding, _dilation,
            false, _ceilMode,
            _countIncludePad, _divisorOverride,
            _SpatialDim, this);
    EXIT_ERROR_CHECK_EQ(l, nullptr, "(new) heap allocation failed");
    dealParams(l);
    _layers.push_back(l);
    return l->link(value);
}

AdaptiveAvgPool2d::AdaptiveAvgPool2d(const vector<UINT> &output_size):
    OpSignature(LayerType::ADAPTIVEAVGPOOL2D),
    _outputSize(output_size) {}

Layer &AdaptiveAvgPool2d::operator()(Value_t &value) {
    Layer *l = (Layer *) new(std::nothrow) AdaptivePool2d_L(
        _type, _outputSize, false, this);
    EXIT_ERROR_CHECK_EQ(l, nullptr, "(new) heap allocation failed");
    dealParams(l);
    _layers.push_back(l);
    return l->link(value);
}

AdaptiveMaxPool2d::AdaptiveMaxPool2d(const vector<UINT> &output_size,
                                     BOOL return_indices):
    OpSignature(LayerType::ADAPTIVEMAXPOOL2D),
    _outputSize(output_size),
    _returnIndices(return_indices) {}

Layer &AdaptiveMaxPool2d::operator()(Value_t &value) {
    Layer *l = (Layer *) new(std::nothrow) AdaptivePool2d_L(
        _type, _outputSize, _returnIndices, this);
    EXIT_ERROR_CHECK_EQ(l, nullptr, "(new) heap allocation failed");
    dealParams(l);
    _layers.push_back(l);
    return l->link(value);
}

} // namespace end of core
} // namespace end of Kernel 
