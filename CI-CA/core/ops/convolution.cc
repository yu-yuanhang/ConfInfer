#include <convolution.h>

namespace Kernel {
namespace core {

namespace {

static void parse_conv_padding_2d(const std::vector<INT>& padding,
                                  UINT spatial_dim,
                                  UINT dim,
                                  INT& pad_l,
                                  INT& pad_r) {
    EXIT_ERROR_CHECK_NE(spatial_dim, 2, "ConvNd padding parser only supports 2D");
    EXIT_ERROR_CHECK_EQ(false,
        padding.size() == spatial_dim || padding.size() == 2 * spatial_dim,
        "ConvNd padding size must be SpatialDim or 2*SpatialDim");

    if (padding.size() == spatial_dim) {
        pad_l = padding[dim];
        pad_r = padding[dim];
        return;
    }

    pad_l = padding[2 * dim + 0];
    pad_r = padding[2 * dim + 1];
}

} // namespace

ConvNd_L::~ConvNd_L() {}
Conv2d::~Conv2d() {}

ConvNd_L::ConvNd_L(LayerType type,
            UINT channel, UINT num,
            const vector<UINT> &size, 
            const vector<UINT> &stride,
            const vector<INT>  &padding,
            const vector<UINT> &dilation,
            UINT groups,
            BOOL isBias,
            PADDING_MODE padding_mode,
            UINT SpatialDim,
            OpSignature *opSignature):
    Layer(type, opSignature),
    _inChannels(channel), _outChannels(num),
    _groups(groups), _isBias(isBias), _paddingMode(padding_mode),
    _kernelSize(size), _stride(stride), _padding(padding), _dilation(dilation),
    // _weights(PARAM_CONST | PARAM_OWN_DATA), _biases(PARAM_CONST | PARAM_OWN_DATA),
    _inChannelsPerGroup(channel / groups), 
    _outChannelsPerGroup(num / groups),
    _SpatialDim(SpatialDim)
{
    LogDebug("ConvNd_L(LayerType type, ...) : _id = %u", _id);
}

// 前提是 _weights/_biases _inputs 初始化完成
void ConvNd_L::makeOutputs() {
    EXIT_ERROR_CHECK_NE(
        1, _inputs.size(),
        "ConvNd only supports a single input Value"
    );
    Value_t* input = _inputs[0];
    EXIT_ERROR_CHECK_EQ(
        nullptr, input,
        "ConvNd input is nullptr"
    );

    const DataShape_t& inShape = input->data.shape;
    EXIT_ERROR_CHECK_NE(
        inShape.ndim,
        static_cast<uint32_t>(_SpatialDim + 2),
        "ConvNd expects NCHW input shape"
    );
    const UINT batch_axis = 0u;
    const UINT channel_axis = 1u;
    const UINT spatial_axis = 2u;

    // ConvNd 默认的输出 Value 个数为 1
    std::unique_ptr<Value_t> out = std::make_unique<Value_t>(PARAM_INTERMEDIATE | PARAM_OWN_DATA);
    
    DataShape_t &outShape = (out->data.shape);
    outShape.ndim = inShape.ndim;
    outShape.size = 1;
    outShape.dims[batch_axis] = inShape.dims[batch_axis];
    outShape.size *= outShape.dims[batch_axis];
    outShape.dims[channel_axis] = _outChannels;
    outShape.size *= _outChannels;
    for (UINT d = 0; d < _SpatialDim; ++d) {
        UINT I = inShape.dims[spatial_axis + d];
        UINT K = _kernelSize[d];
        UINT S = _stride[d];
        UINT D = _dilation[d];

        INT pad_l = 0;
        INT pad_r = 0;
        parse_conv_padding_2d(_padding, _SpatialDim, d, pad_l, pad_r);

        INT O = (INT)(
            (I + pad_l + pad_r
            - (INT)D * ((INT)K - 1) - 1)
            / (INT)S
        ) + 1;
        EXIT_ERROR_CHECK_NE(true, O > 0, "ConvNd output size <= 0");

        outShape.dims[spatial_axis + d] = (UINT)O;
        outShape.size *= (UINT)O;
    }

    // Param 初始化
    Data_t &data  = out->data;
    data.copyTypeFrom(input->data);         // 通常跟随输入
    // out->param.data     = nullptr;
    data.ptr = new(std::nothrow) char[data.shape.size * data.getTypeSize()];
    EXIT_ERROR_CHECK_EQ(nullptr, data.ptr, "ConvNd output allocation failed");

    // Value 元信息
    out->producer     = this;
    out->output_index = 0;
    out->kind         = OutputKind::Default;

    // Layer 作为计算图的基本组成 与 逻辑上的算子是解耦的 
    // 这里每个 Layer 都至少也仅仅有一次通过 link 调用 makeOutputs 的情况
    _outputs.clear();
    _outputs.reserve(1);
    _outputs.push_back(std::move(out));
    return;
}
UINT ConvNd_L::calcWorkspaceSize() {
    EXIT_ERROR_CHECK_NE(
        1, _inputs.size(),
        "ConvNd only supports a single input Value"
    );
    Value_t* input = _inputs[0];
    EXIT_ERROR_CHECK_EQ(nullptr, input, "ConvNd input is nullptr");

    const DataShape_t& inShape = input->data.shape;
    EXIT_ERROR_CHECK_NE(
        inShape.ndim,
        static_cast<uint32_t>(_SpatialDim + 2),
        "ConvNd expects NCHW input shape"
    );

    UINT out_spatial = 1;
    for (UINT d = 0; d < _SpatialDim; ++d) {
        const UINT I = inShape.dims[2 + d];
        const UINT K = _kernelSize[d];
        const UINT S = _stride[d];
        const UINT D = _dilation[d];
        INT pad_l = 0;
        INT pad_r = 0;
        parse_conv_padding_2d(_padding, _SpatialDim, d, pad_l, pad_r);

        const INT O = static_cast<INT>(
            (I + pad_l + pad_r - static_cast<INT>(D) * (static_cast<INT>(K) - 1) - 1)
            / static_cast<INT>(S)
        ) + 1;
        EXIT_ERROR_CHECK_NE(true, O > 0, "ConvNd output size <= 0");
        out_spatial *= static_cast<UINT>(O);
    }

    UINT kernel_area = 1;
    for (auto it = _kernelSize.begin(); it != _kernelSize.end(); ++it) {
        kernel_area *= *it;
    }
    return _inChannelsPerGroup * kernel_area * out_spatial * sizeof(FLOAT);
}
void ConvNd_L::makeParams(Params *params) {
    EXIT_ERROR_CHECK_EQ(nullptr, params, "Params_t *params == nullptr");

{   // ============================ WEIGHT 
    EXIT_ERROR_CHECK_EQ(_groups, 0, "groups must be > 0");
    EXIT_ERROR_CHECK_NE(_inChannels % _groups, 0, "channel must be divisible by groups");
    EXIT_ERROR_CHECK_NE(_outChannels % _groups, 0, "num must be divisible by groups");
    EXIT_ERROR_CHECK_EQ(_kernelSize.empty(), true, "kernel size must not be empty");
    
    Data_t* weight = new(std::nothrow) Data_t(PARAM_CONST | PARAM_OWN_DATA);
    // shape: [c_out, c_in / groups, h, w, ...]
    // 设置权重 shape
    weight->shape.ndim = 2 + _kernelSize.size();    // (_outChannels, in_channels_per_group, kernel_dims...)
    weight->shape.dims[0] = _outChannels;           // _outChannels
    weight->shape.dims[1] = _inChannels / _groups;  // _inChannels per group

    for (size_t i = 0; i < _kernelSize.size(); ++i) {
        EXIT_ERROR_CHECK_EQ(_kernelSize[i], 0, "kernel size must be > 0");
        weight->shape.dims[2 + i] = _kernelSize[i];
    }

    // 计算总 size
    weight->shape.size = 1;
    for (int i = 0; i < weight->shape.ndim; ++i) {
        weight->shape.size *= weight->shape.dims[i];
    }
    weight->ptr = new(std::nothrow) char[weight->shape.size * weight->getTypeSize()];
    fill_random(weight->ptr, weight->dtype, weight->shape.size, TIMESEED);

    params->insert(ParamRole::WEIGHT, weight);
}
{   // ============================ BIAS 
    EXIT_ERROR_CHECK_EQ(_outChannels, 0, "_outChannels must be > 0 for bias");
    if (_isBias) {
        Data_t* bias = new(std::nothrow) Data_t(PARAM_CONST | PARAM_OWN_DATA);
        bias->shape.ndim = 1;
        bias->shape.dims[0] = _outChannels; // bias 一维向量 长度 = out_channels
        bias->shape.size = _outChannels;

        bias->ptr = new(std::nothrow) char[bias->shape.size * bias->getTypeSize()];
        fill_random(bias->ptr, bias->dtype, bias->shape.size, TIMESEED);

        params->insert(ParamRole::BIAS, bias);
    }
}

    return;
}

Conv2d::Conv2d(UINT channel, UINT num,
        const vector<UINT> &size,  // kernel_size 不设置默认值  
        const vector<UINT> &stride,
        const vector<INT>  &padding,
        /* dilation: _size_2_t = 1, */
        const vector<UINT> &dilation,
        UINT groups,
        BOOL isBias,
        PADDING_MODE padding_mode
        /* device */
        /* dtype */):
    OpSignature(LayerType::CONV2D),
    _channel(channel), _num(num),
    _size(size), _stride(stride), _padding(padding), _dilation(dilation),
    _groups(groups), _isBias(isBias), _padding_mode(padding_mode)
{
    LogDebug("Conv2d(UINT channel, UINT num, ...)");
}

Layer &Conv2d::operator()(Value_t &value) {
    // 调用具体的计算 表示一个具体的计算节点
    // 创建 Layer
    Layer *l = (Layer *) new(std::nothrow) ConvNd_L(_type,
            _channel, _num,
            _size, _stride, _padding, _dilation,
            _groups, _isBias, _padding_mode,
            _SpatialDim, this);
    EXIT_ERROR_CHECK_EQ(l, nullptr, "(new) heap allocation failed");
    dealParams(l);
    _layers.push_back(l);
    return l->link(value);
}

} // namespace end of core
} // namespace end of Kernel 
