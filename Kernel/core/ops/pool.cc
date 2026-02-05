#include <pool.h>

namespace Kernel {
namespace core {

PoolNd_L::~PoolNd_L() {}
MaxPool2d::~MaxPool2d() {}

PoolNd_L::PoolNd_L(LayerType type, 
            const std::vector<UINT>& size, 
            const std::vector<UINT>& stride, 
            const std::vector<INT>&  padding, 
            const std::vector<UINT>& dilation, 
            BOOL returnIndices, BOOL ceilMode,
            UINT SpatialDim,
            OpSignature *opSignature):
    Layer(type, opSignature),
    _inChannels(INVALID_VALUE_U),
    _outChannels(INVALID_VALUE_U),
    _kernelSize(size), _stride(stride), _padding(padding), _dilation(dilation),
    _returnIndices(returnIndices), _ceilMode(ceilMode),
    _SpatialDim(SpatialDim)
{
    LogDebug("PoolNd_L(LayerType type, ...) : _id = %u", _id);
}

void PoolNd_L::makeOutputs() {
    
}
UINT PoolNd_L::calcWorkspaceSize() { return 0; }
void PoolNd_L::makeParams(Params *params) { return; }

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
            _SpatialDim, this);
    EXIT_ERROR_CHECK_EQ(l, nullptr, "(new) heap allocation failed");
    _layers.push_back(l);
    dealParams(l);
    return l->link(value);
}

} // namespace end of core
} // namespace end of Kernel 