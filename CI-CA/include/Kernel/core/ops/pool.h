#ifndef __POOL_H_CA__
#define __POOL_H_CA__

#include <All.h>
#include <core/Layer.h>
#include <core/Param.h>

namespace Kernel {
namespace core {

class MaxPool2d;

class PoolNd_L:
virtual public Layer
{
friend class MaxPool2d;
protected:
    PoolNd_L() = delete;
    ~PoolNd_L();
    PoolNd_L(LayerType type, 
        const std::vector<UINT>& size, 
        const std::vector<UINT>& stride, 
        const std::vector<INT>&  padding, 
        const std::vector<UINT>& dilation, 
        BOOL returnIndices, BOOL ceilMode,
        UINT SpatialDim,
        OpSignature *opSignature);

protected:
    void makeParams(Params *params) override;
    void makeOutputs() override;
    UINT calcWorkspaceSize() override;

protected:
    /*
     * 输入通道的确定会影响到 Pool Layer 的适用性
     * 但是从计算图结构完整性的角度上讲 这些也是无法省略的部分
     * 因为 继承自 Layer
     * 而 Layer 的设计又是与计算图强绑定的 
     * 如果需要实现网络定义的层 在语义上可复用 同时保持计算图结构完整
     * 至少需要和 Layer 解耦
     */ 
    UINT _inChannels;          // 输入通道数
    UINT _outChannels;         // 输出通道数 (一般与输入通道相同)

    std::vector<UINT> _kernelSize;
    std::vector<UINT> _stride;
    std::vector<INT>  _padding;
    std::vector<UINT> _dilation;

    BOOL _returnIndices;
    BOOL _ceilMode;

    UINT _SpatialDim;
};

class MaxPool2d:
virtual public OpSignature {
public:
    MaxPool2d() = delete;
    ~MaxPool2d();
    MaxPool2d(const vector<UINT> &size,
              // 若不指定则自动等于 kernel_size
              const vector<UINT> &stride = vector<UINT>(),
              const vector<INT>  &padding = {0, 0},
              /* dilation: _size_2_t = 1, */
              const vector<UINT> &dilation = {1, 1},
              // 是否返回最大值的索引
              BOOL return_indices = false,
              // 是否使用 ceil 来计算输出尺寸
              BOOL ceil_mode = false);

    Layer &operator()() = delete;
    Layer &operator()(Value_t &value);
    template<typename... Args>
    Layer &operator()(Value_t &value, Args &... rest) {
        static_assert(
            (std::is_same_v<Value_t, std::remove_reference_t<Args>> && ...),
            "Layer::operator() only accepts Value_t& arguments"
        );
        Layer *l = (Layer *) new(std::nothrow) PoolNd_L(_type, 
                _size, _stride, _padding, _dilation,
                _returnIndices, _ceilMode,
                _SpatialDim, this);
        EXIT_ERROR_CHECK_EQ(l, nullptr, "(new) heap allocation failed");
        _layers.push_back(l);
        dealParams(l);
        return l->link(value, rest...);
    }
private:
    std::vector<UINT> _size;        // 池化窗口大小 (ndim 长度)
    std::vector<UINT> _stride;      // 步长 (默认等于 kernel_size)
    std::vector<INT>  _padding;     // 边界填充 (每个维度)
    std::vector<UINT> _dilation;    // 空洞参数 (通常=1)
    BOOL _returnIndices;            // 是否返回最大值索引 (MaxPool 用)
    BOOL _ceilMode;                 // 是否使用 ceil 计算输出尺寸

    static constexpr UINT _SpatialDim = 2;
};

/*
class AvgPool2d:
public PoolNd_L {
public:
    AvgPool2d() = delete;
    ~AvgPool2d();

};
*/


} // namespace end of core
} // namespace end of Kernel 
#endif