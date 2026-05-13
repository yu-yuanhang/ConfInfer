#ifndef __POOL_H_CA__
#define __POOL_H_CA__

#include <All.h>
#include <core/Layer.h>
#include <core/Param.h>

namespace Kernel {
namespace core {

class MaxPool2d;
class AvgPool2d;
class AdaptiveAvgPool2d;
class AdaptiveMaxPool2d;

class PoolNd_L:
virtual public Layer
{
friend class MaxPool2d;
friend class AvgPool2d;
protected:
    PoolNd_L() = delete;
    ~PoolNd_L();
    PoolNd_L(LayerType type, 
        const std::vector<UINT>& size, 
        const std::vector<UINT>& stride, 
        const std::vector<INT>&  padding, 
        const std::vector<UINT>& dilation, 
        BOOL returnIndices, BOOL ceilMode,
        BOOL countIncludePad, UINT divisorOverride,
        UINT SpatialDim,
        OpSignature *opSignature);

protected:
    void makeParams(Params *params) override;
    void makeOutputs() override;
    UINT calcWorkspaceSize() override;

public:
    UINT inChannels() const { return _inChannels; }
    UINT outChannels() const { return _outChannels; }
    const std::vector<UINT>& kernelSize() const { return _kernelSize; }
    const std::vector<UINT>& stride() const { return _stride; }
    const std::vector<INT>& padding() const { return _padding; }
    const std::vector<UINT>& dilation() const { return _dilation; }
    BOOL returnIndices() const { return _returnIndices; }
    BOOL ceilMode() const { return _ceilMode; }
    BOOL countIncludePad() const { return _countIncludePad; }
    UINT divisorOverride() const { return _divisorOverride; }
    UINT spatialDim() const { return _SpatialDim; }

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
    BOOL _countIncludePad;
    UINT _divisorOverride;

    UINT _SpatialDim;
};

class AdaptivePool2d_L:
virtual public Layer
{
friend class AdaptiveAvgPool2d;
friend class AdaptiveMaxPool2d;
protected:
    AdaptivePool2d_L() = delete;
    ~AdaptivePool2d_L();
    AdaptivePool2d_L(LayerType type,
        const std::vector<UINT>& outputSize,
        BOOL returnIndices,
        OpSignature *opSignature);

protected:
    void makeParams(Params *params) override;
    void makeOutputs() override;
    UINT calcWorkspaceSize() override;

public:
    const std::vector<UINT>& outputSize() const { return _outputSize; }
    BOOL returnIndices() const { return _returnIndices; }

private:
    std::vector<UINT> _outputSize;
    BOOL _returnIndices;
};

class MaxPool2d:
virtual public OpSignature {
public:
    MaxPool2d() = delete;
    ~MaxPool2d();
    // size: 池化窗口大小 kernel_size.
    // stride: 步长，默认等于 kernel_size.
    // padding: 边界填充.
    // dilation: 空洞池化参数.
    // return_indices: 是否额外返回最大值索引.
    // ceil_mode: 是否使用 ceil 计算输出尺寸.
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
                false, 0,
                _SpatialDim, this);
        EXIT_ERROR_CHECK_EQ(l, nullptr, "(new) heap allocation failed");
        dealParams(l);
        _layers.push_back(l);
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

class AvgPool2d:
virtual public OpSignature {
public:
    AvgPool2d() = delete;
    ~AvgPool2d();
    // size: 池化窗口大小 kernel_size.
    // stride: 步长，默认等于 kernel_size.
    // padding: 边界填充.
    // ceil_mode: 是否使用 ceil 计算输出尺寸.
    // count_include_pad: 均值时是否计入 padding.
    // divisor_override: 自定义除数，0 表示按默认规则.
    AvgPool2d(const vector<UINT> &size,
              const vector<UINT> &stride = vector<UINT>(),
              const vector<INT>  &padding = {0, 0},
              BOOL ceil_mode = false,
              BOOL count_include_pad = true,
              UINT divisor_override = 0);

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
                false, _ceilMode,
                _countIncludePad, _divisorOverride,
                _SpatialDim, this);
        EXIT_ERROR_CHECK_EQ(l, nullptr, "(new) heap allocation failed");
        dealParams(l);
        _layers.push_back(l);
        return l->link(value, rest...);
    }

private:
    std::vector<UINT> _size;
    std::vector<UINT> _stride;
    std::vector<INT>  _padding;
    std::vector<UINT> _dilation;
    BOOL _ceilMode;
    BOOL _countIncludePad;
    UINT _divisorOverride;

    static constexpr UINT _SpatialDim = 2;
};

class AdaptiveAvgPool2d:
virtual public OpSignature {
public:
    AdaptiveAvgPool2d() = delete;
    ~AdaptiveAvgPool2d();
    // output_size: 目标输出空间尺寸，支持 {H, W} 或 {S}.
    explicit AdaptiveAvgPool2d(const vector<UINT> &output_size);

    Layer &operator()() = delete;
    Layer &operator()(Value_t &value);

private:
    std::vector<UINT> _outputSize;
};

class AdaptiveMaxPool2d:
virtual public OpSignature {
public:
    AdaptiveMaxPool2d() = delete;
    ~AdaptiveMaxPool2d();
    // 用来支持 pool return_indices=true
    // output_size: 目标输出空间尺寸，支持 {H, W} 或 {S}.
    // return_indices: 是否额外返回最大值索引.
    AdaptiveMaxPool2d(const vector<UINT> &output_size,
                      BOOL return_indices = false);

    Layer &operator()() = delete;
    Layer &operator()(Value_t &value);

private:
    std::vector<UINT> _outputSize;
    BOOL _returnIndices;
};


} // namespace end of core
} // namespace end of Kernel 
#endif
