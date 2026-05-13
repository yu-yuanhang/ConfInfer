#ifndef __CONVOLUTION_H_CA__
#define __CONVOLUTION_H_CA__

#include <All.h>
#include <core/Layer.h>
#include <core/Param.h>

namespace Kernel {
namespace core {

class Conv2d;

class ConvNd_L: 
virtual public Layer
{
friend class Conv2d;
protected:
    ConvNd_L() = delete;
    ~ConvNd_L();
    ConvNd_L(LayerType type,
           UINT channel, UINT num,
           const vector<UINT> &size, 
           const vector<UINT> &stride,
           const vector<INT>  &padding,
           const vector<UINT> &dilation,
           UINT groups,
           BOOL isBias,
           PADDING_MODE padding_mode,
           /* device */
           /* dtype */
           UINT SpatialDim,
           OpSignature *opSignature);

protected:
    void makeParams(Params *params) override;
    void makeOutputs() override;
    UINT calcWorkspaceSize() override;

public:
    UINT inChannels() const { return _inChannels; }
    UINT outChannels() const { return _outChannels; }
    UINT groups() const { return _groups; }
    BOOL biasEnabled() const { return _isBias; }
    const std::vector<UINT>& kernelSize() const { return _kernelSize; }
    const std::vector<UINT>& stride() const { return _stride; }
    const std::vector<INT>& padding() const { return _padding; }
    const std::vector<UINT>& dilation() const { return _dilation; }
    UINT inChannelsPerGroup() const { return _inChannelsPerGroup; }
    UINT outChannelsPerGroup() const { return _outChannelsPerGroup; }
    UINT spatialDim() const { return _SpatialDim; }

protected:
    UINT _inChannels;
    UINT _outChannels;  // num 
    UINT _groups;
    BOOL _isBias;
    PADDING_MODE _paddingMode;

    // N 维参数
    std::vector<UINT> _kernelSize;
    std::vector<UINT> _stride;
    std::vector<INT>  _padding;
    std::vector<UINT> _dilation;

    UINT _inChannelsPerGroup;
    UINT _outChannelsPerGroup;

    UINT _SpatialDim;
};

// class Conv1d:

class Conv2d:
virtual public OpSignature {
public:
    Conv2d() = delete;
    ~Conv2d();
    // channel: 输入通道数.
    // num: 输出通道数.
    // size: 卷积核尺寸 kernel_size.
    // padding: 支持 {pad_h, pad_w} 或 {top, bottom, left, right}.
    // stride / padding / dilation / groups / isBias / padding_mode
    // 均与用户侧卷积配置直接对应.
    Conv2d(UINT channel, UINT num,
           const vector<UINT> &size,  // kernel_size 不设置默认值  
           const vector<UINT> &stride = {1, 1},
           // 据说某些场景下 padding < 0 来实现裁剪效果
           const vector<INT>  &padding = {0, 0},
           /* dilation: _size_2_t = 1, */
           const vector<UINT> &dilation = {1, 1},
           UINT groups = 1,
           BOOL isBias = true,
           PADDING_MODE padding_mode = ZEROS_PADDING
           /* device */
           /* dtype */);

    Layer &operator()() = delete;
    Layer &operator()(Value_t &value);
    template<typename... Args>
    Layer &operator()(Value_t &value, Args &... rest) {
        static_assert(
            (std::is_same_v<Value_t, std::remove_reference_t<Args>> && ...),
            "Layer::operator() only accepts Value_t& arguments"
        );
        Layer *l = (Layer *) new(std::nothrow) ConvNd_L(_type,
                _channel, _num,
                _size, _stride, _padding, _dilation,
                _groups, _isBias, _padding_mode,
                _SpatialDim, this);
        EXIT_ERROR_CHECK_EQ(l, nullptr, "(new) heap allocation failed");
        dealParams(l);
        _layers.push_back(l);
        return l->link(value, rest...);
    }
private:
    UINT _channel, _num;
    std::vector<UINT> _size;
    std::vector<UINT> _stride;
    std::vector<INT>  _padding;
    std::vector<UINT> _dilation;
    UINT _groups;
    BOOL _isBias;
    PADDING_MODE _padding_mode;

    static constexpr UINT _SpatialDim = 2;
};

// class Conv3d: 

} // namespace end of core
} // namespace end of Kernel 
#endif
