#ifndef __NORMALIZATION_H_CA__
#define __NORMALIZATION_H_CA__

#include <All.h>
#include <core/Layer.h>
#include <core/Param.h>

namespace Kernel {
namespace core {

class BatchNorm2d;
class LayerNorm;
class GroupNorm;

class BatchNorm2d_L:
virtual public Layer
{
friend class BatchNorm2d;
public:
    UINT numFeatures() const { return _numFeatures; }
    FLOAT eps() const { return _eps; }
    BOOL affine() const { return _affine; }
    BOOL trackRunningStats() const { return _trackRunningStats; }

protected:
    BatchNorm2d_L() = delete;
    ~BatchNorm2d_L();
    BatchNorm2d_L(UINT numFeatures,
                  FLOAT eps,
                  FLOAT momentum,
                  BOOL affine,
                  BOOL trackRunningStats,
                  OpSignature *opSignature);

protected:
    void makeParams(Params *params) override;
    void makeOutputs() override;
    UINT calcWorkspaceSize() override;

protected:
    UINT _numFeatures;
    FLOAT _eps;
    FLOAT _momentum;
    BOOL _affine;
    BOOL _trackRunningStats;
};

class LayerNorm_L:
virtual public Layer
{
friend class LayerNorm;
public:
    const std::vector<UINT>& normalizedShape() const { return _normalizedShape; }
    FLOAT eps() const { return _eps; }
    BOOL elementwiseAffine() const { return _elementwiseAffine; }

protected:
    LayerNorm_L() = delete;
    ~LayerNorm_L();
    LayerNorm_L(const std::vector<UINT>& normalizedShape,
                FLOAT eps,
                BOOL elementwiseAffine,
                OpSignature *opSignature);

protected:
    void makeParams(Params *params) override;
    void makeOutputs() override;
    UINT calcWorkspaceSize() override;

protected:
    std::vector<UINT> _normalizedShape;
    FLOAT _eps;
    BOOL _elementwiseAffine;
};

class GroupNorm_L:
virtual public Layer
{
friend class GroupNorm;
public:
    UINT numGroups() const { return _numGroups; }
    UINT numChannels() const { return _numChannels; }
    FLOAT eps() const { return _eps; }
    BOOL affine() const { return _affine; }

protected:
    GroupNorm_L() = delete;
    ~GroupNorm_L();
    GroupNorm_L(UINT numGroups,
                UINT numChannels,
                FLOAT eps,
                BOOL affine,
                OpSignature *opSignature);

protected:
    void makeParams(Params *params) override;
    void makeOutputs() override;
    UINT calcWorkspaceSize() override;

protected:
    UINT _numGroups;
    UINT _numChannels;
    FLOAT _eps;
    BOOL _affine;
};

class BatchNorm2d:
virtual public OpSignature {
public:
    BatchNorm2d() = delete;
    ~BatchNorm2d();
    // (x - mean) / sqrt(var + eps) * gamma + beta
    // num_features: 特征通道数，按 NCHW 语义对应 C.
    // eps: 防止除零.
    // momentum: 训练阶段更新 running_mean / running_var 的动量系数.
    // affine: 是否具有可学习的 gamma / beta.
    // track_running_stats: 是否维护 running_mean / running_var.
    BatchNorm2d(UINT num_features,
                FLOAT eps = 1e-5f,
                FLOAT momentum = 0.1f,
                BOOL affine = true,
                BOOL track_running_stats = true);

    Layer &operator()() = delete;
    Layer &operator()(Value_t &value);

private:
    UINT _numFeatures;
    FLOAT _eps;
    FLOAT _momentum;
    BOOL _affine;
    BOOL _trackRunningStats;
};

class LayerNorm:
virtual public OpSignature {
public:
    LayerNorm() = delete;
    ~LayerNorm();
    // normalized_shape: 参与归一化的尾部维度形状.
    // elementwise_affine: 是否具有逐元素 gamma / beta.
    LayerNorm(const std::vector<UINT>& normalized_shape,
              FLOAT eps = 1e-5f,
              BOOL elementwise_affine = true);

    Layer &operator()() = delete;
    Layer &operator()(Value_t &value);

private:
    std::vector<UINT> _normalizedShape;
    FLOAT _eps;
    BOOL _elementwiseAffine;
};

class GroupNorm:
virtual public OpSignature {
public:
    GroupNorm() = delete;
    ~GroupNorm();
    // num_groups: 通道分组数量.
    // num_channels: 输入通道数.
    GroupNorm(UINT num_groups,
              UINT num_channels,
              FLOAT eps = 1e-5f,
              BOOL affine = true);

    Layer &operator()() = delete;
    Layer &operator()(Value_t &value);

private:
    UINT _numGroups;
    UINT _numChannels;
    FLOAT _eps;
    BOOL _affine;
};

} // namespace core
} // namespace Kernel

#endif
