#ifndef __LINEAR_H_CA__
#define __LINEAR_H_CA__

#include <All.h>
#include <core/Layer.h>
#include <core/Param.h>

namespace Kernel {
namespace core {

class Linear;

class Linear_L:
virtual public Layer
{
friend class Linear;
protected:
    Linear_L() = delete;
    ~Linear_L();
    Linear_L(UINT inFeatures,
             UINT outFeatures,
             BOOL bias,
             OpSignature *opSignature);

protected:
    void makeParams(Params *params) override;
    void makeOutputs() override;
    UINT calcWorkspaceSize() override;

public:
    UINT inFeatures() const { return _inFeatures; }
    UINT outFeatures() const { return _outFeatures; }
    BOOL biasEnabled() const { return _bias; }

private:
    UINT _inFeatures;
    UINT _outFeatures;
    BOOL _bias;
};

class Linear:
virtual public OpSignature {
public:
    Linear() = delete;
    ~Linear();
    // 输入/输出 最后一个维度的特征值数量
    // in_features: 输入最后一维长度.
    // out_features: 输出最后一维长度.
    // bias: 是否带偏置项.
    Linear(UINT in_features,
           UINT out_features,
           BOOL bias = true);

    Layer &operator()() = delete;
    Layer &operator()(Value_t &value);

private:
    UINT _inFeatures;
    UINT _outFeatures;
    BOOL _bias;
};

} // namespace core
} // namespace Kernel

#endif
