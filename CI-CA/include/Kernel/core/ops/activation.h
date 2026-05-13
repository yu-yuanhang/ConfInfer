#ifndef __ACTIVATION_H_CA__
#define __ACTIVATION_H_CA__

#include <All.h>
#include <core/Layer.h>
#include <core/Param.h>

namespace Kernel {
namespace core {

class ReLU;
class Sigmoid;
class Dropout;
class Softmax;

class UnaryOp_L:
virtual public Layer
{
friend class ReLU;
friend class Sigmoid;
friend class Dropout;
protected:
    UnaryOp_L() = delete;
    ~UnaryOp_L();
    UnaryOp_L(LayerType type, OpSignature *opSignature);

protected:
    void makeParams(Params *params) override;
    void makeOutputs() override;
    UINT calcWorkspaceSize() override;
};

class ReLU:
virtual public OpSignature {
public:
    ReLU() = delete;
    ~ReLU();
    // 对于单参数构造函数(所有单参数算子)
    // 这里都最好加上 explicit 避免隐式构造 带来的不确定风险
    // 是否原地修改输入.
    // 当前框架仅支持 inplace=false.
    explicit ReLU(BOOL inplace = false);

    Layer &operator()() = delete;
    Layer &operator()(Value_t &value);

private:
    BOOL _inplace;
};

class Sigmoid:
virtual public OpSignature {
public:
    Sigmoid() = delete;
    ~Sigmoid();
    // 是否原地修改输入.
    // 当前框架仅支持 inplace=false.
    explicit Sigmoid(BOOL inplace = false);

    Layer &operator()() = delete;
    Layer &operator()(Value_t &value);

private:
    BOOL _inplace;
};

class Dropout:
virtual public OpSignature {
public:
    Dropout() = delete;
    ~Dropout();
    // p: 失活概率.
    // inplace: 是否原地修改输入，当前仅支持 false.
    explicit Dropout(FLOAT p = 0.5f, BOOL inplace = false);

    Layer &operator()() = delete;
    Layer &operator()(Value_t &value);

    FLOAT p() const { return _p; }

private:
    FLOAT _p;
    BOOL _inplace;
};

class Softmax_L:
virtual public Layer
{
friend class Softmax;
protected:
    Softmax_L() = delete;
    ~Softmax_L();
    Softmax_L(INT dim, OpSignature *opSignature);

protected:
    void makeParams(Params *params) override;
    void makeOutputs() override;
    UINT calcWorkspaceSize() override;

public:
    INT dim() const { return _dim; }

private:
    INT _dim;
};

class Softmax:
virtual public OpSignature {
public:
    Softmax() = delete;
    ~Softmax();
    // dim: 沿哪个维度做 softmax，支持负索引语义.
    explicit Softmax(INT dim);

    Layer &operator()() = delete;
    Layer &operator()(Value_t &value);

private:
    INT _dim;
};

} // namespace core
} // namespace Kernel

#endif
