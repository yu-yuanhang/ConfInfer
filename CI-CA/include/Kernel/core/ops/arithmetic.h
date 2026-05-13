#ifndef __ARITHMETIC_H_CA__
#define __ARITHMETIC_H_CA__

#include <All.h>
#include <core/Layer.h>
#include <core/Param.h>

namespace Kernel {
namespace core {

class Add;
class BiasAdd;
class MatMul;
class Concat;

class Add_L:
virtual public Layer
{
friend class Add;
protected:
    Add_L() = delete;
    ~Add_L();
    Add_L(FLOAT alpha, OpSignature *opSignature);

protected:
    void makeParams(Params *params) override;
    void makeOutputs() override;
    UINT calcWorkspaceSize() override;

public:
    FLOAT alpha() const { return _alpha; }

private:
    FLOAT _alpha;
};

class BiasAdd_L:
virtual public Layer
{
friend class BiasAdd;
protected:
    BiasAdd_L() = delete;
    ~BiasAdd_L();
    BiasAdd_L(UINT size, INT dim, OpSignature *opSignature);

protected:
    void makeParams(Params *params) override;
    void makeOutputs() override;
    UINT calcWorkspaceSize() override;

public:
    UINT size() const { return _size; }
    INT dim() const { return _dim; }

private:
    UINT _size;
    INT _dim;
};

class MatMul_L:
virtual public Layer
{
friend class MatMul;
protected:
    MatMul_L() = delete;
    ~MatMul_L();
    MatMul_L(OpSignature *opSignature);

protected:
    void makeParams(Params *params) override;
    void makeOutputs() override;
    UINT calcWorkspaceSize() override;
};

class Concat_L:
virtual public Layer
{
friend class Concat;
protected:
    Concat_L() = delete;
    ~Concat_L();
    Concat_L(INT dim, OpSignature *opSignature);

protected:
    void makeParams(Params *params) override;
    void makeOutputs() override;
    UINT calcWorkspaceSize() override;

public:
    INT dim() const { return _dim; }

private:
    INT _dim;
};

class Add:
virtual public OpSignature {
public:
    Add() = delete;
    ~Add();
    // 第二个输入的 缩放系数
    // alpha: 第二个输入的缩放系数，对齐 torch.add(input, other, alpha=...).
    explicit Add(FLOAT alpha = 1.0f);

    Layer &operator()() = delete;
    Layer &operator()(Value_t &input, Value_t &other);

private:
    FLOAT _alpha;
};

class BiasAdd:
virtual public OpSignature {
public:
    BiasAdd() = delete;
    ~BiasAdd();
    // size 表示 bias 向量的元素个数
    // 以卷积为例子 一般 size == C 通道数
    // dim 表示 bias 表示哪个维度 支持负语义索引
    // dim = 0 -> N，batch 维
    // dim = 1 -> C，通道维
    // dim = 2 -> H，高度维
    // dim = 3 -> W，宽度维
    explicit BiasAdd(UINT size, INT dim = 1);

    Layer &operator()() = delete;
    Layer &operator()(Value_t &input);

private:
    UINT _size;
    INT _dim;
};

class MatMul:
virtual public OpSignature {
public:
    ~MatMul();
    MatMul();

    Layer &operator()() = delete;
    Layer &operator()(Value_t &input, Value_t &other);
};

class Concat:
virtual public OpSignature {
public:
    Concat() = delete;
    ~Concat();
    // dim: 沿哪个维度拼接，支持负索引语义.
    explicit Concat(INT dim = 0);

    Layer &operator()() = delete;
    Layer &operator()(Value_t &value);
    template<typename... Args>
    Layer &operator()(Value_t &value, Args &... rest) {
        static_assert(
            (std::is_same_v<Value_t, std::remove_reference_t<Args>> && ...),
            "Layer::operator() only accepts Value_t& arguments"
        );
        Layer *l = (Layer *) new(std::nothrow) Concat_L(_dim, this);
        EXIT_ERROR_CHECK_EQ(l, nullptr, "(new) heap allocation failed");
        dealParams(l);
        _layers.push_back(l);
        return l->link(value, rest...);
    }

private:
    INT _dim;
};

} // namespace core
} // namespace Kernel

#endif
