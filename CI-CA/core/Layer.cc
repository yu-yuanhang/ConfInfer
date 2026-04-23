#include <core/Layer.h>

namespace Kernel {
namespace core {

static Value_t dummyValue;

Layer::Layer(LayerType type, OpSignature *opSignature):
    _id(_counter++), _type(type), _inTEE(false), _lf(opSignature->flags()),
    _inputsLNum(INVALID_VALUE_U), _outputsLNum(INVALID_VALUE_U),
    _inputsL(), _outputsL(),
    _inputs(), _outputs(),
    _workspaceSize(INVALID_VALUE_U),
    _params(nullptr),
    _opSignature(opSignature)
{
    // LogDebug("Layer(LayerType type)");
    // _outputs 应该是可以基于 type 直接初始化的 这里虚函数接口给出去
    // _outputs 初始化过程依赖 _inputs
    // _inputs 应该在 graph 构建过程中初始化
};
Layer::Layer(const Layer &rhs) {
    EXIT_ERROR("Layer::Layer(const Layer &rhs) ...... todo ");
    // ......
}

Layer::~Layer() {
    // Layer 生命周期由 OpSignature 管理
    // 这里无需基于 OpSignature._ownParams 判断是否释放 _params
    _params->release();
    _params = nullptr;
    _opSignature = nullptr;
}

LayerSlice *Layer::makeSliceDesc(UINT sliceId, UINT sliceNum) {
    
    // ......
    LayerSlice *ls = new LayerSlice();

    return ls;
}

void Layer::bind_inputs (Value_t &value) {
    _inputs.push_back(const_cast<Value_t *>(&value));
    // 断言是否存在重边的情况
    if (value.producer && !_inputsL.contains(value.producer)) {
        _inputsL.push_back(value.producer);
        ++_inputsLNum;
    }
    if (value.producer && !value.producer->_outputsL.contains(this)) {
        value.producer->_outputsL.push_back(this);
        ++(value.producer->_outputsLNum);
    }

    // operator() 已经断言了重复绑定情况
    value.consumers.push_back(this);
}
void Layer::linkInit() {
    // 算子相关设定 (派生类处理)
    _workspaceSize = calcWorkspaceSize();
    makeOutputs();
    return;
}
Layer& Layer::link(Value_t &value) {
    // ...... inputs 数量正确性检查
    bind_inputs(value);
    linkInit();
    return *this;
}
Value_t& Layer::output(OutputKind kind, uint32_t slot) {
    uint32_t count = 0;
    for (auto it = _outputs.begin(); it != _outputs.end(); ++it) {
        Value_t *value = it->get();   
        if (value->kind == kind) {
            if (count == slot)
                return *value;
            ++count;
        }
    }
    LogDebug("Error: Output not found");
    std::exit(EXIT_FAILURE);
    // return dummyValue;
}
Value_t& Layer::output(uint32_t idx) {
    if (idx < _outputs.size()) { return *(_outputs[idx]); }
    LogDebug("Error: Output not found");
    std::exit(EXIT_FAILURE);
    // return dummyValue;
}
Value_t& Layer::output() {
    if (_outputs.empty()) {
        LogDebug("Error: Output not found");
        std::exit(EXIT_FAILURE);
        // return dummyValue;
    }
    return *(_outputs[0]);
}
std::vector<Value_t*> Layer::outputs(OutputKind kind) {
    std::vector<Value_t*> result;
    result.reserve(_outputs.size());

    for (auto it = _outputs.begin(); it != _outputs.end(); ++it) {
        Value_t* value = it->get();
        if (value->kind == kind) {
            result.push_back(value);
        }
    }
    // if (result.empty()) {}
    return result;
}

std::atomic<UINT> Layer::_counter{0};

OpSignature::OpSignature(LayerType type):
    _type(type), _inTEE(false), _lf(LF_DEFAULT),
    _layers(),
    _ownParams(true), _params(nullptr) {}

OpSignature::~OpSignature() {
    // 这里需要释放 _layers 中的节点
    for (auto it = _layers.begin(); it != _layers.end();) {
        Layer *l = *it;
        if (l) delete l; // 释放对象 (理论上也不该存在 nullptr 的情况)
        it = _layers.erase(it);  // erase 返回下一个有效迭代器
    }
    _params->release();
    _params = nullptr;
}

void OpSignature::dealParams(Layer *l) {
    // EXIT_ERROR_CHECK_EQ(nullptr, l, "Layer *l == nullptr");

    // Params_t 内部存在引用计数 其生命周期释放方式 应该和 _ownParams 解耦
    // 即对 Layer 和 OpSignature 无感
    // ...... todo
    if (_ownParams) {
        if (nullptr != _params || !_layers.empty()) {
            if (nullptr != _params) {
                _params->retain();
                l->setParams(_params);
            } 
            return;
        }
        Params *p = new(std::nothrow) Params();
        // EXIT_ERROR_CHECK_EQ(nullptr, p, "new Params failed");
        l->makeParams(p);
        _params = p;

        _params->retain();
        l->setParams(_params);
    } else {
        EXIT_ERROR_CHECK_EQ(0, 0, "_ownParams == false");
    }
    
    return;
}

} // namespace end of core
} // namespace end of Kernel 

