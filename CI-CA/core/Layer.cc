#include <core/Layer.h>

namespace Kernel {
namespace core {

static Value_t dummyValue;

namespace {

ExecutionDomain domain_from_flags(uint32_t flags) {
    if (flags & LF_REQUIRE_TEE) {
        return ExecutionDomain::ED_CPU_TEE;
    }
    // 以下是 非 TEE 的情况
    if (flags & LF_PREFER_CPU) {
        return ExecutionDomain::ED_CPU_REE;
    }
    return ExecutionDomain::ED_DEFAULT;
}

void apply_domain_flags(uint32_t &flags, ExecutionDomain domain) {
    flags &= ~(LF_REQUIRE_TEE | LF_PREFER_CPU);

    switch (domain) {
    case ExecutionDomain::ED_CPU_TEE:
        flags |= LF_REQUIRE_TEE;
        break;
    case ExecutionDomain::ED_CPU_REE:
        flags |= LF_PREFER_CPU;
        break;
    case ExecutionDomain::ED_DEFAULT:
    default:
        flags |= LF_DEFAULT;
        break;
    }
}

} // namespace

Layer::Layer(LayerType type, OpSignature *opSignature):
    _id(_counter++), _type(type), _lf(opSignature->flags()),
    _inputsLNum(0), _outputsLNum(0),
    _inputsL(), _outputsL(),
    _inputs(), _outputs(),
    _workspaceSize(0),
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
    if (_params) {
        _params->release();
        _params = nullptr;
    }
    _opSignature = nullptr;
}

LayerSlice *Layer::makeSliceDesc(UINT sliceId, UINT sliceNum) {
    SliceDesc_t desc{};
    desc.sliceId = sliceId;
    desc.sliceNum = sliceNum;
    desc.workspaceOffset = 0;
    desc.workspaceSize = _workspaceSize;

    LayerSlice *ls = new LayerSlice(this, desc);

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
Value_t& Layer::input(uint32_t idx) {
    if (idx < _inputs.size()) { return *(_inputs[idx]); }
    LogDebug("Error: Input not found");
    std::exit(EXIT_FAILURE);
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
Value_t& Layer::output(OutputKind kind) {
    return output(kind, 0);
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
    }
    uint32_t default_count = 0;
    Value_t* default_value = nullptr;
    for (auto it = _outputs.begin(); it != _outputs.end(); ++it) {
        Value_t* value = it->get();
        if (OutputKind::Default == value->kind) {
            ++default_count;
            if (1 == default_count) {
                default_value = value;
            }
        }
    }
    if (1 == default_count && nullptr != default_value) {
        return *default_value;
    }
    LogDebug("Error: Layer has %zu outputs and no unique default output, use output(idx) or output(kind)",
             _outputs.size());
    std::exit(EXIT_FAILURE);
    return dummyValue;
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

ExecutionDomain Layer::execDomain() const {
    return domain_from_flags(_lf);
}

Layer &Layer::setExecDomain(ExecutionDomain domain) {
    EXIT_ERROR_CHECK_EQ(false, is_exec_domain_registered(domain),
        "Layer execution domain is not registered in EXECUTOR");
    apply_domain_flags(_lf, domain);
    return *this;
}

Layer &Layer::requireTEE(bool enable) {
    return setExecDomain(enable ? ExecutionDomain::ED_CPU_TEE : ExecutionDomain::ED_CPU_REE);
}

Layer &Layer::useLocal(bool enable) {
    return setExecDomain(enable ? ExecutionDomain::ED_CPU_REE : ExecutionDomain::ED_DEFAULT);
}

std::atomic<UINT> Layer::_counter{0};

OpSignature::OpSignature(LayerType type):
    _type(type), _lf(LF_DEFAULT),
    _layers(),
    _ownParams(true), _params(nullptr) {}

OpSignature::~OpSignature() {
    // 这里需要释放 _layers 中的节点
    for (auto it = _layers.begin(); it != _layers.end();) {
        Layer *l = *it;
        if (l) delete l; // 释放对象 (理论上也不该存在 nullptr 的情况)
        it = _layers.erase(it);  // erase 返回下一个有效迭代器
    }
    if (_params) {
        _params->release();
        _params = nullptr;
    }
}

ExecutionDomain OpSignature::execDomain() const {
    return domain_from_flags(_lf);
}

OpSignature &OpSignature::setExecDomain(ExecutionDomain domain) {
    EXIT_ERROR_CHECK_EQ(false, _layers.empty(),
        "OpSignature execution domain cannot be changed after Layer instances have been created; "
        "set the execution domain before materializing the graph, or use Layer::setExecDomain() on concrete nodes");
    EXIT_ERROR_CHECK_EQ(false, is_exec_domain_registered(domain),
        "OpSignature execution domain is not registered in EXECUTOR");
    apply_domain_flags(_lf, domain);
    return *this;
}

OpSignature &OpSignature::requireTEE(bool enable) {
    return setExecDomain(enable ? ExecutionDomain::ED_CPU_TEE : ExecutionDomain::ED_CPU_REE);
}

OpSignature &OpSignature::useLocal(bool enable) {
    return setExecDomain(enable ? ExecutionDomain::ED_CPU_REE : ExecutionDomain::ED_DEFAULT);
}

void OpSignature::dealParams(Layer *l) {
    EXIT_ERROR_CHECK_EQ(nullptr, l, "Layer *l == nullptr");

    // 一个 OpSignature 可以派生多个 Layer，但共享同一组参数。
    // 因此参数创建只发生一次，之后所有 Layer 仅绑定同一个 Params 句柄。
    if (_ownParams) {
        if (nullptr == _params) {
            Params *p = new(std::nothrow) Params();
            EXIT_ERROR_CHECK_EQ(nullptr, p, "new Params failed");
            l->makeParams(p);
            _params = p;
        }

        _params->retain();
        l->setParams(_params);
    } else {
        EXIT_ERROR_CHECK_EQ(0, 0, "_ownParams == false");
    }
    
    return;
}

} // namespace end of core
} // namespace end of Kernel 
