#ifndef __EXECUTION_PLAN_H_CA__
#define __EXECUTION_PLAN_H_CA__

#include <core/ExecutionPartition.h>

namespace Kernel {
namespace core {

enum class ExecUnitType : uint8_t {
    EU_LAYER = 0,
    EU_PARTITION,
};

// 这里的执行单元有些乱 
// 最初就是简单的按照 LayerSlice 逐层执行
// 后来为不同执行区域 (主要是为了服务 TEE 内的连续调用) 的执行单元添加了 domain 属性
// 所以新增了 ExecutionPartition 
// 但是 ExecutionPartition 只是作用对象是计算图结构和 Layer
// 而框架原先的执行单元是 LayerSlice 所以需要有一个统一的执行单元
class ExecUnit {
public:
    ExecUnit()
        : _type(ExecUnitType::EU_LAYER),
          _domain(ExecutionDomain::ED_DEFAULT),
          _part(nullptr),
          _slices(),
          _inputs(),
          _outputs() {}

    ExecUnitType type() const { return _type; }
    void setType(ExecUnitType type) { _type = type; }

    ExecutionDomain domain() const { return _domain; }
    void setDomain(ExecutionDomain domain) { _domain = domain; }

    const ExecutionPartition *part() const { return _part; }
    void setPart(const ExecutionPartition *part) { _part = part; }

    const std::vector<LayerSlice *>& slices() const { return _slices; }
    const std::vector<Value_t *>& inputs() const { return _inputs; }
    const std::vector<Value_t *>& outputs() const { return _outputs; }

    void addSlice(LayerSlice *slice) { pushUnique(_slices, slice); }
    void addInput(Value_t *value) { pushUnique(_inputs, value); }
    void addOutput(Value_t *value) { pushUnique(_outputs, value); }

private:
    template <typename T>
    static void pushUnique(std::vector<T>& vec, T item) {
        if (std::find(vec.begin(), vec.end(), item) == vec.end()) {
            vec.push_back(item);
        }
    }

private:
    ExecUnitType _type;
    ExecutionDomain _domain;
    // ExecUnit 不拥有 partition / slice / value 生命周期
    const ExecutionPartition *_part;
    // 一个 ExecUnit 可以表示 一个或多个  LayerSlice
    std::vector<LayerSlice *> _slices;
    std::vector<Value_t *> _inputs;
    std::vector<Value_t *> _outputs;
};

class ExecutionPlan {
public:
    ExecutionPlan() : _units() {}

    void clear() { _units.clear(); }
    bool empty() const { return _units.empty(); }
    UINT size() const { return static_cast<UINT>(_units.size()); }

    const std::vector<ExecUnit>& units() const { return _units; }
    std::vector<ExecUnit>& units() { return _units; }

    void addUnit(const ExecUnit& unit) { _units.push_back(unit); }

private:
    std::vector<ExecUnit> _units;
};

} // namespace core
} // namespace Kernel

#endif
