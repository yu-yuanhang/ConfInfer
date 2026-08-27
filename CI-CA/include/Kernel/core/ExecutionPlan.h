#ifndef __EXECUTION_PLAN_H_CA__
#define __EXECUTION_PLAN_H_CA__

#include <core/ExecutionPartition.h>

namespace Kernel {
namespace core {

enum class ExecUnitType : uint8_t {
    EU_LAYER = 0,
    EU_PARTITION,
};

// ExecutionPlan 是运行阶段真正消费的执行序列。
// 当前 ExecutionPlan 直接持有 Layer*，
class ExecUnit {
public:
    ExecUnit()
        : _type(ExecUnitType::EU_LAYER),
          _domain(ExecutionDomain::ED_DEFAULT),
          _part(nullptr),
          _layers(),
          _inputs(),
          _outputs() {}

    ExecUnitType type() const { return _type; }
    void setType(ExecUnitType type) { _type = type; }

    ExecutionDomain domain() const { return _domain; }
    void setDomain(ExecutionDomain domain) { _domain = domain; }

    const ExecPartition *part() const { return _part; }
    void setPart(const ExecPartition *part) { _part = part; }

    const std::vector<Layer *>& layers() const { return _layers; }
    const std::vector<Value_t *>& inputs() const { return _inputs; }
    const std::vector<Value_t *>& outputs() const { return _outputs; }

    void addLayer(Layer *layer) { pushUnique(_layers, layer); }
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
    // ExecUnit 不拥有 partition / layer / value 生命周期
    const ExecPartition *_part;
    std::vector<Layer *> _layers;
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
