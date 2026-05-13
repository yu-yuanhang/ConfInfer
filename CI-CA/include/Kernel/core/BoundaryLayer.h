#ifndef __BOUNDARY_LAYER_H_CA__
#define __BOUNDARY_LAYER_H_CA__

#include <core/Layer.h>

namespace Kernel {
namespace core {

class GraphInputLayer : public Layer {
public:
    // explicit 还是必要的 
    // 避免用户 (隐式转换 + 拷贝构造) 的方式创建实例
    explicit GraphInputLayer(const std::vector<Value_t*>& values);

protected:
    void makeOutputs() override;
    UINT calcWorkspaceSize() override { return 0; }
    void makeParams(Params* params) override { (void)params; }
};

class GraphOutputLayer : public Layer {
public:
    explicit GraphOutputLayer(const std::vector<Value_t*>& values);
    Value_t& input(UINT idx);

protected:
    void makeOutputs() override;
    UINT calcWorkspaceSize() override { return 0; }
    void makeParams(Params* params) override { (void)params; }

private:
    void bindInputs(const std::vector<Value_t*>& values);
};

} // namespace core
} // namespace Kernel

#endif
