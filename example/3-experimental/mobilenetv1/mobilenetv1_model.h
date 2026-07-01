#ifndef __MOBILENETV1_MODEL_EXAMPLE_H__
#define __MOBILENETV1_MODEL_EXAMPLE_H__

#include <memory>
#include <string>
#include <vector>

#include <core/model_param_loader.h>
#include <ops.h>

namespace Kernel {
namespace core {
class Layer;
}
}

struct MobileNetOpInfo {
    std::string name;
    std::string type;
    Kernel::core::Layer* layer;
};

class MobileNetV1Model {
public:
    MobileNetV1Model(Kernel::UINT num_classes = 100,
                     Kernel::FLOAT width_multiplier = 1.0f,
                     Kernel::FLOAT dropout_rate = 0.2f);

    Kernel::core::Value_t& graph_input() { return graph_input_; }
    Kernel::core::Layer& output_layer() const { return *output_; }
    const std::vector<MobileNetOpInfo>& ops() const { return ops_; }
    Kernel::core::ParamBindingTable build_param_bindings() const;

private:
    static void add_param_binding(Kernel::core::ParamBindingTable& bindings,
                                  const Kernel::core::Layer& layer,
                                  Kernel::UINT op_index,
                                  Kernel::core::ParamRole role,
                                  const std::string& suffix);
    static std::string layer_debug_name(Kernel::UINT op_index, const std::string& suffix);

    template <typename OpT, typename Fn>
    Kernel::core::Layer& add_op(const std::string& name,
                                const std::string& type,
                                Kernel::core::ExecutionDomain domain,
                                std::unique_ptr<OpT> op,
                                Fn&& invoke) {
        op->setExecDomain(domain);
        OpT* raw = op.get();
        Kernel::core::Layer& layer = invoke(*raw);
        owned_ops_.push_back(std::move(op));
        ops_.push_back(MobileNetOpInfo{name, type, &layer});
        return layer;
    }

    Kernel::core::Layer& add_conv_bn_act(Kernel::core::Value_t& input,
                                         Kernel::UINT in_channels,
                                         Kernel::UINT out_channels,
                                         Kernel::UINT stride,
                                         Kernel::core::ExecutionDomain domain,
                                         const std::string& block_id);
    Kernel::core::Layer& add_depthwise_separable_conv(Kernel::core::Value_t& input,
                                                      Kernel::UINT in_channels,
                                                      Kernel::UINT out_channels,
                                                      Kernel::UINT stride,
                                                      Kernel::core::ExecutionDomain domain,
                                                      const std::string& block_id);
    void build();

private:
    Kernel::UINT num_classes_;
    Kernel::FLOAT width_multiplier_;
    Kernel::FLOAT dropout_rate_;
    Kernel::core::Value_t graph_input_;
    std::vector<std::unique_ptr<Kernel::core::OpSignature>> owned_ops_;
    std::vector<MobileNetOpInfo> ops_;
    Kernel::core::Layer* output_;
};

#endif
