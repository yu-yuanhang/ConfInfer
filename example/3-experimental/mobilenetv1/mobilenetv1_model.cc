#include "mobilenetv1_model.h"

using namespace Kernel;
using namespace Kernel::core;

namespace {

UINT make_divisible(FLOAT value, UINT divisor = 8) {
    INT rounded = static_cast<INT>(value + static_cast<FLOAT>(divisor) / 2.0f);
    rounded = (rounded / static_cast<INT>(divisor)) * static_cast<INT>(divisor);
    if (rounded < static_cast<INT>(divisor)) {
        rounded = static_cast<INT>(divisor);
    }
    return static_cast<UINT>(rounded);
}

} // namespace

MobileNetV1Model::MobileNetV1Model(UINT num_classes,
                                   FLOAT width_multiplier,
                                   FLOAT dropout_rate)
    : num_classes_(num_classes),
      width_multiplier_(width_multiplier),
      dropout_rate_(dropout_rate),
      graph_input_({1, 3, 32, 32}),
      owned_ops_(),
      ops_(),
      output_(nullptr) {
    build();
}

ParamBindingTable MobileNetV1Model::build_param_bindings() const {
    ParamBindingTable bindings;

    for (UINT op_index = 0; op_index < ops_.size(); ++op_index) {
        const MobileNetOpInfo& op_info = ops_[op_index];
        const Layer* layer = op_info.layer;
        EXIT_ERROR_CHECK_EQ(nullptr, layer, "MobileNet op layer is nullptr");

        add_param_binding(bindings, *layer, op_index, ParamRole::WEIGHT, "weight");
        add_param_binding(bindings, *layer, op_index, ParamRole::BIAS, "bias");
        add_param_binding(bindings, *layer, op_index, ParamRole::RUNNING_MEAN, "running_mean");
        add_param_binding(bindings, *layer, op_index, ParamRole::RUNNING_VAR, "running_var");
    }

    return bindings;
}

void MobileNetV1Model::add_param_binding(ParamBindingTable& bindings,
                                         const Layer& layer,
                                         UINT op_index,
                                         ParamRole role,
                                         const std::string& suffix) {
    const Data_t* param = layer.param(role);
    if (nullptr == param) {
        return;
    }

    const std::string external_name =
        "ops." + std::to_string(op_index) + "." + suffix;
    bindings.add(external_name,
                 const_cast<Data_t*>(param),
                 layer_debug_name(op_index, suffix));
}

std::string MobileNetV1Model::layer_debug_name(UINT op_index, const std::string& suffix) {
    return "ops[" + std::to_string(op_index) + "]." + suffix;
}

Layer& MobileNetV1Model::add_conv_bn_act(Value_t& input,
                                         UINT in_channels,
                                         UINT out_channels,
                                         UINT stride,
                                         ExecutionDomain domain,
                                         const std::string& block_id) {
    Layer& conv = add_op(
        "conv_" + block_id,
        "conv",
        domain,
        std::make_unique<Conv2d>(in_channels, out_channels,
                                 std::vector<UINT>{3, 3},
                                 std::vector<UINT>{stride, stride},
                                 std::vector<INT>{1, 1, 1, 1},
                                 std::vector<UINT>{1, 1},
                                 1,
                                 false),
        [&](Conv2d& op) -> Layer& { return op(input); });

    Layer& bn = add_op(
        "bn_" + block_id,
        "bn",
        domain,
        std::make_unique<BatchNorm2d>(out_channels),
        [&](BatchNorm2d& op) -> Layer& { return op(conv.output()); });

    Layer& act = add_op(
        "act_" + block_id,
        "relu",
        domain,
        std::make_unique<ReLU>(false),
        [&](ReLU& op) -> Layer& { return op(bn.output()); });

    return act;
}

Layer& MobileNetV1Model::add_depthwise_separable_conv(Value_t& input,
                                                      UINT in_channels,
                                                      UINT out_channels,
                                                      UINT stride,
                                                      ExecutionDomain domain,
                                                      const std::string& block_id) {
    Layer& dw_conv = add_op(
        "dw_conv_" + block_id,
        "depthwise_conv",
        domain,
        std::make_unique<Conv2d>(in_channels, in_channels,
                                 std::vector<UINT>{3, 3},
                                 std::vector<UINT>{stride, stride},
                                 std::vector<INT>{1, 1, 1, 1},
                                 std::vector<UINT>{1, 1},
                                 in_channels,
                                 false),
        [&](Conv2d& op) -> Layer& { return op(input); });

    Layer& dw_bn = add_op(
        "dw_bn_" + block_id,
        "bn",
        domain,
        std::make_unique<BatchNorm2d>(in_channels),
        [&](BatchNorm2d& op) -> Layer& { return op(dw_conv.output()); });

    Layer& dw_act = add_op(
        "dw_act_" + block_id,
        "relu",
        domain,
        std::make_unique<ReLU>(false),
        [&](ReLU& op) -> Layer& { return op(dw_bn.output()); });

    Layer& pw_conv = add_op(
        "pw_conv_" + block_id,
        "pointwise_conv",
        domain,
        std::make_unique<Conv2d>(in_channels, out_channels,
                                 std::vector<UINT>{1, 1},
                                 std::vector<UINT>{1, 1},
                                 std::vector<INT>{0, 0, 0, 0},
                                 std::vector<UINT>{1, 1},
                                 1,
                                 false),
        [&](Conv2d& op) -> Layer& { return op(dw_act.output()); });

    Layer& pw_bn = add_op(
        "pw_bn_" + block_id,
        "bn",
        domain,
        std::make_unique<BatchNorm2d>(out_channels),
        [&](BatchNorm2d& op) -> Layer& { return op(pw_conv.output()); });

    Layer& pw_act = add_op(
        "pw_act_" + block_id,
        "relu",
        domain,
        std::make_unique<ReLU>(false),
        [&](ReLU& op) -> Layer& { return op(pw_bn.output()); });

    return pw_act;
}

void MobileNetV1Model::build() {
    const ExecutionDomain ree = ExecutionDomain::ED_CPU_REE;
    const ExecutionDomain tee = ExecutionDomain::ED_CPU_TEE;
    UINT in_channels = 3;

    UINT out_channels = make_divisible(32.0f * width_multiplier_);
    Layer& stem = add_conv_bn_act(graph_input_, in_channels, out_channels, 1, tee, "stem");
    in_channels = out_channels;

    out_channels = make_divisible(64.0f * width_multiplier_);
    Layer& stage1 = add_depthwise_separable_conv(stem.output(), in_channels, out_channels, 1, tee, "1_0");
    in_channels = out_channels;

    out_channels = make_divisible(128.0f * width_multiplier_);
    Layer& stage2_0 = add_depthwise_separable_conv(stage1.output(), in_channels, out_channels, 2, ree, "2_0");
    Layer& stage2_1 = add_depthwise_separable_conv(stage2_0.output(), out_channels, out_channels, 1, ree, "2_1");
    in_channels = out_channels;

    out_channels = make_divisible(256.0f * width_multiplier_);
    Layer& stage3_0 = add_depthwise_separable_conv(stage2_1.output(), in_channels, out_channels, 2, ree, "3_0");
    Layer& stage3_1 = add_depthwise_separable_conv(stage3_0.output(), out_channels, out_channels, 1, ree, "3_1");
    Layer& stage3_2 = add_depthwise_separable_conv(stage3_1.output(), out_channels, out_channels, 1, ree, "3_2");
    in_channels = out_channels;

    out_channels = make_divisible(512.0f * width_multiplier_);
    Layer& stage4_0 = add_depthwise_separable_conv(stage3_2.output(), in_channels, out_channels, 2, ree, "4_0");
    Layer& stage4_1 = add_depthwise_separable_conv(stage4_0.output(), out_channels, out_channels, 1, ree, "4_1");
    Layer& stage4_2 = add_depthwise_separable_conv(stage4_1.output(), out_channels, out_channels, 1, tee, "4_2");
    Layer& stage4_3 = add_depthwise_separable_conv(stage4_2.output(), out_channels, out_channels, 1, tee, "4_3");
    in_channels = out_channels;

    Layer& avgpool = add_op(
        "avgpool",
        "avgpool",
        ree,
        std::make_unique<AdaptiveAvgPool2d>(std::vector<UINT>{1}),
        [&](AdaptiveAvgPool2d& op) -> Layer& { return op(stage4_3.output()); });

    Layer& flatten = add_op(
        "flatten",
        "flatten",
        ree,
        std::make_unique<Flatten>(1, -1),
        [&](Flatten& op) -> Layer& { return op(avgpool.output()); });

    Layer& dropout = add_op(
        "dropout",
        "dropout",
        ree,
        std::make_unique<Dropout>(dropout_rate_, false),
        [&](Dropout& op) -> Layer& { return op(flatten.output()); });

    Layer& classifier = add_op(
        "classifier",
        "linear",
        ree,
        std::make_unique<Linear>(in_channels, num_classes_, true),
        [&](Linear& op) -> Layer& { return op(dropout.output()); });

    output_ = &classifier;
}
