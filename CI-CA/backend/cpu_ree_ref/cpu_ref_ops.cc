#include "cpu_ref_ops.h"
#include "ref_kernels.h"

#include <activation.h>
#include <convolution.h>
#include <linear.h>
#include <normalization.h>
#include <pool.h>
#include <reshape.h>

namespace Kernel {
namespace backend {
namespace cpu_ref {

namespace {

template <typename LayerT>
LayerT* checked_layer(core::Layer* layer, const char* name) {
    EXIT_ERROR_CHECK_EQ(nullptr, layer, "Layer is nullptr");
    auto* typed = dynamic_cast<LayerT*>(layer);
    EXIT_ERROR_CHECK_EQ(nullptr, typed, "%s layer type mismatch", name);
    layer->setImpl(typed);
    return typed;
}

template <typename PlanT>
void delete_plan(void* ptr) {
    delete static_cast<PlanT*>(ptr);
}

struct Conv2dPlan {
    const FLOAT* weight;
    const FLOAT* bias;
    UINT batch;
    UINT in_c;
    UINT in_h;
    UINT in_w;
    UINT out_c;
    UINT out_h;
    UINT out_w;
    UINT groups;
    UINT in_c_pg;
    UINT out_c_pg;
    UINT k_h;
    UINT k_w;
    UINT s_h;
    UINT s_w;
    UINT d_h;
    UINT d_w;
    INT pad_t;
    INT pad_l;
    BOOL bias_enabled;
};

struct LinearPlan {
    const FLOAT* weight;
    const FLOAT* bias;
    UINT in_features;
    UINT out_features;
    UINT outer;
    BOOL bias_enabled;
};

struct BatchNorm2dPlan {
    const FLOAT* weight;
    const FLOAT* bias;
    const FLOAT* running_mean;
    const FLOAT* running_var;
    UINT batch;
    UINT channels;
    UINT spatial;
    FLOAT eps;
    BOOL track_running_stats;
};

struct AdaptiveAvgPool2dPlan {
    UINT batch;
    UINT channels;
    UINT in_h;
    UINT in_w;
    UINT out_h;
    UINT out_w;
};

void require_runtime_input(const core::Value_t& value, const char* name) {
    EXIT_ERROR_CHECK_EQ(nullptr, value.data.ptr, "%s ptr is nullptr", name);
}

void require_ready_output(const core::Value_t& value, const char* name) {
    EXIT_ERROR_CHECK_EQ(nullptr, value.data.ptr, "%s ptr is nullptr", name);
}

} // namespace

void prepare_graph_input(core::Layer *layer) {
    EXIT_ERROR_CHECK_EQ(nullptr, layer, "Layer is nullptr");
}

void prepare_graph_output(core::Layer *layer) {
    EXIT_ERROR_CHECK_EQ(nullptr, layer, "Layer is nullptr");
}

void execute_graph_input(core::Layer *layer, core::ExecContext_t *ctx) {
    (void)layer;
    (void)ctx;
}

void execute_graph_output(core::Layer *layer, core::ExecContext_t *ctx) {
    (void)layer;
    (void)ctx;
}

void prepare_relu(core::Layer *layer) {
    auto* relu = checked_layer<core::UnaryOp_L>(layer, "ReLU");
    require_ready_output(relu->output(), "ReLU output");
}

void execute_relu(core::Layer *layer, core::ExecContext_t *ctx) {
    (void)ctx;
    auto* relu = layer->impl<core::UnaryOp_L>();
    core::Value_t& input = relu->input(0);
    core::Value_t& output = relu->output();
    require_runtime_input(input, "ReLU input");
    ref_relu_fp32(static_cast<const ref_float32_t*>(input.data.ptr),
                  static_cast<ref_float32_t*>(output.data.ptr),
                  input.data.shape.size);
}

void prepare_dropout(core::Layer *layer) {
    auto* dropout = checked_layer<core::UnaryOp_L>(layer, "Dropout");
    require_ready_output(dropout->output(), "Dropout output");
}

void execute_dropout(core::Layer *layer, core::ExecContext_t *ctx) {
    (void)ctx;
    auto* dropout = layer->impl<core::UnaryOp_L>();
    core::Value_t& input = dropout->input(0);
    core::Value_t& output = dropout->output();
    require_runtime_input(input, "Dropout input");
    ref_copy_bytes(input.data.ptr, output.data.ptr,
                   input.data.shape.size * input.data.getTypeSize());
}

void prepare_flatten(core::Layer *layer) {
    auto* flatten = checked_layer<core::Flatten_L>(layer, "Flatten");
    require_ready_output(flatten->output(), "Flatten output");
}

void execute_flatten(core::Layer *layer, core::ExecContext_t *ctx) {
    (void)ctx;
    auto* flatten = layer->impl<core::Flatten_L>();
    core::Value_t& input = flatten->input(0);
    core::Value_t& output = flatten->output();
    require_runtime_input(input, "Flatten input");
    ref_copy_bytes(input.data.ptr, output.data.ptr,
                   input.data.shape.size * input.data.getTypeSize());
}

void prepare_linear(core::Layer *layer) {
    auto* linear = checked_layer<core::Linear_L>(layer, "Linear");
    core::Value_t& input = linear->input(0);
    core::Value_t& output = linear->output();
    const core::Data_t* weight = linear->param(core::ParamRole::WEIGHT);
    require_ready_output(output, "Linear output");
    EXIT_ERROR_CHECK_EQ(nullptr, weight, "Linear weight is nullptr");
    EXIT_ERROR_CHECK_EQ(nullptr, weight->ptr, "Linear weight ptr is nullptr");
    if (linear->biasEnabled()) {
        const core::Data_t* bias = linear->param(core::ParamRole::BIAS);
        EXIT_ERROR_CHECK_EQ(nullptr, bias, "Linear bias is nullptr");
        EXIT_ERROR_CHECK_EQ(nullptr, bias->ptr, "Linear bias ptr is nullptr");
    }

    auto* plan = new(std::nothrow) LinearPlan();
    EXIT_ERROR_CHECK_EQ(nullptr, plan, "Linear plan allocation failed");
    plan->weight = static_cast<const FLOAT*>(weight->ptr);
    const core::Data_t* bias = linear->param(core::ParamRole::BIAS);
    plan->bias = bias ? static_cast<const FLOAT*>(bias->ptr) : nullptr;
    plan->in_features = linear->inFeatures();
    plan->out_features = linear->outFeatures();
    plan->outer = input.data.shape.size / plan->in_features;
    plan->bias_enabled = linear->biasEnabled();
    layer->setCache(plan, delete_plan<LinearPlan>);
}

void execute_linear(core::Layer *layer, core::ExecContext_t *ctx) {
    (void)ctx;
    auto* linear = layer->impl<core::Linear_L>();
    auto* plan = layer->cache<LinearPlan>();
    core::Value_t& input = linear->input(0);
    core::Value_t& output = linear->output();
    require_runtime_input(input, "Linear input");
    EXIT_ERROR_CHECK_EQ(nullptr, plan, "Linear plan is nullptr");
    ref_linear_fp32(static_cast<const ref_float32_t*>(input.data.ptr),
                    static_cast<const ref_float32_t*>(plan->weight),
                    static_cast<const ref_float32_t*>(plan->bias),
                    static_cast<ref_float32_t*>(output.data.ptr),
                    plan->outer,
                    plan->out_features,
                    plan->in_features,
                    static_cast<ref_bool_t>(plan->bias_enabled));
}

void prepare_conv2d(core::Layer *layer) {
    auto* conv = checked_layer<core::ConvNd_L>(layer, "Conv2d");
    core::Value_t& input = conv->input(0);
    core::Value_t& output = conv->output();
    const core::Data_t* weight = conv->param(core::ParamRole::WEIGHT);
    require_ready_output(output, "Conv2d output");
    EXIT_ERROR_CHECK_NE(4, input.data.shape.ndim, "Conv2d expects NCHW input");
    EXIT_ERROR_CHECK_NE(4, output.data.shape.ndim, "Conv2d expects NCHW output");
    EXIT_ERROR_CHECK_EQ(nullptr, weight, "Conv2d weight is nullptr");
    EXIT_ERROR_CHECK_EQ(nullptr, weight->ptr, "Conv2d weight ptr is nullptr");
    if (conv->biasEnabled()) {
        const core::Data_t* bias = conv->param(core::ParamRole::BIAS);
        EXIT_ERROR_CHECK_EQ(nullptr, bias, "Conv2d bias is nullptr");
        EXIT_ERROR_CHECK_EQ(nullptr, bias->ptr, "Conv2d bias ptr is nullptr");
    }

    auto* plan = new(std::nothrow) Conv2dPlan();
    EXIT_ERROR_CHECK_EQ(nullptr, plan, "Conv2d plan allocation failed");
    const core::Data_t* bias = conv->param(core::ParamRole::BIAS);
    plan->weight = static_cast<const FLOAT*>(weight->ptr);
    plan->bias = bias ? static_cast<const FLOAT*>(bias->ptr) : nullptr;
    plan->batch = input.data.shape.dims[0];
    plan->in_c = input.data.shape.dims[1];
    plan->in_h = input.data.shape.dims[2];
    plan->in_w = input.data.shape.dims[3];
    plan->out_c = output.data.shape.dims[1];
    plan->out_h = output.data.shape.dims[2];
    plan->out_w = output.data.shape.dims[3];
    plan->groups = conv->groups();
    plan->in_c_pg = conv->inChannelsPerGroup();
    plan->out_c_pg = conv->outChannelsPerGroup();
    plan->k_h = conv->kernelSize()[0];
    plan->k_w = conv->kernelSize()[1];
    plan->s_h = conv->stride()[0];
    plan->s_w = conv->stride()[1];
    plan->d_h = conv->dilation()[0];
    plan->d_w = conv->dilation()[1];
    if (conv->padding().size() == 2) {
        plan->pad_t = conv->padding()[0];
        plan->pad_l = conv->padding()[1];
    } else {
        plan->pad_t = conv->padding()[0];
        plan->pad_l = conv->padding()[2];
    }
    plan->bias_enabled = conv->biasEnabled();
    layer->setCache(plan, delete_plan<Conv2dPlan>);
}

void execute_conv2d(core::Layer *layer, core::ExecContext_t *ctx) {
    (void)ctx;
    auto* conv = layer->impl<core::ConvNd_L>();
    auto* plan = layer->cache<Conv2dPlan>();
    core::Value_t& input = conv->input(0);
    core::Value_t& output = conv->output();
    require_runtime_input(input, "Conv2d input");
    EXIT_ERROR_CHECK_EQ(nullptr, plan, "Conv2d plan is nullptr");
    ref_conv2d_nchw_fp32(static_cast<const ref_float32_t*>(input.data.ptr),
                         static_cast<const ref_float32_t*>(plan->weight),
                         static_cast<const ref_float32_t*>(plan->bias),
                         static_cast<ref_float32_t*>(output.data.ptr),
                         plan->batch, plan->in_c, plan->in_h, plan->in_w,
                         plan->out_c, plan->out_h, plan->out_w,
                         plan->groups, plan->in_c_pg, plan->out_c_pg,
                         plan->k_h, plan->k_w, plan->s_h, plan->s_w,
                         plan->d_h, plan->d_w, plan->pad_t, plan->pad_l,
                         static_cast<ref_bool_t>(plan->bias_enabled));
}

void prepare_adaptiveavgpool2d(core::Layer *layer) {
    auto* pool = checked_layer<core::AdaptivePool2d_L>(layer, "AdaptiveAvgPool2d");
    core::Value_t& input = pool->input(0);
    core::Value_t& output = pool->output(core::OutputKind::Default);
    require_ready_output(output, "AdaptiveAvgPool2d output");
    EXIT_ERROR_CHECK_NE(4, input.data.shape.ndim, "AdaptiveAvgPool2d expects NCHW input");

    auto* plan = new(std::nothrow) AdaptiveAvgPool2dPlan();
    EXIT_ERROR_CHECK_EQ(nullptr, plan, "AdaptiveAvgPool2d plan allocation failed");
    plan->batch = input.data.shape.dims[0];
    plan->channels = input.data.shape.dims[1];
    plan->in_h = input.data.shape.dims[2];
    plan->in_w = input.data.shape.dims[3];
    plan->out_h = output.data.shape.dims[2];
    plan->out_w = output.data.shape.dims[3];
    layer->setCache(plan, delete_plan<AdaptiveAvgPool2dPlan>);
}

void execute_adaptiveavgpool2d(core::Layer *layer, core::ExecContext_t *ctx) {
    (void)ctx;
    auto* pool = layer->impl<core::AdaptivePool2d_L>();
    auto* plan = layer->cache<AdaptiveAvgPool2dPlan>();
    core::Value_t& input = pool->input(0);
    core::Value_t& output = pool->output(core::OutputKind::Default);
    require_runtime_input(input, "AdaptiveAvgPool2d input");
    EXIT_ERROR_CHECK_EQ(nullptr, plan, "AdaptiveAvgPool2d plan is nullptr");
    ref_adaptiveavgpool2d_fp32(static_cast<const ref_float32_t*>(input.data.ptr),
                               static_cast<ref_float32_t*>(output.data.ptr),
                               plan->batch, plan->channels,
                               plan->in_h, plan->in_w,
                               plan->out_h, plan->out_w);
}

void prepare_batchnorm2d(core::Layer *layer) {
    auto* bn = checked_layer<core::BatchNorm2d_L>(layer, "BatchNorm2d");
    core::Value_t& input = bn->input(0);
    core::Value_t& output = bn->output();
    require_ready_output(output, "BatchNorm2d output");
    EXIT_ERROR_CHECK_NE(4, input.data.shape.ndim, "BatchNorm2d expects NCHW input");

    auto* plan = new(std::nothrow) BatchNorm2dPlan();
    EXIT_ERROR_CHECK_EQ(nullptr, plan, "BatchNorm2d plan allocation failed");
    const core::Data_t* weight = bn->param(core::ParamRole::WEIGHT);
    const core::Data_t* bias = bn->param(core::ParamRole::BIAS);
    const core::Data_t* running_mean = bn->param(core::ParamRole::RUNNING_MEAN);
    const core::Data_t* running_var = bn->param(core::ParamRole::RUNNING_VAR);
    plan->weight = weight ? static_cast<const FLOAT*>(weight->ptr) : nullptr;
    plan->bias = bias ? static_cast<const FLOAT*>(bias->ptr) : nullptr;
    plan->running_mean = running_mean ? static_cast<const FLOAT*>(running_mean->ptr) : nullptr;
    plan->running_var = running_var ? static_cast<const FLOAT*>(running_var->ptr) : nullptr;
    plan->batch = input.data.shape.dims[0];
    plan->channels = input.data.shape.dims[1];
    plan->spatial = input.data.shape.dims[2] * input.data.shape.dims[3];
    plan->eps = bn->eps();
    plan->track_running_stats = bn->trackRunningStats();
    layer->setCache(plan, delete_plan<BatchNorm2dPlan>);
}

void execute_batchnorm2d(core::Layer *layer, core::ExecContext_t *ctx) {
    (void)ctx;
    auto* bn = layer->impl<core::BatchNorm2d_L>();
    auto* plan = layer->cache<BatchNorm2dPlan>();
    core::Value_t& input = bn->input(0);
    core::Value_t& output = bn->output();
    require_runtime_input(input, "BatchNorm2d input");
    EXIT_ERROR_CHECK_EQ(nullptr, plan, "BatchNorm2d plan is nullptr");
    EXIT_ERROR_CHECK_EQ(false, plan->track_running_stats, "CPU_REE_REF currently only supports eval BatchNorm2d");
    ref_batchnorm2d_eval_fp32(static_cast<const ref_float32_t*>(input.data.ptr),
                              static_cast<ref_float32_t*>(output.data.ptr),
                              static_cast<const ref_float32_t*>(plan->weight),
                              static_cast<const ref_float32_t*>(plan->bias),
                              static_cast<const ref_float32_t*>(plan->running_mean),
                              static_cast<const ref_float32_t*>(plan->running_var),
                              plan->batch, plan->channels, plan->spatial, plan->eps);
}

} // namespace cpu_ref
} // namespace backend
} // namespace Kernel
