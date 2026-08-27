#include "cpu_ops.h"
#include "math/math_utils_cpu.h"
#include <convolution.h>

namespace Kernel {
namespace backend {
namespace cpu {

namespace {
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
    UINT out_spatial;
    UINT kernel_area;
    UINT K;
    UINT N;
    UINT input_batch_stride;
    UINT output_batch_stride;
    UINT group_input_stride;
    UINT group_weight_stride;
    UINT group_output_stride;
    UINT workspace_bytes;
};

void delete_conv2d_plan(void* ptr) {
    delete static_cast<Conv2dPlan*>(ptr);
}

void require_runtime_input(const core::Value_t& value, const char* name) {
    EXIT_ERROR_CHECK_EQ(nullptr, value.data.ptr, "%s ptr is nullptr", name);
}

Conv2dPlan* build_conv2d_plan(core::ConvNd_L* conv) {
    auto* plan = new(std::nothrow) Conv2dPlan();
    EXIT_ERROR_CHECK_EQ(nullptr, plan, "Conv2d plan allocation failed");

    core::Value_t& input = conv->input(0);
    core::Value_t& output = conv->output();
    const core::Data_t* weight = conv->param(core::ParamRole::WEIGHT);
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

    plan->out_spatial = plan->out_h * plan->out_w;
    plan->kernel_area = plan->k_h * plan->k_w;
    plan->K = plan->in_c_pg * plan->kernel_area;
    plan->N = plan->out_spatial;
    plan->input_batch_stride = plan->in_c * plan->in_h * plan->in_w;
    plan->output_batch_stride = plan->out_c * plan->out_h * plan->out_w;
    plan->group_input_stride = plan->in_c_pg * plan->in_h * plan->in_w;
    plan->group_weight_stride = plan->out_c_pg * plan->K;
    plan->group_output_stride = plan->out_c_pg * plan->out_spatial;
    plan->workspace_bytes = plan->K * plan->N * sizeof(FLOAT);
    return plan;
}
} // namespace

void prepare_conv2d(core::Layer *layer) {
    EXIT_ERROR_CHECK_EQ(nullptr, layer, "Layer is nullptr");
    auto* conv = dynamic_cast<core::ConvNd_L*>(layer);
    EXIT_ERROR_CHECK_EQ(nullptr, conv, "Layer is not ConvNd_L");
    layer->setImpl(conv);

    core::Value_t& input = conv->input(0);
    core::Value_t& output = conv->output();

    EXIT_ERROR_CHECK_EQ(nullptr, output.data.ptr, "Conv2d output ptr is nullptr");
    EXIT_ERROR_CHECK_NE(4, input.data.shape.ndim, "Conv2d expects NCHW input");
    EXIT_ERROR_CHECK_NE(4, output.data.shape.ndim, "Conv2d expects NCHW output");
    EXIT_ERROR_CHECK_NE(2, conv->spatialDim(), "Conv2d spatial dim must be 2");
    EXIT_ERROR_CHECK_EQ(false, conv->padding().size() == 4 || conv->padding().size() == 2,
        "Conv2d padding must contain 2 or 4 elements");

    const core::Data_t* weight = conv->param(core::ParamRole::WEIGHT);
    EXIT_ERROR_CHECK_EQ(nullptr, weight, "Conv2d weight is nullptr");
    EXIT_ERROR_CHECK_EQ(nullptr, weight->ptr, "Conv2d weight ptr is nullptr");

    if (conv->biasEnabled()) {
        const core::Data_t* bias = conv->param(core::ParamRole::BIAS);
        EXIT_ERROR_CHECK_EQ(nullptr, bias, "Conv2d bias is nullptr");
        EXIT_ERROR_CHECK_EQ(nullptr, bias->ptr, "Conv2d bias ptr is nullptr");
    }
    layer->setCache(build_conv2d_plan(conv), delete_conv2d_plan);
}

void execute_conv2d(core::Layer *layer, core::ExecContext_t *ctx) {
    auto* conv = layer->impl<core::ConvNd_L>();
    auto* plan = layer->cache<Conv2dPlan>();
    EXIT_ERROR_CHECK_EQ(nullptr, plan, "Conv2d plan is nullptr");

    core::Value_t& input = conv->input(0);
    core::Value_t& output = conv->output();
    require_runtime_input(input, "Conv2d input");

    const FLOAT* in_ptr = static_cast<const FLOAT*>(input.data.ptr);
    FLOAT* out_ptr = static_cast<FLOAT*>(output.data.ptr);
    FLOAT* col_buf = static_cast<FLOAT*>(require_workspace(layer, ctx, plan->workspace_bytes));

    for (UINT n = 0; n < plan->batch; ++n) {
        const FLOAT* batch_input = in_ptr + n * plan->input_batch_stride;
        FLOAT* batch_output = out_ptr + n * plan->output_batch_stride;

        for (UINT g = 0; g < plan->groups; ++g) {
            const FLOAT* group_input = batch_input + g * plan->group_input_stride;
            const FLOAT* group_weight = plan->weight + g * plan->group_weight_stride;
            FLOAT* group_output = batch_output + g * plan->group_output_stride;

            math::im2col_nchw(group_input,
                              plan->in_c_pg,
                              plan->in_h,
                              plan->in_w,
                              plan->k_h,
                              plan->k_w,
                              plan->s_h,
                              plan->s_w,
                              plan->pad_t,
                              plan->pad_l,
                              plan->d_h,
                              plan->d_w,
                              plan->out_h,
                              plan->out_w,
                              col_buf);

            math::gemm_nn(group_weight,
                          col_buf,
                          group_output,
                          plan->out_c_pg,
                          plan->N,
                          plan->K);

            if (plan->bias) {
                math::add_bias_channelwise(group_output,
                                           plan->bias + g * plan->out_c_pg,
                                           plan->out_c_pg,
                                           plan->out_spatial);
            }
        }
    }
}

} // namespace cpu
} // namespace backend
} // namespace Kernel
