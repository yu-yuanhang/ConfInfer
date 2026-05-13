#include "cpu_ops.h"
#include "math/math_utils_cpu.h"
#include <convolution.h>

namespace Kernel {
namespace backend {
namespace cpu {

void execute_conv2d(core::LayerSlice *ls, ThreadCtx_t *ctx) {
    EXIT_ERROR_CHECK_EQ(nullptr, ls, "LayerSlice is nullptr");
    auto* conv = dynamic_cast<core::ConvNd_L*>(ls->layer());
    EXIT_ERROR_CHECK_EQ(nullptr, conv, "Layer is not ConvNd_L");

    core::Value_t& input = conv->input(0);
    core::Value_t& output = conv->output();

    EXIT_ERROR_CHECK_NE(core::DataType::FP32, input.data.dtype, "Conv2d only supports FP32 input");
    EXIT_ERROR_CHECK_NE(core::DataType::FP32, output.data.dtype, "Conv2d only supports FP32 output");
    EXIT_ERROR_CHECK_EQ(nullptr, input.data.ptr, "Conv2d input ptr is nullptr");
    EXIT_ERROR_CHECK_EQ(nullptr, output.data.ptr, "Conv2d output ptr is nullptr");
    EXIT_ERROR_CHECK_NE(4, input.data.shape.ndim, "Conv2d expects NCHW input");
    EXIT_ERROR_CHECK_NE(4, output.data.shape.ndim, "Conv2d expects NCHW output");
    EXIT_ERROR_CHECK_NE(2, conv->spatialDim(), "Conv2d spatial dim must be 2");
    EXIT_ERROR_CHECK_EQ(false, conv->padding().size() == 4 || conv->padding().size() == 2,
        "Conv2d padding must contain 2 or 4 elements");

    const core::Data_t* weight = conv->param(core::ParamRole::WEIGHT);
    EXIT_ERROR_CHECK_EQ(nullptr, weight, "Conv2d weight is nullptr");
    EXIT_ERROR_CHECK_EQ(nullptr, weight->ptr, "Conv2d weight ptr is nullptr");
    EXIT_ERROR_CHECK_NE(core::DataType::FP32, weight->dtype, "Conv2d only supports FP32 weight");

    const core::Data_t* bias = conv->param(core::ParamRole::BIAS);
    if (conv->biasEnabled()) {
        EXIT_ERROR_CHECK_EQ(nullptr, bias, "Conv2d bias is nullptr");
        EXIT_ERROR_CHECK_EQ(nullptr, bias->ptr, "Conv2d bias ptr is nullptr");
        EXIT_ERROR_CHECK_NE(core::DataType::FP32, bias->dtype, "Conv2d only supports FP32 bias");
    }

    const FLOAT* in_ptr = static_cast<const FLOAT*>(input.data.ptr);
    FLOAT* out_ptr = static_cast<FLOAT*>(output.data.ptr);
    const FLOAT* w_ptr = static_cast<const FLOAT*>(weight->ptr);
    const FLOAT* b_ptr = bias ? static_cast<const FLOAT*>(bias->ptr) : nullptr;

    const UINT batch = input.data.shape.dims[0];
    const UINT in_c = input.data.shape.dims[1];
    const UINT in_h = input.data.shape.dims[2];
    const UINT in_w = input.data.shape.dims[3];
    const UINT out_c = output.data.shape.dims[1];
    const UINT out_h = output.data.shape.dims[2];
    const UINT out_w = output.data.shape.dims[3];

    const UINT groups = conv->groups();
    const UINT in_c_pg = conv->inChannelsPerGroup();
    const UINT out_c_pg = conv->outChannelsPerGroup();

    const UINT k_h = conv->kernelSize()[0];
    const UINT k_w = conv->kernelSize()[1];
    const UINT s_h = conv->stride()[0];
    const UINT s_w = conv->stride()[1];
    const UINT d_h = conv->dilation()[0];
    const UINT d_w = conv->dilation()[1];

    INT pad_t = 0, pad_b = 0, pad_l = 0, pad_r = 0;
    if (conv->padding().size() == 2) {
        pad_t = conv->padding()[0];
        pad_b = conv->padding()[0];
        pad_l = conv->padding()[1];
        pad_r = conv->padding()[1];
    } else {
        pad_t = conv->padding()[0];
        pad_b = conv->padding()[1];
        pad_l = conv->padding()[2];
        pad_r = conv->padding()[3];
    }
    (void)pad_b;
    (void)pad_r;

    const UINT out_spatial = out_h * out_w;
    const UINT kernel_area = k_h * k_w;
    const UINT K = in_c_pg * kernel_area;
    const UINT N = out_spatial;

    FLOAT* col_buf = static_cast<FLOAT*>(require_workspace(ls, ctx, K * N * sizeof(FLOAT)));
    const UINT input_batch_stride = in_c * in_h * in_w;
    const UINT output_batch_stride = out_c * out_h * out_w;

    for (UINT n = 0; n < batch; ++n) {
        const FLOAT* batch_input = in_ptr + n * input_batch_stride;
        FLOAT* batch_output = out_ptr + n * output_batch_stride;

        for (UINT g = 0; g < groups; ++g) {
            const FLOAT* group_input = batch_input + g * in_c_pg * in_h * in_w;
            const FLOAT* group_weight = w_ptr + g * out_c_pg * K;
            FLOAT* group_output = batch_output + g * out_c_pg * out_spatial;

            math::im2col_nchw(group_input,
                              in_c_pg,
                              in_h,
                              in_w,
                              k_h,
                              k_w,
                              s_h,
                              s_w,
                              pad_t,
                              pad_l,
                              d_h,
                              d_w,
                              out_h,
                              out_w,
                              col_buf);

            math::gemm_nn(group_weight,
                          col_buf,
                          group_output,
                          out_c_pg,
                          N,
                          K);

            if (b_ptr) {
                math::add_bias_channelwise(group_output,
                                           b_ptr + g * out_c_pg,
                                           out_c_pg,
                                           out_spatial);
            }
        }
    }
}

} // namespace cpu
} // namespace backend
} // namespace Kernel
