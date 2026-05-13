#include "cpu_ops.h"

#include <cmath>

#include "math/math_utils_cpu.h"
#include <normalization.h>

namespace Kernel {
namespace backend {
namespace cpu {

namespace {
const FLOAT* fp32_ptr_or_null(const core::Data_t* data) {
    if (nullptr == data || nullptr == data->ptr) {
        return nullptr;
    }
    EXIT_ERROR_CHECK_NE(core::DataType::FP32, data->dtype, "Only FP32 normalization params are supported");
    return static_cast<const FLOAT*>(data->ptr);
}

void require_fp32(const core::Value_t& input, const core::Value_t& output) {
    EXIT_ERROR_CHECK_NE(core::DataType::FP32, input.data.dtype, "Only FP32 normalization input is supported");
    EXIT_ERROR_CHECK_NE(core::DataType::FP32, output.data.dtype, "Only FP32 normalization output is supported");
    EXIT_ERROR_CHECK_EQ(nullptr, input.data.ptr, "Normalization input ptr is nullptr");
    EXIT_ERROR_CHECK_EQ(nullptr, output.data.ptr, "Normalization output ptr is nullptr");
}
} // namespace

void execute_batchnorm2d(core::LayerSlice *ls, ThreadCtx_t *ctx) {
    (void)ctx;
    EXIT_ERROR_CHECK_EQ(nullptr, ls, "LayerSlice is nullptr");
    auto* bn = dynamic_cast<core::BatchNorm2d_L*>(ls->layer());
    EXIT_ERROR_CHECK_EQ(nullptr, bn, "Layer is not BatchNorm2d_L");

    core::Value_t& input = bn->input(0);
    core::Value_t& output = bn->output();
    require_fp32(input, output);

    FLOAT* in_ptr = static_cast<FLOAT*>(input.data.ptr);
    FLOAT* out_ptr = static_cast<FLOAT*>(output.data.ptr);

    const core::Data_t* weight = bn->param(core::ParamRole::WEIGHT);
    const core::Data_t* bias = bn->param(core::ParamRole::BIAS);
    const core::Data_t* running_mean = bn->param(core::ParamRole::RUNNING_MEAN);
    const core::Data_t* running_var = bn->param(core::ParamRole::RUNNING_VAR);
    const FLOAT* weight_ptr = fp32_ptr_or_null(weight);
    const FLOAT* bias_ptr = fp32_ptr_or_null(bias);
    const FLOAT* running_mean_ptr = fp32_ptr_or_null(running_mean);
    const FLOAT* running_var_ptr = fp32_ptr_or_null(running_var);

    const core::DataShape_t& shape = input.data.shape;
    EXIT_ERROR_CHECK_NE(4, shape.ndim, "BatchNorm2d expects NCHW input");
    const UINT channels = shape.dims[1];
    const UINT batch = shape.dims[0];
    UINT spatial = 1;
    for (UINT i = 2; i < shape.ndim; ++i) {
        spatial *= shape.dims[i];
    }

    if (bn->trackRunningStats()) {
        for (UINT b = 0; b < batch; ++b) {
            const UINT batch_base = b * channels * spatial;
            for (UINT c = 0; c < channels; ++c) {
                const UINT base = batch_base + c * spatial;
                math::normalize_affine_scalar_fp32(in_ptr + base,
                                                   out_ptr + base,
                                                   spatial,
                                                   running_mean_ptr ? running_mean_ptr[c] : 0.0f,
                                                   running_var_ptr ? running_var_ptr[c] : 1.0f,
                                                   bn->eps(),
                                                   weight_ptr ? weight_ptr[c] : 1.0f,
                                                   bias_ptr ? bias_ptr[c] : 0.0f);
            }
        }
        return;
    }

    for (UINT b = 0; b < batch; ++b) {
        const UINT batch_base = b * channels * spatial;
        for (UINT c = 0; c < channels; ++c) {
            const UINT base = batch_base + c * spatial;
            const FLOAT mean = math::mean_fp32(in_ptr + base, spatial);
            const FLOAT var = math::variance_fp32(in_ptr + base, spatial, mean);
            math::normalize_affine_scalar_fp32(in_ptr + base,
                                               out_ptr + base,
                                               spatial,
                                               mean,
                                               var,
                                               bn->eps(),
                                               weight_ptr ? weight_ptr[c] : 1.0f,
                                               bias_ptr ? bias_ptr[c] : 0.0f);
        }
    }
}

void execute_layernorm(core::LayerSlice *ls, ThreadCtx_t *ctx) {
    (void)ctx;
    EXIT_ERROR_CHECK_EQ(nullptr, ls, "LayerSlice is nullptr");
    auto* ln = dynamic_cast<core::LayerNorm_L*>(ls->layer());
    EXIT_ERROR_CHECK_EQ(nullptr, ln, "Layer is not LayerNorm_L");

    core::Value_t& input = ln->input(0);
    core::Value_t& output = ln->output();
    require_fp32(input, output);

    FLOAT* in_ptr = static_cast<FLOAT*>(input.data.ptr);
    FLOAT* out_ptr = static_cast<FLOAT*>(output.data.ptr);

    const core::Data_t* weight = ln->param(core::ParamRole::WEIGHT);
    const core::Data_t* bias = ln->param(core::ParamRole::BIAS);
    const FLOAT* weight_ptr = fp32_ptr_or_null(weight);
    const FLOAT* bias_ptr = fp32_ptr_or_null(bias);

    const std::vector<UINT>& normalized_shape = ln->normalizedShape();
    UINT norm_size = 1;
    for (auto it = normalized_shape.begin(); it != normalized_shape.end(); ++it) {
        norm_size *= *it;
    }
    EXIT_ERROR_CHECK_EQ(0, norm_size, "LayerNorm norm_size is zero");
    EXIT_ERROR_CHECK_NE(0, input.data.shape.size % norm_size, "LayerNorm input size mismatch");

    UINT outer = input.data.shape.size / norm_size;
    for (UINT o = 0; o < outer; ++o) {
        const UINT base = o * norm_size;
        const FLOAT mean = math::mean_fp32(in_ptr + base, norm_size);
        const FLOAT var = math::variance_fp32(in_ptr + base, norm_size, mean);
        math::normalize_affine_fp32(in_ptr + base,
                                    out_ptr + base,
                                    norm_size,
                                    mean,
                                    var,
                                    ln->eps(),
                                    weight_ptr,
                                    bias_ptr);
    }
}

void execute_groupnorm(core::LayerSlice *ls, ThreadCtx_t *ctx) {
    (void)ctx;
    EXIT_ERROR_CHECK_EQ(nullptr, ls, "LayerSlice is nullptr");
    auto* gn = dynamic_cast<core::GroupNorm_L*>(ls->layer());
    EXIT_ERROR_CHECK_EQ(nullptr, gn, "Layer is not GroupNorm_L");

    core::Value_t& input = gn->input(0);
    core::Value_t& output = gn->output();
    require_fp32(input, output);

    FLOAT* in_ptr = static_cast<FLOAT*>(input.data.ptr);
    FLOAT* out_ptr = static_cast<FLOAT*>(output.data.ptr);

    const core::Data_t* weight = gn->param(core::ParamRole::WEIGHT);
    const core::Data_t* bias = gn->param(core::ParamRole::BIAS);
    const FLOAT* weight_ptr = fp32_ptr_or_null(weight);
    const FLOAT* bias_ptr = fp32_ptr_or_null(bias);

    const core::DataShape_t& shape = input.data.shape;
    const UINT channels = shape.dims[1];
    const UINT batch = shape.dims[0];
    const UINT group_size = channels / gn->numGroups();

    UINT spatial = 1;
    for (UINT i = 2; i < shape.ndim; ++i) {
        spatial *= shape.dims[i];
    }
    const UINT batch_stride = channels * spatial;
    const UINT groups = gn->numGroups();

    for (UINT b = 0; b < batch; ++b) {
        const UINT batch_base = b * batch_stride;
        for (UINT g = 0; g < groups; ++g) {
            const UINT count = group_size * spatial;
            const UINT group_channel_base = g * group_size;
            const UINT group_base = batch_base + group_channel_base * spatial;
            const FLOAT mean = math::mean_fp32(in_ptr + group_base, count);
            const FLOAT var = math::variance_fp32(in_ptr + group_base, count, mean);
            for (UINT gc = 0; gc < group_size; ++gc) {
                const UINT c = group_channel_base + gc;
                const UINT base = batch_base + c * spatial;
                math::normalize_affine_scalar_fp32(in_ptr + base,
                                                   out_ptr + base,
                                                   spatial,
                                                   mean,
                                                   var,
                                                   gn->eps(),
                                                   weight_ptr ? weight_ptr[c] : 1.0f,
                                                   bias_ptr ? bias_ptr[c] : 0.0f);
            }
        }
    }
}

} // namespace cpu
} // namespace backend
} // namespace Kernel
