#include "cpu_ops.h"
#include "math/math_utils_cpu.h"
#include <pool.h>

#include <limits>

namespace Kernel {
namespace backend {
namespace cpu {

namespace {

void parse_padding(const std::vector<INT>& padding,
                   INT& pad_t,
                   INT& pad_b,
                   INT& pad_l,
                   INT& pad_r) {
    EXIT_ERROR_CHECK_EQ(false, padding.size() == 2 || padding.size() == 4,
        "Pool2d padding must contain 2 or 4 elements");

    if (2 == padding.size()) {
        pad_t = padding[0];
        pad_b = padding[0];
        pad_l = padding[1];
        pad_r = padding[1];
        return;
    }

    pad_t = padding[0];
    pad_b = padding[1];
    pad_l = padding[2];
    pad_r = padding[3];
}

core::PoolNd_L* checked_pool2d(core::LayerSlice* ls) {
    EXIT_ERROR_CHECK_EQ(nullptr, ls, "LayerSlice is nullptr");
    auto* pool = dynamic_cast<core::PoolNd_L*>(ls->layer());
    EXIT_ERROR_CHECK_EQ(nullptr, pool, "Layer is not PoolNd_L");
    EXIT_ERROR_CHECK_NE(2, pool->spatialDim(), "Pool layer spatial dim must be 2");
    return pool;
}

core::AdaptivePool2d_L* checked_adaptive_pool2d(core::LayerSlice* ls) {
    EXIT_ERROR_CHECK_EQ(nullptr, ls, "LayerSlice is nullptr");
    auto* pool = dynamic_cast<core::AdaptivePool2d_L*>(ls->layer());
    EXIT_ERROR_CHECK_EQ(nullptr, pool, "Layer is not AdaptivePool2d_L");
    return pool;
}

void validate_pool_io(const core::Value_t& input, const core::Value_t& output) {
    EXIT_ERROR_CHECK_NE(core::DataType::FP32, input.data.dtype, "Pool2d only supports FP32 input");
    EXIT_ERROR_CHECK_NE(core::DataType::FP32, output.data.dtype, "Pool2d only supports FP32 output");
    EXIT_ERROR_CHECK_EQ(nullptr, input.data.ptr, "Pool2d input ptr is nullptr");
    EXIT_ERROR_CHECK_EQ(nullptr, output.data.ptr, "Pool2d output ptr is nullptr");
    EXIT_ERROR_CHECK_NE(4, input.data.shape.ndim, "Pool2d expects NCHW input");
    EXIT_ERROR_CHECK_NE(4, output.data.shape.ndim, "Pool2d expects NCHW output");
}

void execute_maxpool2d_impl(core::PoolNd_L* pool) {
    core::Value_t& input = pool->input(0);
    core::Value_t& output = pool->output(core::OutputKind::Default);
    validate_pool_io(input, output);
    int32_t* indices_ptr = nullptr;
    if (pool->returnIndices()) {
        core::Value_t& indices = pool->output(core::OutputKind::Indices);
        EXIT_ERROR_CHECK_NE(core::DataType::INT32, indices.data.dtype,
            "MaxPool2d indices only support INT32 output");
        EXIT_ERROR_CHECK_EQ(nullptr, indices.data.ptr, "MaxPool2d indices ptr is nullptr");
        indices_ptr = static_cast<int32_t*>(indices.data.ptr);
    }

    const FLOAT* in_ptr = static_cast<const FLOAT*>(input.data.ptr);
    FLOAT* out_ptr = static_cast<FLOAT*>(output.data.ptr);

    const UINT batch = input.data.shape.dims[0];
    const UINT channels = input.data.shape.dims[1];
    const UINT in_h = input.data.shape.dims[2];
    const UINT in_w = input.data.shape.dims[3];
    const UINT out_h = output.data.shape.dims[2];
    const UINT out_w = output.data.shape.dims[3];

    const UINT kernel_h = pool->kernelSize()[0];
    const UINT kernel_w = pool->kernelSize()[1];
    const std::vector<UINT>& stride = pool->stride().empty() ? pool->kernelSize() : pool->stride();
    const UINT stride_h = stride[0];
    const UINT stride_w = stride[1];
    const std::vector<UINT>& dilation = pool->dilation().empty()
        ? std::vector<UINT>{1, 1}
        : pool->dilation();
    const UINT dilation_h = dilation[0];
    const UINT dilation_w = dilation[1];

    INT pad_t = 0;
    INT pad_b = 0;
    INT pad_l = 0;
    INT pad_r = 0;
    parse_padding(pool->padding(), pad_t, pad_b, pad_l, pad_r);
    (void)pad_b;
    (void)pad_r;

    const UINT in_channel_stride = in_h * in_w;
    const UINT out_channel_stride = out_h * out_w;
    const UINT input_batch_stride = channels * in_channel_stride;
    const UINT output_batch_stride = channels * out_channel_stride;
    const UINT indices_batch_stride = channels * out_channel_stride;

    for (UINT n = 0; n < batch; ++n) {
        const FLOAT* in_n = in_ptr + n * input_batch_stride;
        FLOAT* out_n = out_ptr + n * output_batch_stride;
        int32_t* idx_n = (nullptr == indices_ptr) ? nullptr : (indices_ptr + n * indices_batch_stride);
        for (UINT c = 0; c < channels; ++c) {
            const FLOAT* in_c = in_n + c * in_channel_stride;
            FLOAT* out_c = out_n + c * out_channel_stride;
            int32_t* idx_c = (nullptr == idx_n) ? nullptr : (idx_n + c * out_channel_stride);

            for (UINT oh = 0; oh < out_h; ++oh) {
                const INT base_h = static_cast<INT>(oh * stride_h) - pad_t;
                FLOAT* out_row = out_c + oh * out_w;
                int32_t* idx_row = (nullptr == idx_c) ? nullptr : (idx_c + oh * out_w);
                for (UINT ow = 0; ow < out_w; ++ow) {
                    const INT base_w = static_cast<INT>(ow * stride_w) - pad_l;
                    FLOAT max_value = -std::numeric_limits<FLOAT>::infinity();
                    int32_t max_index = -1;
                    bool has_value = false;

                    for (UINT kh = 0; kh < kernel_h; ++kh) {
                        const INT ih = base_h + static_cast<INT>(kh * dilation_h);
                        if (ih < 0 || ih >= static_cast<INT>(in_h)) {
                            continue;
                        }
                        const FLOAT* in_row = in_c + static_cast<UINT>(ih) * in_w;
                        for (UINT kw = 0; kw < kernel_w; ++kw) {
                            const INT iw = base_w + static_cast<INT>(kw * dilation_w);
                            if (iw < 0 || iw >= static_cast<INT>(in_w)) {
                                continue;
                            }
                            const FLOAT value = in_row[static_cast<UINT>(iw)];
                            if (!has_value || value > max_value) {
                                max_value = value;
                                max_index = static_cast<int32_t>(static_cast<UINT>(ih) * in_w + static_cast<UINT>(iw));
                                has_value = true;
                            }
                        }
                    }

                    out_row[ow] = has_value ? max_value : 0.0f;
                    if (nullptr != idx_row) {
                        idx_row[ow] = has_value ? max_index : -1;
                    }
                }
            }
        }
    }
}

void execute_avgpool2d_impl(core::PoolNd_L* pool) {
    core::Value_t& input = pool->input(0);
    core::Value_t& output = pool->output();
    validate_pool_io(input, output);

    const FLOAT* in_ptr = static_cast<const FLOAT*>(input.data.ptr);
    FLOAT* out_ptr = static_cast<FLOAT*>(output.data.ptr);

    const UINT batch = input.data.shape.dims[0];
    const UINT channels = input.data.shape.dims[1];
    const UINT in_h = input.data.shape.dims[2];
    const UINT in_w = input.data.shape.dims[3];
    const UINT out_h = output.data.shape.dims[2];
    const UINT out_w = output.data.shape.dims[3];

    const UINT kernel_h = pool->kernelSize()[0];
    const UINT kernel_w = pool->kernelSize()[1];
    const std::vector<UINT>& stride = pool->stride().empty() ? pool->kernelSize() : pool->stride();
    const UINT stride_h = stride[0];
    const UINT stride_w = stride[1];

    INT pad_t = 0;
    INT pad_b = 0;
    INT pad_l = 0;
    INT pad_r = 0;
    parse_padding(pool->padding(), pad_t, pad_b, pad_l, pad_r);
    (void)pad_b;
    (void)pad_r;

    const UINT in_channel_stride = in_h * in_w;
    const UINT out_channel_stride = out_h * out_w;
    const UINT input_batch_stride = channels * in_channel_stride;
    const UINT output_batch_stride = channels * out_channel_stride;
    const UINT kernel_area = kernel_h * kernel_w;
    const UINT divisor_override = pool->divisorOverride();
    const BOOL count_include_pad = pool->countIncludePad();

    for (UINT n = 0; n < batch; ++n) {
        const FLOAT* in_n = in_ptr + n * input_batch_stride;
        FLOAT* out_n = out_ptr + n * output_batch_stride;
        for (UINT c = 0; c < channels; ++c) {
            const FLOAT* in_c = in_n + c * in_channel_stride;
            FLOAT* out_c = out_n + c * out_channel_stride;

            for (UINT oh = 0; oh < out_h; ++oh) {
                const INT base_h = static_cast<INT>(oh * stride_h) - pad_t;
                FLOAT* out_row = out_c + oh * out_w;
                for (UINT ow = 0; ow < out_w; ++ow) {
                    const INT base_w = static_cast<INT>(ow * stride_w) - pad_l;
                    FLOAT sum = 0.0f;
                    UINT valid_count = 0;

                    for (UINT kh = 0; kh < kernel_h; ++kh) {
                        const INT ih = base_h + static_cast<INT>(kh);
                        if (ih < 0 || ih >= static_cast<INT>(in_h)) {
                            continue;
                        }
                        const FLOAT* in_row = in_c + static_cast<UINT>(ih) * in_w;
                        for (UINT kw = 0; kw < kernel_w; ++kw) {
                            const INT iw = base_w + static_cast<INT>(kw);
                            if (iw < 0 || iw >= static_cast<INT>(in_w)) {
                                continue;
                            }
                            sum += in_row[static_cast<UINT>(iw)];
                            ++valid_count;
                        }
                    }

                    UINT divisor = divisor_override;
                    if (0 == divisor) {
                        divisor = count_include_pad ? kernel_area : valid_count;
                    }
                    EXIT_ERROR_CHECK_EQ(0, divisor, "AvgPool2d divisor must be > 0");
                    out_row[ow] = sum / static_cast<FLOAT>(divisor);
                }
            }
        }
    }
}

} // namespace

void execute_maxpool2d(core::LayerSlice *ls, ThreadCtx_t *ctx) {
    (void)ctx;
    execute_maxpool2d_impl(checked_pool2d(ls));
}

void execute_avgpool2d(core::LayerSlice *ls, ThreadCtx_t *ctx) {
    (void)ctx;
    execute_avgpool2d_impl(checked_pool2d(ls));
}

void execute_adaptiveavgpool2d(core::LayerSlice *ls, ThreadCtx_t *ctx) {
    (void)ctx;
    auto* pool = checked_adaptive_pool2d(ls);
    core::Value_t& input = pool->input(0);
    core::Value_t& output = pool->output();
    validate_pool_io(input, output);
    math::adaptive_avgpool2d_nchw(static_cast<const FLOAT*>(input.data.ptr),
                                  static_cast<FLOAT*>(output.data.ptr),
                                  input.data.shape.dims[0],
                                  input.data.shape.dims[1],
                                  input.data.shape.dims[2],
                                  input.data.shape.dims[3],
                                  output.data.shape.dims[2],
                                  output.data.shape.dims[3]);
}

void execute_adaptivemaxpool2d(core::LayerSlice *ls, ThreadCtx_t *ctx) {
    (void)ctx;
    auto* pool = checked_adaptive_pool2d(ls);
    core::Value_t& input = pool->input(0);
    core::Value_t& output = pool->output(core::OutputKind::Default);
    validate_pool_io(input, output);
    int32_t* indices_ptr = nullptr;
    if (pool->returnIndices()) {
        core::Value_t& indices = pool->output(core::OutputKind::Indices);
        EXIT_ERROR_CHECK_NE(core::DataType::INT32, indices.data.dtype,
            "AdaptiveMaxPool2d indices only support INT32 output");
        EXIT_ERROR_CHECK_EQ(nullptr, indices.data.ptr, "AdaptiveMaxPool2d indices ptr is nullptr");
        indices_ptr = static_cast<int32_t*>(indices.data.ptr);
    }

    const FLOAT* in_ptr = static_cast<const FLOAT*>(input.data.ptr);
    FLOAT* out_ptr = static_cast<FLOAT*>(output.data.ptr);
    const UINT batch = input.data.shape.dims[0];
    const UINT channels = input.data.shape.dims[1];
    const UINT in_h = input.data.shape.dims[2];
    const UINT in_w = input.data.shape.dims[3];
    const UINT out_h = output.data.shape.dims[2];
    const UINT out_w = output.data.shape.dims[3];
    const UINT in_channel_stride = in_h * in_w;
    const UINT out_channel_stride = out_h * out_w;
    const UINT input_batch_stride = channels * in_channel_stride;
    const UINT output_batch_stride = channels * out_channel_stride;

    for (UINT n = 0; n < batch; ++n) {
        const FLOAT* in_n = in_ptr + n * input_batch_stride;
        FLOAT* out_n = out_ptr + n * output_batch_stride;
        int32_t* idx_n = (nullptr == indices_ptr) ? nullptr : (indices_ptr + n * output_batch_stride);
        for (UINT c = 0; c < channels; ++c) {
            const FLOAT* in_c = in_n + c * in_channel_stride;
            FLOAT* out_c = out_n + c * out_channel_stride;
            int32_t* idx_c = (nullptr == idx_n) ? nullptr : (idx_n + c * out_channel_stride);
            for (UINT oh = 0; oh < out_h; ++oh) {
                const UINT h_start = (oh * in_h) / out_h;
                const UINT h_end = ((oh + 1) * in_h + out_h - 1) / out_h;
                FLOAT* out_row = out_c + oh * out_w;
                int32_t* idx_row = (nullptr == idx_c) ? nullptr : (idx_c + oh * out_w);
                for (UINT ow = 0; ow < out_w; ++ow) {
                    const UINT w_start = (ow * in_w) / out_w;
                    const UINT w_end = ((ow + 1) * in_w + out_w - 1) / out_w;
                    FLOAT max_value = -std::numeric_limits<FLOAT>::infinity();
                    int32_t max_index = -1;
                    for (UINT ih = h_start; ih < h_end; ++ih) {
                        const FLOAT* in_row = in_c + ih * in_w;
                        for (UINT iw = w_start; iw < w_end; ++iw) {
                            const FLOAT value = in_row[iw];
                            if (value > max_value) {
                                max_value = value;
                                max_index = static_cast<int32_t>(ih * in_w + iw);
                            }
                        }
                    }
                    out_row[ow] = max_value;
                    if (nullptr != idx_row) {
                        idx_row[ow] = max_index;
                    }
                }
            }
        }
    }
}

} // namespace cpu
} // namespace backend
} // namespace Kernel
