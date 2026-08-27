#include "cpu_ops.h"
#include "math/math_utils_cpu.h"
#include <pool.h>

#include <limits>

namespace Kernel {
namespace backend {
namespace cpu {

namespace {
enum class Pool2dFastKind : uint8_t {
    NONE = 0,
    MAXPOOL_2X2_S2,
    AVGPOOL_2X2_S2,
    MAXPOOL_GLOBAL,
    AVGPOOL_GLOBAL,
};

struct Pool2dPlan {
    Pool2dFastKind fast_kind;
};

struct AdaptivePoolPlan {
    UINT in_h;
    UINT in_w;
    UINT out_h;
    UINT out_w;
    std::vector<UINT> h_starts;
    std::vector<UINT> h_ends;
    std::vector<UINT> w_starts;
    std::vector<UINT> w_ends;
};

void delete_adaptive_pool_plan(void* ptr) {
    delete static_cast<AdaptivePoolPlan*>(ptr);
}

void delete_pool2d_plan(void* ptr) {
    delete static_cast<Pool2dPlan*>(ptr);
}

void parse_padding(const std::vector<INT>& padding,
                   INT& pad_t,
                   INT& pad_b,
                   INT& pad_l,
                   INT& pad_r);

Pool2dPlan* build_pool2d_plan(core::PoolNd_L* pool) {
    auto* plan = new(std::nothrow) Pool2dPlan();
    EXIT_ERROR_CHECK_EQ(nullptr, plan, "Pool2d plan allocation failed");
    plan->fast_kind = Pool2dFastKind::NONE;

    const std::vector<UINT>& stride = pool->stride().empty() ? pool->kernelSize() : pool->stride();
    const std::vector<UINT>& dilation = pool->dilation().empty()
        ? std::vector<UINT>{1, 1}
        : pool->dilation();

    INT pad_t = 0;
    INT pad_b = 0;
    INT pad_l = 0;
    INT pad_r = 0;
    parse_padding(pool->padding(), pad_t, pad_b, pad_l, pad_r);

    const bool is_2x2_s2 = pool->kernelSize().size() == 2
        && stride.size() == 2
        && dilation.size() == 2
        && pool->kernelSize()[0] == 2
        && pool->kernelSize()[1] == 2
        && stride[0] == 2
        && stride[1] == 2
        && dilation[0] == 1
        && dilation[1] == 1
        && pad_t == 0
        && pad_b == 0
        && pad_l == 0
        && pad_r == 0
        && !pool->ceilMode();

    if (is_2x2_s2) {
        if (core::LayerType::MAXPOOL2D == pool->type()) {
            plan->fast_kind = Pool2dFastKind::MAXPOOL_2X2_S2;
        } else if (core::LayerType::AVGPOOL2D == pool->type()
                   && 0 == pool->divisorOverride()
                   && pool->countIncludePad()) {
            plan->fast_kind = Pool2dFastKind::AVGPOOL_2X2_S2;
        }
        return plan;
    }

    core::Value_t& input = pool->input(0);
    core::Value_t& output = pool->output(core::OutputKind::Default);
    const UINT in_h = input.data.shape.dims[2];
    const UINT in_w = input.data.shape.dims[3];
    const UINT out_h = output.data.shape.dims[2];
    const UINT out_w = output.data.shape.dims[3];
    const UINT kernel_h = pool->kernelSize()[0];
    const UINT kernel_w = pool->kernelSize()[1];

    const bool is_global_pool = out_h == 1
        && out_w == 1
        && kernel_h == in_h
        && kernel_w == in_w
        && dilation.size() == 2
        && dilation[0] == 1
        && dilation[1] == 1
        && pad_t == 0
        && pad_b == 0
        && pad_l == 0
        && pad_r == 0;

    if (is_global_pool) {
        if (core::LayerType::MAXPOOL2D == pool->type()) {
            plan->fast_kind = Pool2dFastKind::MAXPOOL_GLOBAL;
        } else if (core::LayerType::AVGPOOL2D == pool->type()) {
            const UINT kernel_area = kernel_h * kernel_w;
            if (0 == pool->divisorOverride() || kernel_area == pool->divisorOverride()) {
                plan->fast_kind = Pool2dFastKind::AVGPOOL_GLOBAL;
            }
        }
    }

    return plan;
}

AdaptivePoolPlan* build_adaptive_pool_plan(const core::Value_t& input,
                                           const core::Value_t& output) {
    auto* plan = new(std::nothrow) AdaptivePoolPlan();
    EXIT_ERROR_CHECK_EQ(nullptr, plan, "AdaptivePool plan allocation failed");

    plan->in_h = input.data.shape.dims[2];
    plan->in_w = input.data.shape.dims[3];
    plan->out_h = output.data.shape.dims[2];
    plan->out_w = output.data.shape.dims[3];
    plan->h_starts.resize(plan->out_h);
    plan->h_ends.resize(plan->out_h);
    plan->w_starts.resize(plan->out_w);
    plan->w_ends.resize(plan->out_w);

    math::build_adaptive_pool_bounds(plan->in_h, plan->out_h,
                                     plan->h_starts.data(), plan->h_ends.data());
    math::build_adaptive_pool_bounds(plan->in_w, plan->out_w,
                                     plan->w_starts.data(), plan->w_ends.data());
    return plan;
}

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

core::PoolNd_L* checked_pool2d(core::Layer* layer) {
    EXIT_ERROR_CHECK_EQ(nullptr, layer, "Layer is nullptr");
    auto* pool = dynamic_cast<core::PoolNd_L*>(layer);
    EXIT_ERROR_CHECK_EQ(nullptr, pool, "Layer is not PoolNd_L");
    EXIT_ERROR_CHECK_NE(2, pool->spatialDim(), "Pool layer spatial dim must be 2");
    layer->setImpl(pool);
    return pool;
}

core::AdaptivePool2d_L* checked_adaptive_pool2d(core::Layer* layer) {
    EXIT_ERROR_CHECK_EQ(nullptr, layer, "Layer is nullptr");
    auto* pool = dynamic_cast<core::AdaptivePool2d_L*>(layer);
    EXIT_ERROR_CHECK_EQ(nullptr, pool, "Layer is not AdaptivePool2d_L");
    layer->setImpl(pool);
    return pool;
}

void validate_pool_io(const core::Value_t& input, const core::Value_t& output) {
    EXIT_ERROR_CHECK_EQ(nullptr, output.data.ptr, "Pool2d output ptr is nullptr");
    EXIT_ERROR_CHECK_NE(4, input.data.shape.ndim, "Pool2d expects NCHW input");
    EXIT_ERROR_CHECK_NE(4, output.data.shape.ndim, "Pool2d expects NCHW output");
}

void require_runtime_input(const core::Value_t& value, const char* name) {
    EXIT_ERROR_CHECK_EQ(nullptr, value.data.ptr, "%s ptr is nullptr", name);
}

void execute_maxpool2d_impl(core::PoolNd_L* pool) {
    core::Value_t& input = pool->input(0);
    core::Value_t& output = pool->output(core::OutputKind::Default);
    require_runtime_input(input, "MaxPool2d input");
    int32_t* indices_ptr = nullptr;
    if (pool->returnIndices()) {
        core::Value_t& indices = pool->output(core::OutputKind::Indices);
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
    require_runtime_input(input, "AvgPool2d input");

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

void execute_maxpool2d_fast_2x2s2(core::PoolNd_L* pool) {
    core::Value_t& input = pool->input(0);
    core::Value_t& output = pool->output(core::OutputKind::Default);
    require_runtime_input(input, "MaxPool2d input");
    int32_t* indices_ptr = nullptr;
    if (pool->returnIndices()) {
        core::Value_t& indices = pool->output(core::OutputKind::Indices);
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
                const UINT ih0 = oh << 1;
                const UINT ih1 = ih0 + 1;
                const FLOAT* row0 = in_c + ih0 * in_w;
                const FLOAT* row1 = in_c + ih1 * in_w;
                FLOAT* out_row = out_c + oh * out_w;
                int32_t* idx_row = (nullptr == idx_c) ? nullptr : (idx_c + oh * out_w);
                for (UINT ow = 0; ow < out_w; ++ow) {
                    const UINT iw0 = ow << 1;
                    const UINT iw1 = iw0 + 1;
                    FLOAT max_value = row0[iw0];
                    int32_t max_index = static_cast<int32_t>(ih0 * in_w + iw0);
                    if (row0[iw1] > max_value) {
                        max_value = row0[iw1];
                        max_index = static_cast<int32_t>(ih0 * in_w + iw1);
                    }
                    if (row1[iw0] > max_value) {
                        max_value = row1[iw0];
                        max_index = static_cast<int32_t>(ih1 * in_w + iw0);
                    }
                    if (row1[iw1] > max_value) {
                        max_value = row1[iw1];
                        max_index = static_cast<int32_t>(ih1 * in_w + iw1);
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

void execute_avgpool2d_fast_2x2s2(core::PoolNd_L* pool) {
    core::Value_t& input = pool->input(0);
    core::Value_t& output = pool->output();
    require_runtime_input(input, "AvgPool2d input");

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
        for (UINT c = 0; c < channels; ++c) {
            const FLOAT* in_c = in_n + c * in_channel_stride;
            FLOAT* out_c = out_n + c * out_channel_stride;
            for (UINT oh = 0; oh < out_h; ++oh) {
                const UINT ih0 = oh << 1;
                const UINT ih1 = ih0 + 1;
                const FLOAT* row0 = in_c + ih0 * in_w;
                const FLOAT* row1 = in_c + ih1 * in_w;
                FLOAT* out_row = out_c + oh * out_w;
                for (UINT ow = 0; ow < out_w; ++ow) {
                    const UINT iw0 = ow << 1;
                    const UINT iw1 = iw0 + 1;
                    out_row[ow] = 0.25f * (row0[iw0] + row0[iw1] + row1[iw0] + row1[iw1]);
                }
            }
        }
    }
}

void execute_maxpool2d_global(core::PoolNd_L* pool) {
    core::Value_t& input = pool->input(0);
    core::Value_t& output = pool->output(core::OutputKind::Default);
    require_runtime_input(input, "MaxPool2d input");
    int32_t* indices_ptr = nullptr;
    if (pool->returnIndices()) {
        core::Value_t& indices = pool->output(core::OutputKind::Indices);
        indices_ptr = static_cast<int32_t*>(indices.data.ptr);
    }

    const FLOAT* in_ptr = static_cast<const FLOAT*>(input.data.ptr);
    FLOAT* out_ptr = static_cast<FLOAT*>(output.data.ptr);
    const UINT batch = input.data.shape.dims[0];
    const UINT channels = input.data.shape.dims[1];
    const UINT in_h = input.data.shape.dims[2];
    const UINT in_w = input.data.shape.dims[3];
    const UINT spatial = in_h * in_w;
    const UINT batch_stride = channels * spatial;

    for (UINT n = 0; n < batch; ++n) {
        const FLOAT* in_n = in_ptr + n * batch_stride;
        FLOAT* out_n = out_ptr + n * channels;
        int32_t* idx_n = (nullptr == indices_ptr) ? nullptr : (indices_ptr + n * channels);
        for (UINT c = 0; c < channels; ++c) {
            const FLOAT* in_c = in_n + c * spatial;
            FLOAT max_value = in_c[0];
            int32_t max_index = 0;
            for (UINT i = 1; i < spatial; ++i) {
                if (in_c[i] > max_value) {
                    max_value = in_c[i];
                    max_index = static_cast<int32_t>(i);
                }
            }
            out_n[c] = max_value;
            if (nullptr != idx_n) {
                idx_n[c] = max_index;
            }
        }
    }
}

void execute_avgpool2d_global(core::PoolNd_L* pool) {
    core::Value_t& input = pool->input(0);
    core::Value_t& output = pool->output();
    require_runtime_input(input, "AvgPool2d input");

    const FLOAT* in_ptr = static_cast<const FLOAT*>(input.data.ptr);
    FLOAT* out_ptr = static_cast<FLOAT*>(output.data.ptr);
    const UINT batch = input.data.shape.dims[0];
    const UINT channels = input.data.shape.dims[1];
    const UINT in_h = input.data.shape.dims[2];
    const UINT in_w = input.data.shape.dims[3];
    const UINT spatial = in_h * in_w;
    const UINT batch_stride = channels * spatial;
    const FLOAT scale = 1.0f / static_cast<FLOAT>(spatial);

    for (UINT n = 0; n < batch; ++n) {
        const FLOAT* in_n = in_ptr + n * batch_stride;
        FLOAT* out_n = out_ptr + n * channels;
        for (UINT c = 0; c < channels; ++c) {
            const FLOAT* in_c = in_n + c * spatial;
            FLOAT sum = 0.0f;
            for (UINT i = 0; i < spatial; ++i) {
                sum += in_c[i];
            }
            out_n[c] = sum * scale;
        }
    }
}

} // namespace

void prepare_maxpool2d(core::Layer *layer) {
    auto* pool = checked_pool2d(layer);
    core::Value_t& input = pool->input(0);
    core::Value_t& output = pool->output(core::OutputKind::Default);
    validate_pool_io(input, output);
    if (pool->returnIndices()) {
        core::Value_t& indices = pool->output(core::OutputKind::Indices);
        EXIT_ERROR_CHECK_NE(core::DataType::INT32, indices.data.dtype,
            "MaxPool2d indices only support INT32 output");
        EXIT_ERROR_CHECK_EQ(nullptr, indices.data.ptr, "MaxPool2d indices ptr is nullptr");
    }
    layer->setCache(build_pool2d_plan(pool), delete_pool2d_plan);
}

void execute_maxpool2d(core::Layer *layer, core::ExecContext_t *ctx) {
    (void)ctx;
    auto* plan = layer->cache<Pool2dPlan>();
    EXIT_ERROR_CHECK_EQ(nullptr, plan, "MaxPool2d plan is nullptr");
    switch (plan->fast_kind) {
        case Pool2dFastKind::MAXPOOL_2X2_S2:
            execute_maxpool2d_fast_2x2s2(layer->impl<core::PoolNd_L>());
            return;
        case Pool2dFastKind::MAXPOOL_GLOBAL:
            execute_maxpool2d_global(layer->impl<core::PoolNd_L>());
            return;
        default:
            execute_maxpool2d_impl(layer->impl<core::PoolNd_L>());
            return;
    }
}

void prepare_avgpool2d(core::Layer *layer) {
    auto* pool = checked_pool2d(layer);
    validate_pool_io(pool->input(0), pool->output());
    layer->setCache(build_pool2d_plan(pool), delete_pool2d_plan);
}

void execute_avgpool2d(core::Layer *layer, core::ExecContext_t *ctx) {
    (void)ctx;
    auto* plan = layer->cache<Pool2dPlan>();
    EXIT_ERROR_CHECK_EQ(nullptr, plan, "AvgPool2d plan is nullptr");
    switch (plan->fast_kind) {
        case Pool2dFastKind::AVGPOOL_2X2_S2:
            execute_avgpool2d_fast_2x2s2(layer->impl<core::PoolNd_L>());
            return;
        case Pool2dFastKind::AVGPOOL_GLOBAL:
            execute_avgpool2d_global(layer->impl<core::PoolNd_L>());
            return;
        default:
            execute_avgpool2d_impl(layer->impl<core::PoolNd_L>());
            return;
    }
}

void prepare_adaptiveavgpool2d(core::Layer *layer) {
    auto* pool = checked_adaptive_pool2d(layer);
    core::Value_t& input = pool->input(0);
    core::Value_t& output = pool->output();
    validate_pool_io(input, output);
    // build_adaptive_pool_plan(input, output) 这里直接初始化 cache
    // 下面的实现方式 就一般保持一致
    layer->setCache(build_adaptive_pool_plan(input, output), delete_adaptive_pool_plan);
}

void execute_adaptiveavgpool2d(core::Layer *layer, core::ExecContext_t *ctx) {
    (void)ctx;
    auto* pool = layer->impl<core::AdaptivePool2d_L>();
    auto* plan = layer->cache<AdaptivePoolPlan>();
    core::Value_t& input = pool->input(0);
    core::Value_t& output = pool->output();
    EXIT_ERROR_CHECK_EQ(nullptr, plan, "AdaptiveAvgPool2d plan is nullptr");
    require_runtime_input(input, "AdaptiveAvgPool2d input");
    math::adaptive_avgpool2d_nchw(static_cast<const FLOAT*>(input.data.ptr),
                                  static_cast<FLOAT*>(output.data.ptr),
                                  input.data.shape.dims[0],
                                  input.data.shape.dims[1],
                                  input.data.shape.dims[2],
                                  input.data.shape.dims[3],
                                  output.data.shape.dims[2],
                                  output.data.shape.dims[3],
                                  plan->h_starts.data(),
                                  plan->h_ends.data(),
                                  plan->w_starts.data(),
                                  plan->w_ends.data());
}

void prepare_adaptivemaxpool2d(core::Layer *layer) {
    auto* pool = checked_adaptive_pool2d(layer);
    core::Value_t& input = pool->input(0);
    core::Value_t& output = pool->output(core::OutputKind::Default);
    validate_pool_io(input, output);
    if (pool->returnIndices()) {
        core::Value_t& indices = pool->output(core::OutputKind::Indices);
        EXIT_ERROR_CHECK_NE(core::DataType::INT32, indices.data.dtype,
            "AdaptiveMaxPool2d indices only support INT32 output");
        EXIT_ERROR_CHECK_EQ(nullptr, indices.data.ptr, "AdaptiveMaxPool2d indices ptr is nullptr");
    }
    layer->setCache(build_adaptive_pool_plan(input, output), delete_adaptive_pool_plan);
}

void execute_adaptivemaxpool2d(core::Layer *layer, core::ExecContext_t *ctx) {
    (void)ctx;
    auto* pool = layer->impl<core::AdaptivePool2d_L>();
    auto* plan = layer->cache<AdaptivePoolPlan>();
    core::Value_t& input = pool->input(0);
    core::Value_t& output = pool->output(core::OutputKind::Default);
    EXIT_ERROR_CHECK_EQ(nullptr, plan, "AdaptiveMaxPool2d plan is nullptr");
    require_runtime_input(input, "AdaptiveMaxPool2d input");
    int32_t* indices_ptr = nullptr;
    if (pool->returnIndices()) {
        core::Value_t& indices = pool->output(core::OutputKind::Indices);
        indices_ptr = static_cast<int32_t*>(indices.data.ptr);
    }
    math::adaptive_maxpool2d_nchw(static_cast<const FLOAT*>(input.data.ptr),
                                  static_cast<FLOAT*>(output.data.ptr),
                                  input.data.shape.dims[0],
                                  input.data.shape.dims[1],
                                  input.data.shape.dims[2],
                                  input.data.shape.dims[3],
                                  output.data.shape.dims[2],
                                  output.data.shape.dims[3],
                                  plan->h_starts.data(),
                                  plan->h_ends.data(),
                                  plan->w_starts.data(),
                                  plan->w_ends.data(),
                                  indices_ptr);
}

} // namespace cpu
} // namespace backend
} // namespace Kernel
