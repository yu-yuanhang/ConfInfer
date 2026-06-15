#include "cpu_ops.h"

#include <cmath>

#include "math/math_utils_cpu.h"
#include <normalization.h>

namespace Kernel {
namespace backend {
namespace cpu {

namespace {
struct BatchNorm2dPlan {
    const FLOAT* weight;
    const FLOAT* bias;
    const FLOAT* running_mean;
    const FLOAT* running_var;
    UINT batch;
    UINT channels;
    UINT spatial;
    UINT batch_stride;
    FLOAT eps;
    BOOL track_running_stats;
};

struct LayerNormPlan {
    const FLOAT* weight;
    const FLOAT* bias;
    UINT norm_size;
    UINT outer;
    FLOAT eps;
};

struct GroupNormPlan {
    const FLOAT* weight;
    const FLOAT* bias;
    UINT batch;
    UINT channels;
    UINT groups;
    UINT group_size;
    UINT spatial;
    UINT batch_stride;
    UINT group_count;
    FLOAT eps;
};

template <typename LayerT>
LayerT* checked_layer(core::LayerSlice* ls, const char* name) {
    EXIT_ERROR_CHECK_EQ(nullptr, ls, "LayerSlice is nullptr");
    auto* layer = dynamic_cast<LayerT*>(ls->layer());
    EXIT_ERROR_CHECK_EQ(nullptr, layer, "%s layer type mismatch", name);
    ls->setImpl(layer);
    return layer;
}

const FLOAT* fp32_ptr_or_null(const core::Data_t* data) {
    if (nullptr == data || nullptr == data->ptr) {
        return nullptr;
    }
    return static_cast<const FLOAT*>(data->ptr);
}

void require_ready_output(const core::Value_t& output) {
    EXIT_ERROR_CHECK_EQ(nullptr, output.data.ptr, "Normalization output ptr is nullptr");
}

void require_runtime_input(const core::Value_t& value, const char* name) {
    EXIT_ERROR_CHECK_EQ(nullptr, value.data.ptr, "%s ptr is nullptr", name);
}

void delete_batchnorm2d_plan(void* ptr) {
    delete static_cast<BatchNorm2dPlan*>(ptr);
}

void delete_layernorm_plan(void* ptr) {
    delete static_cast<LayerNormPlan*>(ptr);
}

void delete_groupnorm_plan(void* ptr) {
    delete static_cast<GroupNormPlan*>(ptr);
}
} // namespace

void prepare_batchnorm2d(core::LayerSlice *ls) {
    auto* bn = checked_layer<core::BatchNorm2d_L>(ls, "BatchNorm2d");
    core::Value_t& input = bn->input(0);
    core::Value_t& output = bn->output();
    require_ready_output(output);
    EXIT_ERROR_CHECK_NE(4, input.data.shape.ndim, "BatchNorm2d expects NCHW input");
    auto* plan = new(std::nothrow) BatchNorm2dPlan();
    EXIT_ERROR_CHECK_EQ(nullptr, plan, "BatchNorm2d plan allocation failed");
    plan->weight = fp32_ptr_or_null(bn->param(core::ParamRole::WEIGHT));
    plan->bias = fp32_ptr_or_null(bn->param(core::ParamRole::BIAS));
    plan->running_mean = fp32_ptr_or_null(bn->param(core::ParamRole::RUNNING_MEAN));
    plan->running_var = fp32_ptr_or_null(bn->param(core::ParamRole::RUNNING_VAR));
    plan->batch = input.data.shape.dims[0];
    plan->channels = input.data.shape.dims[1];
    plan->spatial = input.data.shape.dims[2] * input.data.shape.dims[3];
    plan->batch_stride = plan->channels * plan->spatial;
    plan->eps = bn->eps();
    plan->track_running_stats = bn->trackRunningStats();
    ls->setCache(plan, delete_batchnorm2d_plan);
}

void execute_batchnorm2d(core::LayerSlice *ls, ThreadCtx_t *ctx) {
    (void)ctx;
    auto* plan = ls->cache<BatchNorm2dPlan>();
    EXIT_ERROR_CHECK_EQ(nullptr, plan, "BatchNorm2d plan is nullptr");

    core::Value_t& input = ls->impl<core::BatchNorm2d_L>()->input(0);
    core::Value_t& output = ls->impl<core::BatchNorm2d_L>()->output();
    require_runtime_input(input, "BatchNorm2d input");

    FLOAT* in_ptr = static_cast<FLOAT*>(input.data.ptr);
    FLOAT* out_ptr = static_cast<FLOAT*>(output.data.ptr);

    if (plan->track_running_stats) {
        for (UINT b = 0; b < plan->batch; ++b) {
            const UINT batch_base = b * plan->batch_stride;
            for (UINT c = 0; c < plan->channels; ++c) {
                const UINT base = batch_base + c * plan->spatial;
                math::normalize_affine_scalar_fp32(in_ptr + base,
                                                   out_ptr + base,
                                                   plan->spatial,
                                                   plan->running_mean ? plan->running_mean[c] : 0.0f,
                                                   plan->running_var ? plan->running_var[c] : 1.0f,
                                                   plan->eps,
                                                   plan->weight ? plan->weight[c] : 1.0f,
                                                   plan->bias ? plan->bias[c] : 0.0f);
            }
        }
        return;
    }

    for (UINT b = 0; b < plan->batch; ++b) {
        const UINT batch_base = b * plan->batch_stride;
        for (UINT c = 0; c < plan->channels; ++c) {
            const UINT base = batch_base + c * plan->spatial;
            FLOAT mean = 0.0f;
            FLOAT var = 0.0f;
            math::mean_variance_fp32(in_ptr + base, plan->spatial, mean, var);
            math::normalize_affine_scalar_fp32(in_ptr + base,
                                               out_ptr + base,
                                               plan->spatial,
                                               mean,
                                               var,
                                               plan->eps,
                                               plan->weight ? plan->weight[c] : 1.0f,
                                               plan->bias ? plan->bias[c] : 0.0f);
        }
    }
}

void prepare_layernorm(core::LayerSlice *ls) {
    auto* ln = checked_layer<core::LayerNorm_L>(ls, "LayerNorm");
    core::Value_t& input = ln->input(0);
    core::Value_t& output = ln->output();
    require_ready_output(output);
    (void)fp32_ptr_or_null(ln->param(core::ParamRole::WEIGHT));
    (void)fp32_ptr_or_null(ln->param(core::ParamRole::BIAS));

    const std::vector<UINT>& normalized_shape = ln->normalizedShape();
    UINT norm_size = 1;
    for (auto it = normalized_shape.begin(); it != normalized_shape.end(); ++it) {
        norm_size *= *it;
    }
    EXIT_ERROR_CHECK_EQ(0, norm_size, "LayerNorm norm_size is zero");
    EXIT_ERROR_CHECK_NE(0, input.data.shape.size % norm_size, "LayerNorm input size mismatch");
    auto* plan = new(std::nothrow) LayerNormPlan();
    EXIT_ERROR_CHECK_EQ(nullptr, plan, "LayerNorm plan allocation failed");
    plan->weight = fp32_ptr_or_null(ln->param(core::ParamRole::WEIGHT));
    plan->bias = fp32_ptr_or_null(ln->param(core::ParamRole::BIAS));
    plan->norm_size = norm_size;
    plan->outer = input.data.shape.size / norm_size;
    plan->eps = ln->eps();
    ls->setCache(plan, delete_layernorm_plan);
}

void execute_layernorm(core::LayerSlice *ls, ThreadCtx_t *ctx) {
    (void)ctx;
    auto* plan = ls->cache<LayerNormPlan>();
    EXIT_ERROR_CHECK_EQ(nullptr, plan, "LayerNorm plan is nullptr");

    core::Value_t& input = ls->impl<core::LayerNorm_L>()->input(0);
    core::Value_t& output = ls->impl<core::LayerNorm_L>()->output();
    require_runtime_input(input, "LayerNorm input");

    FLOAT* in_ptr = static_cast<FLOAT*>(input.data.ptr);
    FLOAT* out_ptr = static_cast<FLOAT*>(output.data.ptr);

    for (UINT o = 0; o < plan->outer; ++o) {
        const UINT base = o * plan->norm_size;
        FLOAT mean = 0.0f;
        FLOAT var = 0.0f;
        math::mean_variance_fp32(in_ptr + base, plan->norm_size, mean, var);
        math::normalize_affine_fp32(in_ptr + base,
                                    out_ptr + base,
                                    plan->norm_size,
                                    mean,
                                    var,
                                    plan->eps,
                                    plan->weight,
                                    plan->bias);
    }
}

void prepare_groupnorm(core::LayerSlice *ls) {
    auto* gn = checked_layer<core::GroupNorm_L>(ls, "GroupNorm");
    core::Value_t& input = gn->input(0);
    core::Value_t& output = gn->output();
    require_ready_output(output);
    (void)fp32_ptr_or_null(gn->param(core::ParamRole::WEIGHT));
    (void)fp32_ptr_or_null(gn->param(core::ParamRole::BIAS));
    EXIT_ERROR_CHECK_EQ(false, input.data.shape.ndim >= 3, "GroupNorm expects input shape [N,C,*]");
    EXIT_ERROR_CHECK_EQ(0, gn->numGroups(), "GroupNorm numGroups must be > 0");
    EXIT_ERROR_CHECK_NE(0, input.data.shape.dims[1] % gn->numGroups(),
        "GroupNorm channels must be divisible by numGroups");
    auto* plan = new(std::nothrow) GroupNormPlan();
    EXIT_ERROR_CHECK_EQ(nullptr, plan, "GroupNorm plan allocation failed");
    plan->weight = fp32_ptr_or_null(gn->param(core::ParamRole::WEIGHT));
    plan->bias = fp32_ptr_or_null(gn->param(core::ParamRole::BIAS));
    plan->batch = input.data.shape.dims[0];
    plan->channels = input.data.shape.dims[1];
    plan->groups = gn->numGroups();
    plan->group_size = plan->channels / plan->groups;
    plan->spatial = 1;
    for (UINT i = 2; i < input.data.shape.ndim; ++i) {
        plan->spatial *= input.data.shape.dims[i];
    }
    plan->batch_stride = plan->channels * plan->spatial;
    plan->group_count = plan->group_size * plan->spatial;
    plan->eps = gn->eps();
    ls->setCache(plan, delete_groupnorm_plan);
}

void execute_groupnorm(core::LayerSlice *ls, ThreadCtx_t *ctx) {
    (void)ctx;
    auto* plan = ls->cache<GroupNormPlan>();
    EXIT_ERROR_CHECK_EQ(nullptr, plan, "GroupNorm plan is nullptr");

    core::Value_t& input = ls->impl<core::GroupNorm_L>()->input(0);
    core::Value_t& output = ls->impl<core::GroupNorm_L>()->output();
    require_runtime_input(input, "GroupNorm input");

    FLOAT* in_ptr = static_cast<FLOAT*>(input.data.ptr);
    FLOAT* out_ptr = static_cast<FLOAT*>(output.data.ptr);

    for (UINT b = 0; b < plan->batch; ++b) {
        const UINT batch_base = b * plan->batch_stride;
        for (UINT g = 0; g < plan->groups; ++g) {
            const UINT group_channel_base = g * plan->group_size;
            const UINT group_base = batch_base + group_channel_base * plan->spatial;
            FLOAT mean = 0.0f;
            FLOAT var = 0.0f;
            math::mean_variance_fp32(in_ptr + group_base, plan->group_count, mean, var);
            for (UINT gc = 0; gc < plan->group_size; ++gc) {
                const UINT c = group_channel_base + gc;
                const UINT base = batch_base + c * plan->spatial;
                math::normalize_affine_scalar_fp32(in_ptr + base,
                                                   out_ptr + base,
                                                   plan->spatial,
                                                   mean,
                                                   var,
                                                   plan->eps,
                                                   plan->weight ? plan->weight[c] : 1.0f,
                                                   plan->bias ? plan->bias[c] : 0.0f);
            }
        }
    }
}

} // namespace cpu
} // namespace backend
} // namespace Kernel
