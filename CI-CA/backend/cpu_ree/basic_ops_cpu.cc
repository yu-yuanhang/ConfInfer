#include "cpu_ops.h"
#include "math/math_utils_cpu.h"
#include <activation.h>
#include <arithmetic.h>
#include <linear.h>
#include <reshape.h>
#include <cstring>

namespace Kernel {
namespace backend {
namespace cpu {

namespace {
struct ConcatPlanEntry {
    UINT copy_size;
    UINT local_offset;
};

struct ConcatPlan {
    UINT axis;
    UINT outer;
    UINT inner;
    UINT total_axis;
    std::vector<ConcatPlanEntry> entries;
};

struct SoftmaxPlan {
    UINT axis;
};

struct MatMulPlan {
    UINT M;
    UINT N;
    UINT K;
    UINT batch;
    UINT a_stride;
    UINT b_stride;
    UINT c_stride;
};

struct LinearPlan {
    const FLOAT* weight;
    const FLOAT* bias;
    UINT in_features;
    UINT out_features;
    UINT outer;
    BOOL bias_enabled;
};

template <typename LayerT>
LayerT* checked_layer(core::LayerSlice* ls, const char* name) {
    EXIT_ERROR_CHECK_EQ(nullptr, ls, "LayerSlice is nullptr");
    auto* layer = dynamic_cast<LayerT*>(ls->layer());
    EXIT_ERROR_CHECK_EQ(nullptr, layer, "%s layer type mismatch", name);
    ls->setImpl(layer);
    return layer;
}

UINT normalize_axis(INT axis, UINT ndim, const char* name) {
    if (axis < 0) {
        axis += static_cast<INT>(ndim);
    }
    EXIT_ERROR_CHECK_EQ(false, axis >= 0 && axis < static_cast<INT>(ndim), "%s dim out of range", name);
    return static_cast<UINT>(axis);
}

void delete_concat_plan(void* ptr) {
    delete static_cast<ConcatPlan*>(ptr);
}

void delete_softmax_plan(void* ptr) {
    delete static_cast<SoftmaxPlan*>(ptr);
}

void delete_matmul_plan(void* ptr) {
    delete static_cast<MatMulPlan*>(ptr);
}

void delete_linear_plan(void* ptr) {
    delete static_cast<LinearPlan*>(ptr);
}

ConcatPlan* build_concat_plan(core::Concat_L* concat) {
    core::Value_t& output = concat->output();
    const UINT axis = normalize_axis(concat->dim(), output.data.shape.ndim, "Concat");

    auto* plan = new(std::nothrow) ConcatPlan();
    EXIT_ERROR_CHECK_EQ(nullptr, plan, "Concat plan allocation failed");
    plan->axis = axis;
    plan->outer = 1;
    plan->inner = 1;
    plan->total_axis = output.data.shape.dims[axis];

    for (UINT i = 0; i < axis; ++i) {
        plan->outer *= output.data.shape.dims[i];
    }
    for (UINT i = axis + 1; i < output.data.shape.ndim; ++i) {
        plan->inner *= output.data.shape.dims[i];
    }

    plan->entries.reserve(concat->inputNum());
    UINT local_offset = 0;
    for (UINT i = 0; i < concat->inputNum(); ++i) {
        core::Value_t& value = concat->input(i);
        const UINT axis_size = value.data.shape.dims[axis];
        const UINT copy_size = axis_size * plan->inner;
        plan->entries.push_back(ConcatPlanEntry{copy_size, local_offset});
        local_offset += copy_size;
    }
    EXIT_ERROR_CHECK_NE(plan->total_axis * plan->inner, local_offset,
        "Concat plan total axis mismatch");
    return plan;
}
}

void prepare_relu(core::LayerSlice *ls) {
    auto* relu = checked_layer<core::UnaryOp_L>(ls, "ReLU");
    EXIT_ERROR_CHECK_EQ(nullptr, relu->output().data.ptr, "ReLU output ptr is nullptr");
}

void execute_relu(core::LayerSlice *ls, ThreadCtx_t *ctx) {
    (void)ctx;
    auto* relu = ls->impl<core::UnaryOp_L>();
    core::Value_t& input = relu->input(0);
    core::Value_t& output = relu->output();
    EXIT_ERROR_CHECK_EQ(nullptr, input.data.ptr, "ReLU input ptr is nullptr");
    math::relu_fp32(static_cast<const FLOAT*>(input.data.ptr),
                    static_cast<FLOAT*>(output.data.ptr),
                    input.data.shape.size);
}

void prepare_sigmoid(core::LayerSlice *ls) {
    auto* sigmoid = checked_layer<core::UnaryOp_L>(ls, "Sigmoid");
    EXIT_ERROR_CHECK_EQ(nullptr, sigmoid->output().data.ptr, "Sigmoid output ptr is nullptr");
}

void execute_sigmoid(core::LayerSlice *ls, ThreadCtx_t *ctx) {
    (void)ctx;
    auto* sigmoid = ls->impl<core::UnaryOp_L>();
    core::Value_t& input = sigmoid->input(0);
    core::Value_t& output = sigmoid->output();
    EXIT_ERROR_CHECK_EQ(nullptr, input.data.ptr, "Sigmoid input ptr is nullptr");
    math::sigmoid_fp32(static_cast<const FLOAT*>(input.data.ptr),
                       static_cast<FLOAT*>(output.data.ptr),
                       input.data.shape.size);
}

void prepare_dropout(core::LayerSlice *ls) {
    auto* dropout = checked_layer<core::UnaryOp_L>(ls, "Dropout");
    EXIT_ERROR_CHECK_EQ(nullptr, dropout->output().data.ptr, "Dropout output ptr is nullptr");
}

void execute_dropout(core::LayerSlice *ls, ThreadCtx_t *ctx) {
    (void)ctx;
    auto* dropout = ls->impl<core::UnaryOp_L>();
    core::Value_t& input = dropout->input(0);
    core::Value_t& output = dropout->output();
    EXIT_ERROR_CHECK_EQ(nullptr, input.data.ptr, "Dropout input ptr is nullptr");
    std::memcpy(output.data.ptr, input.data.ptr,
                input.data.shape.size * input.data.getTypeSize());
}

void prepare_flatten(core::LayerSlice *ls) {
    auto* flatten = checked_layer<core::Flatten_L>(ls, "Flatten");
    EXIT_ERROR_CHECK_EQ(nullptr, flatten->output().data.ptr, "Flatten output ptr is nullptr");
}

void execute_flatten(core::LayerSlice *ls, ThreadCtx_t *ctx) {
    (void)ctx;
    auto* flatten = ls->impl<core::Flatten_L>();
    core::Value_t& input = flatten->input(0);
    core::Value_t& output = flatten->output();
    EXIT_ERROR_CHECK_EQ(nullptr, input.data.ptr, "Flatten input ptr is nullptr");
    std::memcpy(output.data.ptr, input.data.ptr,
                input.data.shape.size * input.data.getTypeSize());
}

void prepare_softmax(core::LayerSlice *ls) {
    auto* softmax = checked_layer<core::Softmax_L>(ls, "Softmax");
    core::Value_t& input = softmax->input(0);
    core::Value_t& output = softmax->output();
    EXIT_ERROR_CHECK_EQ(nullptr, output.data.ptr, "Softmax output ptr is nullptr");
    auto* plan = new(std::nothrow) SoftmaxPlan();
    EXIT_ERROR_CHECK_EQ(nullptr, plan, "Softmax plan allocation failed");
    plan->axis = normalize_axis(softmax->dim(), input.data.shape.ndim, "Softmax");
    ls->setCache(plan, delete_softmax_plan);
}

void execute_softmax(core::LayerSlice *ls, ThreadCtx_t *ctx) {
    (void)ctx;
    auto* softmax = ls->impl<core::Softmax_L>();
    auto* plan = ls->cache<SoftmaxPlan>();
    core::Value_t& input = softmax->input(0);
    core::Value_t& output = softmax->output();
    EXIT_ERROR_CHECK_EQ(nullptr, plan, "Softmax plan is nullptr");
    EXIT_ERROR_CHECK_EQ(nullptr, input.data.ptr, "Softmax input ptr is nullptr");

    math::softmax_axis_fp32(static_cast<const FLOAT*>(input.data.ptr),
                            static_cast<FLOAT*>(output.data.ptr),
                            input.data.shape,
                            plan->axis);
}

void prepare_add(core::LayerSlice *ls) {
    auto* add = checked_layer<core::Add_L>(ls, "Add");
    EXIT_ERROR_CHECK_EQ(nullptr, add->output().data.ptr, "Add output ptr is nullptr");
}

void execute_add(core::LayerSlice *ls, ThreadCtx_t *ctx) {
    (void)ctx;
    auto* add = ls->impl<core::Add_L>();
    core::Value_t& lhs = add->input(0);
    core::Value_t& rhs = add->input(1);
    core::Value_t& output = add->output();
    EXIT_ERROR_CHECK_EQ(nullptr, lhs.data.ptr, "Add lhs ptr is nullptr");
    EXIT_ERROR_CHECK_EQ(nullptr, rhs.data.ptr, "Add rhs ptr is nullptr");
    math::add_fp32(static_cast<const FLOAT*>(lhs.data.ptr),
                   static_cast<const FLOAT*>(rhs.data.ptr),
                   static_cast<FLOAT*>(output.data.ptr),
                   lhs.data.shape.size,
                   add->alpha());
}

void prepare_biasadd(core::LayerSlice *ls) {
    auto* biasadd = checked_layer<core::BiasAdd_L>(ls, "BiasAdd");
    core::Value_t& input = biasadd->input(0);
    core::Value_t& output = biasadd->output();
    const core::Data_t* bias = biasadd->param(core::ParamRole::BIAS);
    EXIT_ERROR_CHECK_EQ(nullptr, output.data.ptr, "BiasAdd output ptr is nullptr");
    EXIT_ERROR_CHECK_EQ(nullptr, bias, "BiasAdd bias is nullptr");
    EXIT_ERROR_CHECK_EQ(nullptr, bias->ptr, "BiasAdd bias ptr is nullptr");
    normalize_axis(biasadd->dim(), input.data.shape.ndim, "BiasAdd");
}

void execute_biasadd(core::LayerSlice *ls, ThreadCtx_t *ctx) {
    (void)ctx;
    auto* biasadd = ls->impl<core::BiasAdd_L>();
    core::Value_t& input = biasadd->input(0);
    core::Value_t& output = biasadd->output();
    const core::Data_t* bias = biasadd->param(core::ParamRole::BIAS);
    const UINT axis = normalize_axis(biasadd->dim(), input.data.shape.ndim, "BiasAdd");
    EXIT_ERROR_CHECK_EQ(nullptr, input.data.ptr, "BiasAdd input ptr is nullptr");

    math::add_bias_axis_fp32(static_cast<const FLOAT*>(input.data.ptr),
                             static_cast<FLOAT*>(output.data.ptr),
                             static_cast<const FLOAT*>(bias->ptr),
                             input.data.shape,
                             axis);
}

void prepare_concat(core::LayerSlice *ls) {
    auto* concat = checked_layer<core::Concat_L>(ls, "Concat");
    core::Value_t& output = concat->output();
    EXIT_ERROR_CHECK_EQ(nullptr, output.data.ptr, "Concat output ptr is nullptr");
    const UINT axis = normalize_axis(concat->dim(), output.data.shape.ndim, "Concat");
    for (UINT i = 0; i < concat->inputNum(); ++i) {
        core::Value_t& value = concat->input(i);
        EXIT_ERROR_CHECK_NE(output.data.shape.ndim, value.data.shape.ndim, "Concat input ndim mismatch");
        for (UINT d = 0; d < output.data.shape.ndim; ++d) {
            if (d == axis) {
                continue;
            }
            EXIT_ERROR_CHECK_NE(output.data.shape.dims[d], value.data.shape.dims[d],
                "Concat input shape mismatch");
        }
    }
    ls->setCache(build_concat_plan(concat), delete_concat_plan);
}

void execute_concat(core::LayerSlice *ls, ThreadCtx_t *ctx) {
    (void)ctx;
    auto* concat = ls->impl<core::Concat_L>();
    auto* plan = ls->cache<ConcatPlan>();
    core::Value_t& output = concat->output();
    EXIT_ERROR_CHECK_EQ(nullptr, plan, "Concat plan is nullptr");
    FLOAT* out_ptr = static_cast<FLOAT*>(output.data.ptr);

    for (UINT o = 0; o < plan->outer; ++o) {
        const UINT dst_base = o * plan->total_axis * plan->inner;
        for (UINT i = 0; i < concat->inputNum(); ++i) {
            core::Value_t& value = concat->input(i);
            EXIT_ERROR_CHECK_EQ(nullptr, value.data.ptr, "Concat input ptr is nullptr");
            const FLOAT* in_ptr = static_cast<const FLOAT*>(value.data.ptr);
            const ConcatPlanEntry& entry = plan->entries[i];
            const UINT src_base = o * entry.copy_size;
            std::memcpy(out_ptr + dst_base + entry.local_offset,
                        in_ptr + src_base,
                        entry.copy_size * sizeof(FLOAT));
        }
    }
}

void prepare_matmul(core::LayerSlice *ls) {
    auto* matmul = checked_layer<core::MatMul_L>(ls, "MatMul");
    core::Value_t& lhs = matmul->input(0);
    core::Value_t& rhs = matmul->input(1);
    core::Value_t& output = matmul->output();
    (void)lhs;
    (void)rhs;
    EXIT_ERROR_CHECK_EQ(nullptr, output.data.ptr, "MatMul output ptr is nullptr");
    auto* plan = new(std::nothrow) MatMulPlan();
    EXIT_ERROR_CHECK_EQ(nullptr, plan, "MatMul plan allocation failed");
    const core::DataShape_t& a = lhs.data.shape;
    const core::DataShape_t& b = rhs.data.shape;
    plan->M = a.dims[a.ndim - 2];
    plan->K = a.dims[a.ndim - 1];
    plan->N = b.dims[b.ndim - 1];
    plan->batch = 1;
    for (UINT i = 0; i + 2 < a.ndim; ++i) {
        plan->batch *= a.dims[i];
    }
    plan->a_stride = plan->M * plan->K;
    plan->b_stride = plan->K * plan->N;
    plan->c_stride = plan->M * plan->N;
    ls->setCache(plan, delete_matmul_plan);
}

void execute_matmul(core::LayerSlice *ls, ThreadCtx_t *ctx) {
    (void)ctx;
    auto* matmul = ls->impl<core::MatMul_L>();
    auto* plan = ls->cache<MatMulPlan>();
    core::Value_t& lhs = matmul->input(0);
    core::Value_t& rhs = matmul->input(1);
    core::Value_t& output = matmul->output();
    EXIT_ERROR_CHECK_EQ(nullptr, plan, "MatMul plan is nullptr");
    EXIT_ERROR_CHECK_EQ(nullptr, lhs.data.ptr, "MatMul lhs ptr is nullptr");
    EXIT_ERROR_CHECK_EQ(nullptr, rhs.data.ptr, "MatMul rhs ptr is nullptr");
    const FLOAT* lhs_ptr = static_cast<const FLOAT*>(lhs.data.ptr);
    const FLOAT* rhs_ptr = static_cast<const FLOAT*>(rhs.data.ptr);
    FLOAT* c_ptr = static_cast<FLOAT*>(output.data.ptr);

    for (UINT i = 0; i < plan->batch; ++i) {
        math::gemm_nn(lhs_ptr + i * plan->a_stride,
                      rhs_ptr + i * plan->b_stride,
                      c_ptr + i * plan->c_stride,
                      plan->M, plan->N, plan->K);
    }
}

void prepare_linear(core::LayerSlice *ls) {
    auto* linear = checked_layer<core::Linear_L>(ls, "Linear");
    core::Value_t& input = linear->input(0);
    core::Value_t& output = linear->output();
    const core::Data_t* weight = linear->param(core::ParamRole::WEIGHT);
    EXIT_ERROR_CHECK_EQ(nullptr, output.data.ptr, "Linear output ptr is nullptr");
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
    ls->setCache(plan, delete_linear_plan);
}

void execute_linear(core::LayerSlice *ls, ThreadCtx_t *ctx) {
    (void)ctx;
    auto* linear = ls->impl<core::Linear_L>();
    auto* plan = ls->cache<LinearPlan>();
    core::Value_t& input = linear->input(0);
    core::Value_t& output = linear->output();
    EXIT_ERROR_CHECK_EQ(nullptr, plan, "Linear plan is nullptr");
    EXIT_ERROR_CHECK_EQ(nullptr, input.data.ptr, "Linear input ptr is nullptr");

    math::gemm_nt(static_cast<const FLOAT*>(input.data.ptr),
                  plan->weight,
                  static_cast<FLOAT*>(output.data.ptr),
                  plan->outer,
                  plan->out_features,
                  plan->in_features);
    if (plan->bias_enabled) {
        FLOAT* out_ptr = static_cast<FLOAT*>(output.data.ptr);
        for (UINT row = 0; row < plan->outer; ++row) {
            for (UINT col = 0; col < plan->out_features; ++col) {
                out_ptr[row * plan->out_features + col] += plan->bias[col];
            }
        }
    }
}

} // namespace cpu
} // namespace backend
} // namespace Kernel
