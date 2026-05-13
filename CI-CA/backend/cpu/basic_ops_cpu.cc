#include "cpu_ops.h"
#include "math/math_utils_cpu.h"
#include <activation.h>
#include <arithmetic.h>
#include <linear.h>
#include <cstring>

namespace Kernel {
namespace backend {
namespace cpu {

namespace {
void require_fp32_io(const core::Value_t& input, const core::Value_t& output, const char* name) {
    EXIT_ERROR_CHECK_NE(core::DataType::FP32, input.data.dtype, "%s only supports FP32 input", name);
    EXIT_ERROR_CHECK_NE(core::DataType::FP32, output.data.dtype, "%s only supports FP32 output", name);
    EXIT_ERROR_CHECK_EQ(nullptr, input.data.ptr, "%s input ptr is nullptr", name);
    EXIT_ERROR_CHECK_EQ(nullptr, output.data.ptr, "%s output ptr is nullptr", name);
}
}

void execute_relu(core::LayerSlice *ls, ThreadCtx_t *ctx) {
    (void)ctx;
    auto* relu = dynamic_cast<core::UnaryOp_L*>(ls->layer());
    EXIT_ERROR_CHECK_EQ(nullptr, relu, "Layer is not UnaryOp_L");
    core::Value_t& input = relu->input(0);
    core::Value_t& output = relu->output();
    require_fp32_io(input, output, "ReLU");
    math::relu_fp32(static_cast<const FLOAT*>(input.data.ptr),
                    static_cast<FLOAT*>(output.data.ptr),
                    input.data.shape.size);
}

void execute_sigmoid(core::LayerSlice *ls, ThreadCtx_t *ctx) {
    (void)ctx;
    auto* sigmoid = dynamic_cast<core::UnaryOp_L*>(ls->layer());
    EXIT_ERROR_CHECK_EQ(nullptr, sigmoid, "Layer is not UnaryOp_L");
    core::Value_t& input = sigmoid->input(0);
    core::Value_t& output = sigmoid->output();
    require_fp32_io(input, output, "Sigmoid");
    math::sigmoid_fp32(static_cast<const FLOAT*>(input.data.ptr),
                       static_cast<FLOAT*>(output.data.ptr),
                       input.data.shape.size);
}

void execute_dropout(core::LayerSlice *ls, ThreadCtx_t *ctx) {
    (void)ctx;
    auto* dropout = dynamic_cast<core::UnaryOp_L*>(ls->layer());
    EXIT_ERROR_CHECK_EQ(nullptr, dropout, "Layer is not UnaryOp_L");
    core::Value_t& input = dropout->input(0);
    core::Value_t& output = dropout->output();
    require_fp32_io(input, output, "Dropout");
    std::memcpy(output.data.ptr, input.data.ptr, input.data.shape.size * input.data.getTypeSize());
}

void execute_softmax(core::LayerSlice *ls, ThreadCtx_t *ctx) {
    (void)ctx;
    auto* softmax = dynamic_cast<core::Softmax_L*>(ls->layer());
    EXIT_ERROR_CHECK_EQ(nullptr, softmax, "Layer is not Softmax_L");
    core::Value_t& input = softmax->input(0);
    core::Value_t& output = softmax->output();
    require_fp32_io(input, output, "Softmax");

    INT axis = softmax->dim();
    if (axis < 0) {
        axis += static_cast<INT>(input.data.shape.ndim);
    }
    EXIT_ERROR_CHECK_EQ(false,
        axis >= 0 && axis < static_cast<INT>(input.data.shape.ndim),
        "Softmax dim out of range");

    math::softmax_axis_fp32(static_cast<const FLOAT*>(input.data.ptr),
                            static_cast<FLOAT*>(output.data.ptr),
                            input.data.shape,
                            static_cast<UINT>(axis));
}

void execute_add(core::LayerSlice *ls, ThreadCtx_t *ctx) {
    (void)ctx;
    auto* add = dynamic_cast<core::Add_L*>(ls->layer());
    EXIT_ERROR_CHECK_EQ(nullptr, add, "Layer is not Add_L");
    core::Value_t& input = add->input(0);
    core::Value_t& other = add->input(1);
    core::Value_t& output = add->output();
    require_fp32_io(input, output, "Add");
    EXIT_ERROR_CHECK_NE(core::DataType::FP32, other.data.dtype, "Add only supports FP32 other");
    EXIT_ERROR_CHECK_EQ(nullptr, other.data.ptr, "Add other ptr is nullptr");
    math::add_fp32(static_cast<const FLOAT*>(input.data.ptr),
                   static_cast<const FLOAT*>(other.data.ptr),
                   static_cast<FLOAT*>(output.data.ptr),
                   input.data.shape.size,
                   add->alpha());
}

void execute_biasadd(core::LayerSlice *ls, ThreadCtx_t *ctx) {
    (void)ctx;
    auto* biasadd = dynamic_cast<core::BiasAdd_L*>(ls->layer());
    EXIT_ERROR_CHECK_EQ(nullptr, biasadd, "Layer is not BiasAdd_L");
    core::Value_t& input = biasadd->input(0);
    core::Value_t& output = biasadd->output();
    require_fp32_io(input, output, "BiasAdd");

    const core::Data_t* bias = biasadd->param(core::ParamRole::BIAS);
    EXIT_ERROR_CHECK_EQ(nullptr, bias, "BiasAdd bias is nullptr");
    EXIT_ERROR_CHECK_EQ(nullptr, bias->ptr, "BiasAdd bias ptr is nullptr");
    EXIT_ERROR_CHECK_NE(core::DataType::FP32, bias->dtype, "BiasAdd only supports FP32 bias");

    INT axis = biasadd->dim();
    if (axis < 0) {
        axis += static_cast<INT>(input.data.shape.ndim);
    }
    EXIT_ERROR_CHECK_EQ(false,
        axis >= 0 && axis < static_cast<INT>(input.data.shape.ndim),
        "BiasAdd dim out of range");

    math::add_bias_axis_fp32(static_cast<const FLOAT*>(input.data.ptr),
                             static_cast<FLOAT*>(output.data.ptr),
                             static_cast<const FLOAT*>(bias->ptr),
                             input.data.shape,
                             static_cast<UINT>(axis));
}

void execute_concat(core::LayerSlice *ls, ThreadCtx_t *ctx) {
    (void)ctx;
    auto* concat = dynamic_cast<core::Concat_L*>(ls->layer());
    EXIT_ERROR_CHECK_EQ(nullptr, concat, "Layer is not Concat_L");
    core::Value_t& output = concat->output();
    EXIT_ERROR_CHECK_NE(core::DataType::FP32, output.data.dtype, "Concat only supports FP32 output");
    EXIT_ERROR_CHECK_EQ(nullptr, output.data.ptr, "Concat output ptr is nullptr");

    INT axis = concat->dim();
    if (axis < 0) {
        axis += static_cast<INT>(output.data.shape.ndim);
    }
    EXIT_ERROR_CHECK_EQ(false,
        axis >= 0 && axis < static_cast<INT>(output.data.shape.ndim),
        "Concat dim out of range");

    std::vector<const FLOAT*> inputs;
    std::vector<core::DataShape_t> shapes;
    inputs.reserve(concat->inputNum());
    shapes.reserve(concat->inputNum());
    for (UINT i = 0; i < concat->inputNum(); ++i) {
        core::Value_t& value = concat->input(i);
        EXIT_ERROR_CHECK_NE(core::DataType::FP32, value.data.dtype, "Concat only supports FP32 input");
        EXIT_ERROR_CHECK_EQ(nullptr, value.data.ptr, "Concat input ptr is nullptr");
        inputs.push_back(static_cast<const FLOAT*>(value.data.ptr));
        shapes.push_back(value.data.shape);
    }

    math::concat_axis_fp32(inputs,
                           static_cast<FLOAT*>(output.data.ptr),
                           shapes,
                           static_cast<UINT>(axis));
}

void execute_matmul(core::LayerSlice *ls, ThreadCtx_t *ctx) {
    (void)ctx;
    auto* matmul = dynamic_cast<core::MatMul_L*>(ls->layer());
    EXIT_ERROR_CHECK_EQ(nullptr, matmul, "Layer is not MatMul_L");
    core::Value_t& input = matmul->input(0);
    core::Value_t& other = matmul->input(1);
    core::Value_t& output = matmul->output();
    require_fp32_io(input, output, "MatMul");
    EXIT_ERROR_CHECK_NE(core::DataType::FP32, other.data.dtype, "MatMul only supports FP32 other");
    EXIT_ERROR_CHECK_EQ(nullptr, other.data.ptr, "MatMul other ptr is nullptr");

    const core::DataShape_t& a = input.data.shape;
    const core::DataShape_t& b = other.data.shape;
    const UINT M = a.dims[a.ndim - 2];
    const UINT K = a.dims[a.ndim - 1];
    const UINT N = b.dims[b.ndim - 1];
    UINT batch = 1;
    for (UINT i = 0; i + 2 < a.ndim; ++i) {
        batch *= a.dims[i];
    }

    const UINT a_stride = M * K;
    const UINT b_stride = K * N;
    const UINT c_stride = M * N;
    const FLOAT* a_ptr = static_cast<const FLOAT*>(input.data.ptr);
    const FLOAT* b_ptr = static_cast<const FLOAT*>(other.data.ptr);
    FLOAT* c_ptr = static_cast<FLOAT*>(output.data.ptr);

    for (UINT i = 0; i < batch; ++i) {
        math::gemm_nn(a_ptr + i * a_stride,
                      b_ptr + i * b_stride,
                      c_ptr + i * c_stride,
                      M, N, K);
    }
}

void execute_linear(core::LayerSlice *ls, ThreadCtx_t *ctx) {
    (void)ctx;
    auto* linear = dynamic_cast<core::Linear_L*>(ls->layer());
    EXIT_ERROR_CHECK_EQ(nullptr, linear, "Layer is not Linear_L");
    core::Value_t& input = linear->input(0);
    core::Value_t& output = linear->output();
    require_fp32_io(input, output, "Linear");

    const core::Data_t* weight = linear->param(core::ParamRole::WEIGHT);
    EXIT_ERROR_CHECK_EQ(nullptr, weight, "Linear weight is nullptr");
    EXIT_ERROR_CHECK_EQ(nullptr, weight->ptr, "Linear weight ptr is nullptr");
    EXIT_ERROR_CHECK_NE(core::DataType::FP32, weight->dtype, "Linear only supports FP32 weight");
    const core::Data_t* bias = linear->param(core::ParamRole::BIAS);

    const UINT in_features = linear->inFeatures();
    const UINT out_features = linear->outFeatures();
    const UINT outer = input.data.shape.size / in_features;

    math::gemm_nt(static_cast<const FLOAT*>(input.data.ptr),
                  static_cast<const FLOAT*>(weight->ptr),
                  static_cast<FLOAT*>(output.data.ptr),
                  outer,
                  out_features,
                  in_features);
    if (linear->biasEnabled()) {
        EXIT_ERROR_CHECK_EQ(nullptr, bias, "Linear bias is nullptr");
        EXIT_ERROR_CHECK_EQ(nullptr, bias->ptr, "Linear bias ptr is nullptr");
        EXIT_ERROR_CHECK_NE(core::DataType::FP32, bias->dtype, "Linear only supports FP32 bias");
        FLOAT* out_ptr = static_cast<FLOAT*>(output.data.ptr);
        const FLOAT* bias_ptr = static_cast<const FLOAT*>(bias->ptr);
        for (UINT row = 0; row < outer; ++row) {
            for (UINT col = 0; col < out_features; ++col) {
                out_ptr[row * out_features + col] += bias_ptr[col];
            }
        }
    }
}

} // namespace cpu
} // namespace backend
} // namespace Kernel
