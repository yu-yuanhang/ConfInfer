#include <tee_internal_api.h>

#include <backend/confinfer_ta_backend_common.h>
#include <backend/confinfer_ta_math.h>
#include <backend/confinfer_ta_backend_ops.h>

TEE_Result ta_execute_relu_fp32(ta_layer_exec_ctx_t *ctx)
{
    ta_value_t *input = NULL;
    ta_value_t *output = NULL;
    float *src = NULL;
    float *dst = NULL;
    uint32_t i = 0;

    if (TEE_SUCCESS != ta_backend_require_layer_io(ctx, 1, 1)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    input = ta_backend_input(ctx, 0);
    output = ta_backend_output(ctx, 0);
    if (!input || !output || !input->data.ptr || !output->data.ptr) {
        return TEE_ERROR_BAD_STATE;
    }
    if (input->data.dtype != CONFINFER_DTYPE_FP32 ||
        output->data.dtype != CONFINFER_DTYPE_FP32 ||
        input->data.shape.elem_count != output->data.shape.elem_count) {
        return TEE_ERROR_NOT_SUPPORTED;
    }

    src = (float *)input->data.ptr;
    dst = (float *)output->data.ptr;
    for (i = 0; i < input->data.shape.elem_count; ++i) {
        dst[i] = src[i] > 0.0f ? src[i] : 0.0f;
    }
    return TEE_SUCCESS;
}

TEE_Result ta_execute_sigmoid_fp32(ta_layer_exec_ctx_t *ctx)
{
    ta_value_t *input = NULL;
    ta_value_t *output = NULL;
    float *src = NULL;
    float *dst = NULL;
    uint32_t i = 0;

    if (TEE_SUCCESS != ta_backend_require_layer_io(ctx, 1, 1)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    input = ta_backend_input(ctx, 0);
    output = ta_backend_output(ctx, 0);
    if (!input || !output || !input->data.ptr || !output->data.ptr) {
        return TEE_ERROR_BAD_STATE;
    }
    if (input->data.dtype != CONFINFER_DTYPE_FP32 ||
        output->data.dtype != CONFINFER_DTYPE_FP32 ||
        input->data.shape.elem_count != output->data.shape.elem_count) {
        return TEE_ERROR_NOT_SUPPORTED;
    }

    src = (float *)input->data.ptr;
    dst = (float *)output->data.ptr;
    for (i = 0; i < input->data.shape.elem_count; ++i) {
        dst[i] = 1.0f / (1.0f + ta_math_exp_f32(-src[i]));
    }
    return TEE_SUCCESS;
}

TEE_Result ta_execute_dropout_fp32(ta_layer_exec_ctx_t *ctx)
{
    ta_value_t *input = NULL;
    ta_value_t *output = NULL;

    if (TEE_SUCCESS != ta_backend_require_layer_io(ctx, 1, 1)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    input = ta_backend_input(ctx, 0);
    output = ta_backend_output(ctx, 0);
    if (!input || !output || !input->data.ptr || !output->data.ptr) {
        return TEE_ERROR_BAD_STATE;
    }
    if (input->data.byte_size != output->data.byte_size) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    // 当前 TA 路径默认按推理语义处理 Dropout，即直接透传。
    TEE_MemMove(output->data.ptr, input->data.ptr, input->data.byte_size);
    return TEE_SUCCESS;
}

TEE_Result ta_execute_softmax_fp32(ta_layer_exec_ctx_t *ctx)
{
    const confinfer_model_image_axis_attr_t *attr = NULL;
    ta_value_t *input = NULL;
    ta_value_t *output = NULL;
    float *src = NULL;
    float *dst = NULL;
    int32_t axis = 0;
    uint32_t outer = 1;
    uint32_t axis_dim = 0;
    uint32_t inner = 1;
    uint32_t o = 0;
    uint32_t i = 0;
    uint32_t a = 0;

    if (TEE_SUCCESS != ta_backend_require_layer_io(ctx, 1, 1)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    attr = (const confinfer_model_image_axis_attr_t *)ta_backend_attr(ctx, sizeof(*attr));
    input = ta_backend_input(ctx, 0);
    output = ta_backend_output(ctx, 0);
    if (!attr || !input || !output || !input->data.ptr || !output->data.ptr) {
        return TEE_ERROR_BAD_STATE;
    }
    if (input->data.dtype != CONFINFER_DTYPE_FP32 ||
        output->data.dtype != CONFINFER_DTYPE_FP32 ||
        input->data.shape.ndim == 0 ||
        input->data.shape.elem_count != output->data.shape.elem_count) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    axis = attr->dim;
    if (axis < 0) {
        axis += (int32_t)input->data.shape.ndim;
    }
    if (axis < 0 || axis >= (int32_t)input->data.shape.ndim) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    for (i = 0; i < (uint32_t)axis; ++i) {
        outer *= input->data.shape.dims[i];
    }
    axis_dim = input->data.shape.dims[(uint32_t)axis];
    for (i = (uint32_t)axis + 1; i < input->data.shape.ndim; ++i) {
        inner *= input->data.shape.dims[i];
    }

    src = (float *)input->data.ptr;
    dst = (float *)output->data.ptr;
    for (o = 0; o < outer; ++o) {
        for (i = 0; i < inner; ++i) {
            float max_val = ta_math_neg_inf_f32();
            float sum = 0.0f;
            for (a = 0; a < axis_dim; ++a) {
                float value = src[(o * axis_dim + a) * inner + i];
                if (value > max_val) {
                    max_val = value;
                }
            }
            for (a = 0; a < axis_dim; ++a) {
                float ex = ta_math_exp_f32(src[(o * axis_dim + a) * inner + i] - max_val);
                dst[(o * axis_dim + a) * inner + i] = ex;
                sum += ex;
            }
            if (sum == 0.0f) {
                return TEE_ERROR_BAD_STATE;
            }
            for (a = 0; a < axis_dim; ++a) {
                dst[(o * axis_dim + a) * inner + i] /= sum;
            }
        }
    }

    return TEE_SUCCESS;
}
