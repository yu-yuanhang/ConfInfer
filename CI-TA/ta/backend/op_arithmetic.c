#include <tee_internal_api.h>

#include <backend/confinfer_ta_backend_common.h>
#include <backend/confinfer_ta_backend_ops.h>

TEE_Result ta_execute_add_fp32(ta_layer_exec_ctx_t *ctx)
{
    const confinfer_model_image_add_attr_t *attr = NULL;
    ta_value_t *lhs = NULL;
    ta_value_t *rhs = NULL;
    ta_value_t *output = NULL;
    float *a = NULL;
    float *b = NULL;
    float *dst = NULL;
    uint32_t i = 0;

    if (TEE_SUCCESS != ta_backend_require_layer_io(ctx, 2, 1)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    attr = (const confinfer_model_image_add_attr_t *)ta_backend_attr(ctx, sizeof(*attr));
    lhs = ta_backend_input(ctx, 0);
    rhs = ta_backend_input(ctx, 1);
    output = ta_backend_output(ctx, 0);
    if (!attr || !lhs || !rhs || !output || !lhs->data.ptr || !rhs->data.ptr || !output->data.ptr) {
        return TEE_ERROR_BAD_STATE;
    }
    if (lhs->data.dtype != CONFINFER_DTYPE_FP32 ||
        rhs->data.dtype != CONFINFER_DTYPE_FP32 ||
        output->data.dtype != CONFINFER_DTYPE_FP32 ||
        lhs->data.shape.elem_count != rhs->data.shape.elem_count ||
        lhs->data.shape.elem_count != output->data.shape.elem_count) {
        return TEE_ERROR_NOT_SUPPORTED;
    }

    a = (float *)lhs->data.ptr;
    b = (float *)rhs->data.ptr;
    dst = (float *)output->data.ptr;
    for (i = 0; i < lhs->data.shape.elem_count; ++i) {
        dst[i] = a[i] + attr->alpha * b[i];
    }
    return TEE_SUCCESS;
}

TEE_Result ta_execute_matmul_fp32(ta_layer_exec_ctx_t *ctx)
{
    ta_value_t *lhs = NULL;
    ta_value_t *rhs = NULL;
    ta_value_t *output = NULL;
    float *a = NULL;
    float *b = NULL;
    float *dst = NULL;
    uint32_t batch = 1;
    uint32_t m = 0;
    uint32_t k = 0;
    uint32_t n = 0;
    uint32_t i = 0;
    uint32_t j = 0;
    uint32_t p = 0;

    if (TEE_SUCCESS != ta_backend_require_layer_io(ctx, 2, 1)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    lhs = ta_backend_input(ctx, 0);
    rhs = ta_backend_input(ctx, 1);
    output = ta_backend_output(ctx, 0);
    if (!lhs || !rhs || !output || !lhs->data.ptr || !rhs->data.ptr || !output->data.ptr) {
        return TEE_ERROR_BAD_STATE;
    }
    if (lhs->data.dtype != CONFINFER_DTYPE_FP32 ||
        rhs->data.dtype != CONFINFER_DTYPE_FP32 ||
        output->data.dtype != CONFINFER_DTYPE_FP32 ||
        lhs->data.shape.ndim < 2 ||
        rhs->data.shape.ndim < 2 ||
        lhs->data.shape.ndim != rhs->data.shape.ndim ||
        output->data.shape.ndim != lhs->data.shape.ndim) {
        return TEE_ERROR_NOT_SUPPORTED;
    }

    for (i = 0; i + 2 < lhs->data.shape.ndim; ++i) {
        if (lhs->data.shape.dims[i] != rhs->data.shape.dims[i] ||
            lhs->data.shape.dims[i] != output->data.shape.dims[i]) {
            return TEE_ERROR_NOT_SUPPORTED;
        }
        batch *= lhs->data.shape.dims[i];
    }

    m = lhs->data.shape.dims[lhs->data.shape.ndim - 2];
    k = lhs->data.shape.dims[lhs->data.shape.ndim - 1];
    if (k != rhs->data.shape.dims[rhs->data.shape.ndim - 2]) {
        return TEE_ERROR_NOT_SUPPORTED;
    }
    n = rhs->data.shape.dims[rhs->data.shape.ndim - 1];
    if (output->data.shape.dims[output->data.shape.ndim - 2] != m ||
        output->data.shape.dims[output->data.shape.ndim - 1] != n) {
        return TEE_ERROR_NOT_SUPPORTED;
    }

    a = (float *)lhs->data.ptr;
    b = (float *)rhs->data.ptr;
    dst = (float *)output->data.ptr;

    for (i = 0; i < batch; ++i) {
        float *a_batch = a + i * m * k;
        float *b_batch = b + i * k * n;
        float *dst_batch = dst + i * m * n;
        for (j = 0; j < m; ++j) {
            for (uint32_t col = 0; col < n; ++col) {
                float sum = 0.0f;
                for (p = 0; p < k; ++p) {
                    sum += a_batch[j * k + p] * b_batch[p * n + col];
                }
                dst_batch[j * n + col] = sum;
            }
        }
    }

    return TEE_SUCCESS;
}

TEE_Result ta_execute_mul_fp32(ta_layer_exec_ctx_t *ctx)
{
    ta_value_t *lhs = NULL;
    ta_value_t *rhs = NULL;
    ta_value_t *output = NULL;
    float *a = NULL;
    float *b = NULL;
    float *dst = NULL;
    uint32_t i = 0;

    if (TEE_SUCCESS != ta_backend_require_layer_io(ctx, 2, 1)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    lhs = ta_backend_input(ctx, 0);
    rhs = ta_backend_input(ctx, 1);
    output = ta_backend_output(ctx, 0);
    if (!lhs || !rhs || !output || !lhs->data.ptr || !rhs->data.ptr || !output->data.ptr) {
        return TEE_ERROR_BAD_STATE;
    }
    if (lhs->data.dtype != CONFINFER_DTYPE_FP32 ||
        rhs->data.dtype != CONFINFER_DTYPE_FP32 ||
        output->data.dtype != CONFINFER_DTYPE_FP32 ||
        lhs->data.shape.elem_count != rhs->data.shape.elem_count ||
        lhs->data.shape.elem_count != output->data.shape.elem_count) {
        return TEE_ERROR_NOT_SUPPORTED;
    }

    a = (float *)lhs->data.ptr;
    b = (float *)rhs->data.ptr;
    dst = (float *)output->data.ptr;
    for (i = 0; i < lhs->data.shape.elem_count; ++i) {
        dst[i] = a[i] * b[i];
    }
    return TEE_SUCCESS;
}

TEE_Result ta_execute_bias_add_fp32(ta_layer_exec_ctx_t *ctx)
{
    const confinfer_model_image_bias_add_attr_t *attr = NULL;
    ta_value_t *input = NULL;
    ta_value_t *output = NULL;
    ta_param_t *bias = NULL;
    float *src = NULL;
    float *dst = NULL;
    float *bias_ptr = NULL;
    int32_t axis = 0;
    uint32_t outer = 1;
    uint32_t axis_dim = 0;
    uint32_t inner = 1;
    uint32_t o = 0;
    uint32_t a = 0;
    uint32_t i = 0;

    if (TEE_SUCCESS != ta_backend_require_layer_io(ctx, 1, 1)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    attr = (const confinfer_model_image_bias_add_attr_t *)ta_backend_attr(ctx, sizeof(*attr));
    input = ta_backend_input(ctx, 0);
    output = ta_backend_output(ctx, 0);
    bias = ta_backend_param_by_role(ctx, CONFINFER_PARAM_ROLE_BIAS);
    if (!attr || !input || !output || !bias ||
        !input->data.ptr || !output->data.ptr || !bias->data.ptr) {
        return TEE_ERROR_BAD_STATE;
    }
    if (input->data.dtype != CONFINFER_DTYPE_FP32 ||
        output->data.dtype != CONFINFER_DTYPE_FP32 ||
        bias->data.dtype != CONFINFER_DTYPE_FP32 ||
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
    axis_dim = input->data.shape.dims[(uint32_t)axis];
    if (axis_dim != attr->size || bias->data.shape.elem_count != axis_dim) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    for (i = 0; i < (uint32_t)axis; ++i) {
        outer *= input->data.shape.dims[i];
    }
    for (i = (uint32_t)axis + 1; i < input->data.shape.ndim; ++i) {
        inner *= input->data.shape.dims[i];
    }

    src = (float *)input->data.ptr;
    dst = (float *)output->data.ptr;
    bias_ptr = (float *)bias->data.ptr;
    for (o = 0; o < outer; ++o) {
        for (a = 0; a < axis_dim; ++a) {
            for (i = 0; i < inner; ++i) {
                const uint32_t index = (o * axis_dim + a) * inner + i;
                dst[index] = src[index] + bias_ptr[a];
            }
        }
    }

    return TEE_SUCCESS;
}

TEE_Result ta_execute_concat_fp32(ta_layer_exec_ctx_t *ctx)
{
    const confinfer_model_image_axis_attr_t *attr = NULL;
    ta_value_t *output = NULL;
    float *dst = NULL;
    int32_t axis = 0;
    uint32_t outer = 1;
    uint32_t inner = 1;
    uint32_t axis_offset = 0;
    uint32_t in_idx = 0;
    uint32_t i = 0;
    uint32_t o = 0;

    if (!ctx || !ctx->partition || !ctx->layer || ctx->layer->input_value_count == 0) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    if (ctx->layer->output_value_count != 1) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    attr = (const confinfer_model_image_axis_attr_t *)ta_backend_attr(ctx, sizeof(*attr));
    output = ta_backend_output(ctx, 0);
    if (!attr || !output || !output->data.ptr || output->data.dtype != CONFINFER_DTYPE_FP32 ||
        output->data.shape.ndim == 0) {
        return TEE_ERROR_BAD_STATE;
    }

    axis = attr->dim;
    if (axis < 0) {
        axis += (int32_t)output->data.shape.ndim;
    }
    if (axis < 0 || axis >= (int32_t)output->data.shape.ndim) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    for (i = 0; i < (uint32_t)axis; ++i) {
        outer *= output->data.shape.dims[i];
    }
    for (i = (uint32_t)axis + 1; i < output->data.shape.ndim; ++i) {
        inner *= output->data.shape.dims[i];
    }

    dst = (float *)output->data.ptr;
    axis_offset = 0;
    for (in_idx = 0; in_idx < ctx->layer->input_value_count; ++in_idx) {
        ta_value_t *input = ta_backend_input(ctx, in_idx);
        float *src = NULL;
        uint32_t axis_dim = 0;
        if (!input || !input->data.ptr || input->data.dtype != CONFINFER_DTYPE_FP32 ||
            input->data.shape.ndim != output->data.shape.ndim) {
            return TEE_ERROR_BAD_PARAMETERS;
        }
        for (i = 0; i < input->data.shape.ndim; ++i) {
            if (i == (uint32_t)axis) {
                continue;
            }
            if (input->data.shape.dims[i] != output->data.shape.dims[i]) {
                return TEE_ERROR_BAD_PARAMETERS;
            }
        }
        axis_dim = input->data.shape.dims[(uint32_t)axis];
        src = (float *)input->data.ptr;
        for (o = 0; o < outer; ++o) {
            TEE_MemMove(dst + (o * output->data.shape.dims[(uint32_t)axis] + axis_offset) * inner,
                        src + o * axis_dim * inner,
                        axis_dim * inner * sizeof(float));
        }
        axis_offset += axis_dim;
    }
    if (axis_offset != output->data.shape.dims[(uint32_t)axis]) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    return TEE_SUCCESS;
}
