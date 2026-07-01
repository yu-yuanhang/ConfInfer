#include <tee_internal_api.h>

#include <backend/confinfer_ta_backend_common.h>

TEE_Result ta_execute_linear_fp32(ta_layer_exec_ctx_t *ctx)
{
    const confinfer_linear_attr_t *attr = NULL;
    ta_value_t *input = NULL;
    ta_value_t *output = NULL;
    ta_param_t *weight = NULL;
    ta_param_t *bias = NULL;
    float *src = NULL;
    float *dst = NULL;
    float *w = NULL;
    float *b = NULL;
    uint32_t batch = 0;
    uint32_t in_features = 0;
    uint32_t out_features = 0;
    uint32_t row = 0;
    uint32_t col = 0;
    uint32_t k = 0;

    if (TEE_SUCCESS != ta_backend_require_layer_io(ctx, 1, 1)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    attr = (const confinfer_linear_attr_t *)ta_backend_attr(ctx, sizeof(*attr));
    input = ta_backend_input(ctx, 0);
    output = ta_backend_output(ctx, 0);
    weight = ta_backend_param_by_role(ctx, CONFINFER_PARAM_ROLE_WEIGHT);
    bias = ta_backend_param_by_role(ctx, CONFINFER_PARAM_ROLE_BIAS);
    if (!attr || !input || !output || !weight ||
        !input->data.ptr || !output->data.ptr || !weight->data.ptr) {
        return TEE_ERROR_BAD_STATE;
    }
    if (input->data.dtype != CONFINFER_DTYPE_FP32 ||
        output->data.dtype != CONFINFER_DTYPE_FP32 ||
        weight->data.dtype != CONFINFER_DTYPE_FP32) {
        return TEE_ERROR_NOT_SUPPORTED;
    }
    if (input->data.shape.ndim == 0 || output->data.shape.ndim == 0 ||
        input->data.shape.dims[input->data.shape.ndim - 1] != attr->in_features ||
        output->data.shape.dims[output->data.shape.ndim - 1] != attr->out_features) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    in_features = attr->in_features;
    out_features = attr->out_features;
    if (in_features == 0 || out_features == 0 ||
        weight->data.shape.elem_count != out_features * in_features) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    if (attr->has_bias) {
        if (!bias || !bias->data.ptr || bias->data.dtype != CONFINFER_DTYPE_FP32 ||
            bias->data.shape.elem_count != out_features) {
            return TEE_ERROR_BAD_PARAMETERS;
        }
        b = (float *)bias->data.ptr;
    }
    if (input->data.shape.elem_count % in_features != 0 ||
        output->data.shape.elem_count !=
            (input->data.shape.elem_count / in_features) * out_features) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    batch = input->data.shape.elem_count / in_features;
    src = (float *)input->data.ptr;
    dst = (float *)output->data.ptr;
    w = (float *)weight->data.ptr;

    for (row = 0; row < batch; ++row) {
        float *src_row = src + row * in_features;
        float *dst_row = dst + row * out_features;
        for (col = 0; col < out_features; ++col) {
            float sum = b ? b[col] : 0.0f;
            const float *w_row = w + col * in_features;
            for (k = 0; k < in_features; ++k) {
                sum += src_row[k] * w_row[k];
            }
            dst_row[col] = sum;
        }
    }

    return TEE_SUCCESS;
}
