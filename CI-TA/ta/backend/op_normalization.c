#include <tee_internal_api.h>

#include <backend/confinfer_ta_backend_common.h>
#include <backend/confinfer_ta_math.h>

static float param_value_or_default(const ta_param_t *param, uint32_t index, float default_value)
{
    if (!param || !param->data.ptr || param->data.dtype != CONFINFER_DTYPE_FP32 ||
        index >= param->data.shape.elem_count) {
        return default_value;
    }
    return ((float *)param->data.ptr)[index];
}

TEE_Result ta_execute_batchnorm2d_fp32(ta_layer_exec_ctx_t *ctx)
{
    const confinfer_model_image_batchnorm_attr_t *attr = NULL;
    ta_value_t *input = NULL;
    ta_value_t *output = NULL;
    ta_param_t *weight = NULL;
    ta_param_t *bias = NULL;
    ta_param_t *running_mean = NULL;
    ta_param_t *running_var = NULL;
    float *src = NULL;
    float *dst = NULL;
    uint32_t n = 0;
    uint32_t c = 0;
    uint32_t hw = 0;
    uint32_t channel_stride = 0;

    if (TEE_SUCCESS != ta_backend_require_layer_io(ctx, 1, 1)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    attr = (const confinfer_model_image_batchnorm_attr_t *)ta_backend_attr(ctx, sizeof(*attr));
    input = ta_backend_input(ctx, 0);
    output = ta_backend_output(ctx, 0);
    weight = ta_backend_param_by_role(ctx, CONFINFER_PARAM_ROLE_WEIGHT);
    bias = ta_backend_param_by_role(ctx, CONFINFER_PARAM_ROLE_BIAS);
    running_mean = ta_backend_param_by_role(ctx, CONFINFER_PARAM_ROLE_RUNNING_MEAN);
    running_var = ta_backend_param_by_role(ctx, CONFINFER_PARAM_ROLE_RUNNING_VAR);
    if (!attr || !input || !output || !input->data.ptr || !output->data.ptr) {
        return TEE_ERROR_BAD_STATE;
    }
    if (input->data.dtype != CONFINFER_DTYPE_FP32 ||
        output->data.dtype != CONFINFER_DTYPE_FP32 ||
        input->data.shape.ndim != 4 || output->data.shape.ndim != 4 ||
        input->data.shape.elem_count != output->data.shape.elem_count ||
        input->data.shape.dims[1] != attr->num_features) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    src = (float *)input->data.ptr;
    dst = (float *)output->data.ptr;
    channel_stride = input->data.shape.dims[2] * input->data.shape.dims[3];

    for (n = 0; n < input->data.shape.dims[0]; ++n) {
        for (c = 0; c < input->data.shape.dims[1]; ++c) {
            float gamma = attr->affine ? param_value_or_default(weight, c, 1.0f) : 1.0f;
            float beta = attr->affine ? param_value_or_default(bias, c, 0.0f) : 0.0f;
            float mean = param_value_or_default(running_mean, c, 0.0f);
            float var = param_value_or_default(running_var, c, 1.0f);
            float inv_std = 1.0f / ta_math_sqrt_f32(var + attr->eps);
            uint32_t offset = (n * input->data.shape.dims[1] + c) * channel_stride;
            for (hw = 0; hw < channel_stride; ++hw) {
                dst[offset + hw] = ((src[offset + hw] - mean) * inv_std) * gamma + beta;
            }
        }
    }

    return TEE_SUCCESS;
}

TEE_Result ta_execute_layernorm_fp32(ta_layer_exec_ctx_t *ctx)
{
    const confinfer_model_image_norm_attr_t *attr = NULL;
    ta_value_t *input = NULL;
    ta_value_t *output = NULL;
    ta_param_t *weight = NULL;
    ta_param_t *bias = NULL;
    float *src = NULL;
    float *dst = NULL;
    uint32_t block = 0;
    uint32_t block_count = 1;
    uint32_t feature_count = 1;
    uint32_t i = 0;

    if (TEE_SUCCESS != ta_backend_require_layer_io(ctx, 1, 1)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    attr = (const confinfer_model_image_norm_attr_t *)ta_backend_attr(ctx, sizeof(*attr));
    input = ta_backend_input(ctx, 0);
    output = ta_backend_output(ctx, 0);
    weight = ta_backend_param_by_role(ctx, CONFINFER_PARAM_ROLE_WEIGHT);
    bias = ta_backend_param_by_role(ctx, CONFINFER_PARAM_ROLE_BIAS);
    if (!attr || !input || !output || !input->data.ptr || !output->data.ptr) {
        return TEE_ERROR_BAD_STATE;
    }
    if (input->data.dtype != CONFINFER_DTYPE_FP32 ||
        output->data.dtype != CONFINFER_DTYPE_FP32 ||
        input->data.shape.elem_count != output->data.shape.elem_count ||
        attr->normalized_ndim == 0 ||
        attr->normalized_ndim > input->data.shape.ndim) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    for (i = 0; i < input->data.shape.ndim - attr->normalized_ndim; ++i) {
        block_count *= input->data.shape.dims[i];
    }
    for (i = 0; i < attr->normalized_ndim; ++i) {
        if (input->data.shape.dims[input->data.shape.ndim - attr->normalized_ndim + i] !=
            attr->normalized_shape[i]) {
            return TEE_ERROR_BAD_PARAMETERS;
        }
        feature_count *= attr->normalized_shape[i];
    }
    if (feature_count == 0 || block_count * feature_count != input->data.shape.elem_count) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    if (attr->affine) {
        if ((weight && weight->data.shape.elem_count != feature_count) ||
            (bias && bias->data.shape.elem_count != feature_count)) {
            return TEE_ERROR_BAD_PARAMETERS;
        }
    }

    src = (float *)input->data.ptr;
    dst = (float *)output->data.ptr;

    for (block = 0; block < block_count; ++block) {
        float mean = 0.0f;
        float var = 0.0f;
        float *src_block = src + block * feature_count;
        float *dst_block = dst + block * feature_count;
        for (i = 0; i < feature_count; ++i) {
            mean += src_block[i];
        }
        mean /= (float)feature_count;
        for (i = 0; i < feature_count; ++i) {
            float diff = src_block[i] - mean;
            var += diff * diff;
        }
        var /= (float)feature_count;

        for (i = 0; i < feature_count; ++i) {
            float gamma = attr->affine ? param_value_or_default(weight, i, 1.0f) : 1.0f;
            float beta = attr->affine ? param_value_or_default(bias, i, 0.0f) : 0.0f;
            float norm = (src_block[i] - mean) / ta_math_sqrt_f32(var + attr->eps);
            dst_block[i] = norm * gamma + beta;
        }
    }

    return TEE_SUCCESS;
}

TEE_Result ta_execute_groupnorm_fp32(ta_layer_exec_ctx_t *ctx)
{
    const confinfer_model_image_norm_attr_t *attr = NULL;
    ta_value_t *input = NULL;
    ta_value_t *output = NULL;
    ta_param_t *weight = NULL;
    ta_param_t *bias = NULL;
    float *src = NULL;
    float *dst = NULL;
    uint32_t n = 0;
    uint32_t g = 0;
    uint32_t c = 0;
    uint32_t inner = 1;
    uint32_t channels = 0;
    uint32_t groups = 0;
    uint32_t channels_per_group = 0;
    uint32_t sample_stride = 0;
    uint32_t i = 0;

    if (TEE_SUCCESS != ta_backend_require_layer_io(ctx, 1, 1)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    attr = (const confinfer_model_image_norm_attr_t *)ta_backend_attr(ctx, sizeof(*attr));
    input = ta_backend_input(ctx, 0);
    output = ta_backend_output(ctx, 0);
    weight = ta_backend_param_by_role(ctx, CONFINFER_PARAM_ROLE_WEIGHT);
    bias = ta_backend_param_by_role(ctx, CONFINFER_PARAM_ROLE_BIAS);
    if (!attr || !input || !output || !input->data.ptr || !output->data.ptr) {
        return TEE_ERROR_BAD_STATE;
    }
    if (input->data.dtype != CONFINFER_DTYPE_FP32 ||
        output->data.dtype != CONFINFER_DTYPE_FP32 ||
        input->data.shape.ndim < 3 ||
        input->data.shape.elem_count != output->data.shape.elem_count) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    channels = input->data.shape.dims[1];
    groups = attr->num_groups;
    if (groups == 0 || channels != attr->num_channels || channels % groups != 0) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    if (attr->affine) {
        if ((weight && weight->data.shape.elem_count != channels) ||
            (bias && bias->data.shape.elem_count != channels)) {
            return TEE_ERROR_BAD_PARAMETERS;
        }
    }
    channels_per_group = channels / groups;
    for (i = 2; i < input->data.shape.ndim; ++i) {
        inner *= input->data.shape.dims[i];
    }
    sample_stride = channels * inner;
    src = (float *)input->data.ptr;
    dst = (float *)output->data.ptr;

    for (n = 0; n < input->data.shape.dims[0]; ++n) {
        float *src_sample = src + n * sample_stride;
        float *dst_sample = dst + n * sample_stride;
        for (g = 0; g < groups; ++g) {
            float mean = 0.0f;
            float var = 0.0f;
            uint32_t group_elems = channels_per_group * inner;
            uint32_t group_offset = g * group_elems;
            for (i = 0; i < group_elems; ++i) {
                mean += src_sample[group_offset + i];
            }
            mean /= (float)group_elems;
            for (i = 0; i < group_elems; ++i) {
                float diff = src_sample[group_offset + i] - mean;
                var += diff * diff;
            }
            var /= (float)group_elems;

            for (c = 0; c < channels_per_group; ++c) {
                uint32_t global_c = g * channels_per_group + c;
                float gamma = attr->affine ? param_value_or_default(weight, global_c, 1.0f) : 1.0f;
                float beta = attr->affine ? param_value_or_default(bias, global_c, 0.0f) : 0.0f;
                uint32_t base = group_offset + c * inner;
                for (i = 0; i < inner; ++i) {
                    float norm = (src_sample[base + i] - mean) / ta_math_sqrt_f32(var + attr->eps);
                    dst_sample[base + i] = norm * gamma + beta;
                }
            }
        }
    }

    return TEE_SUCCESS;
}
