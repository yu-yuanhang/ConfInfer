#include <float.h>
#include <tee_internal_api.h>

#include <backend/confinfer_ta_backend_common.h>

static TEE_Result check_pool_io(ta_layer_exec_ctx_t *ctx)
{
    if (TEE_SUCCESS != ta_backend_require_layer_io(ctx, 1, 1)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    return TEE_SUCCESS;
}

static void resolve_pool_padding(const confinfer_model_image_pool_attr_t *attr, int32_t *pad_top, int32_t *pad_left)
{
    *pad_top = 0;
    *pad_left = 0;
    if (attr->padding_count == 2) {
        *pad_top = attr->padding[0];
        *pad_left = attr->padding[1];
    } else if (attr->padding_count >= 4) {
        *pad_top = attr->padding[0];
        *pad_left = attr->padding[2];
    }
}

TEE_Result ta_execute_maxpool2d_fp32(ta_layer_exec_ctx_t *ctx)
{
    const confinfer_model_image_pool_attr_t *attr = NULL;
    ta_value_t *input = NULL;
    ta_value_t *output = NULL;
    float *src = NULL;
    float *dst = NULL;
    int32_t pad_top = 0;
    int32_t pad_left = 0;
    uint32_t n = 0, c = 0, oh = 0, ow = 0, kh = 0, kw = 0;

    if (TEE_SUCCESS != check_pool_io(ctx)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    attr = (const confinfer_model_image_pool_attr_t *)ta_backend_attr(ctx, sizeof(*attr));
    input = ta_backend_input(ctx, 0);
    output = ta_backend_output(ctx, 0);
    if (!attr || !input || !output || !input->data.ptr || !output->data.ptr) {
        return TEE_ERROR_BAD_STATE;
    }
    if (attr->spatial_dim != 2 || attr->return_indices ||
        input->data.dtype != CONFINFER_DTYPE_FP32 || output->data.dtype != CONFINFER_DTYPE_FP32 ||
        input->data.shape.ndim != 4 || output->data.shape.ndim != 4) {
        return TEE_ERROR_NOT_SUPPORTED;
    }

    resolve_pool_padding(attr, &pad_top, &pad_left);
    src = (float *)input->data.ptr;
    dst = (float *)output->data.ptr;
    for (n = 0; n < output->data.shape.dims[0]; ++n) {
        for (c = 0; c < output->data.shape.dims[1]; ++c) {
            for (oh = 0; oh < output->data.shape.dims[2]; ++oh) {
                for (ow = 0; ow < output->data.shape.dims[3]; ++ow) {
                    float best = -FLT_MAX;
                    for (kh = 0; kh < attr->kernel_size[0]; ++kh) {
                        int32_t ih = (int32_t)(oh * attr->stride[0] + kh * attr->dilation[0]) - pad_top;
                        if (ih < 0 || ih >= (int32_t)input->data.shape.dims[2]) {
                            continue;
                        }
                        for (kw = 0; kw < attr->kernel_size[1]; ++kw) {
                            int32_t iw = (int32_t)(ow * attr->stride[1] + kw * attr->dilation[1]) - pad_left;
                            float value;
                            if (iw < 0 || iw >= (int32_t)input->data.shape.dims[3]) {
                                continue;
                            }
                            value = src[((n * input->data.shape.dims[1] + c) * input->data.shape.dims[2] + (uint32_t)ih) *
                                        input->data.shape.dims[3] + (uint32_t)iw];
                            if (value > best) {
                                best = value;
                            }
                        }
                    }
                    dst[((n * output->data.shape.dims[1] + c) * output->data.shape.dims[2] + oh) *
                        output->data.shape.dims[3] + ow] = best;
                }
            }
        }
    }
    return TEE_SUCCESS;
}

TEE_Result ta_execute_avgpool2d_fp32(ta_layer_exec_ctx_t *ctx)
{
    const confinfer_model_image_pool_attr_t *attr = NULL;
    ta_value_t *input = NULL;
    ta_value_t *output = NULL;
    float *src = NULL;
    float *dst = NULL;
    int32_t pad_top = 0;
    int32_t pad_left = 0;
    uint32_t n = 0, c = 0, oh = 0, ow = 0, kh = 0, kw = 0;

    if (TEE_SUCCESS != check_pool_io(ctx)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    attr = (const confinfer_model_image_pool_attr_t *)ta_backend_attr(ctx, sizeof(*attr));
    input = ta_backend_input(ctx, 0);
    output = ta_backend_output(ctx, 0);
    if (!attr || !input || !output || !input->data.ptr || !output->data.ptr) {
        return TEE_ERROR_BAD_STATE;
    }
    if (attr->spatial_dim != 2 || attr->return_indices ||
        input->data.dtype != CONFINFER_DTYPE_FP32 || output->data.dtype != CONFINFER_DTYPE_FP32 ||
        input->data.shape.ndim != 4 || output->data.shape.ndim != 4) {
        return TEE_ERROR_NOT_SUPPORTED;
    }

    resolve_pool_padding(attr, &pad_top, &pad_left);
    src = (float *)input->data.ptr;
    dst = (float *)output->data.ptr;
    for (n = 0; n < output->data.shape.dims[0]; ++n) {
        for (c = 0; c < output->data.shape.dims[1]; ++c) {
            for (oh = 0; oh < output->data.shape.dims[2]; ++oh) {
                for (ow = 0; ow < output->data.shape.dims[3]; ++ow) {
                    float sum = 0.0f;
                    uint32_t count = 0;
                    for (kh = 0; kh < attr->kernel_size[0]; ++kh) {
                        int32_t ih = (int32_t)(oh * attr->stride[0] + kh * attr->dilation[0]) - pad_top;
                        for (kw = 0; kw < attr->kernel_size[1]; ++kw) {
                            int32_t iw = (int32_t)(ow * attr->stride[1] + kw * attr->dilation[1]) - pad_left;
                            if (ih < 0 || ih >= (int32_t)input->data.shape.dims[2] ||
                                iw < 0 || iw >= (int32_t)input->data.shape.dims[3]) {
                                if (attr->count_include_pad) {
                                    ++count;
                                }
                                continue;
                            }
                            sum += src[((n * input->data.shape.dims[1] + c) * input->data.shape.dims[2] + (uint32_t)ih) *
                                       input->data.shape.dims[3] + (uint32_t)iw];
                            ++count;
                        }
                    }
                    if (attr->divisor_override > 0) {
                        count = attr->divisor_override;
                    }
                    if (count == 0) {
                        return TEE_ERROR_BAD_PARAMETERS;
                    }
                    dst[((n * output->data.shape.dims[1] + c) * output->data.shape.dims[2] + oh) *
                        output->data.shape.dims[3] + ow] = sum / (float)count;
                }
            }
        }
    }
    return TEE_SUCCESS;
}

TEE_Result ta_execute_adaptive_avgpool2d_fp32(ta_layer_exec_ctx_t *ctx)
{
    const confinfer_model_image_adaptive_pool_attr_t *attr = NULL;
    ta_value_t *input = NULL;
    ta_value_t *output = NULL;
    float *src = NULL;
    float *dst = NULL;
    uint32_t n = 0, c = 0, oh = 0, ow = 0, ih = 0, iw = 0;

    if (TEE_SUCCESS != check_pool_io(ctx)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    attr = (const confinfer_model_image_adaptive_pool_attr_t *)ta_backend_attr(ctx, sizeof(*attr));
    input = ta_backend_input(ctx, 0);
    output = ta_backend_output(ctx, 0);
    if (!attr || !input || !output || !input->data.ptr || !output->data.ptr) {
        return TEE_ERROR_BAD_STATE;
    }
    if (attr->return_indices || attr->output_ndim == 0 || attr->output_ndim > 2 ||
        input->data.dtype != CONFINFER_DTYPE_FP32 || output->data.dtype != CONFINFER_DTYPE_FP32 ||
        input->data.shape.ndim != 4 || output->data.shape.ndim != 4) {
        return TEE_ERROR_NOT_SUPPORTED;
    }

    src = (float *)input->data.ptr;
    dst = (float *)output->data.ptr;
    for (n = 0; n < output->data.shape.dims[0]; ++n) {
        for (c = 0; c < output->data.shape.dims[1]; ++c) {
            for (oh = 0; oh < output->data.shape.dims[2]; ++oh) {
                uint32_t h_start = (oh * input->data.shape.dims[2]) / output->data.shape.dims[2];
                uint32_t h_end = ((oh + 1) * input->data.shape.dims[2] + output->data.shape.dims[2] - 1) /
                                 output->data.shape.dims[2];
                for (ow = 0; ow < output->data.shape.dims[3]; ++ow) {
                    uint32_t w_start = (ow * input->data.shape.dims[3]) / output->data.shape.dims[3];
                    uint32_t w_end = ((ow + 1) * input->data.shape.dims[3] + output->data.shape.dims[3] - 1) /
                                     output->data.shape.dims[3];
                    float sum = 0.0f;
                    uint32_t count = 0;
                    for (ih = h_start; ih < h_end; ++ih) {
                        for (iw = w_start; iw < w_end; ++iw) {
                            sum += src[((n * input->data.shape.dims[1] + c) * input->data.shape.dims[2] + ih) *
                                       input->data.shape.dims[3] + iw];
                            ++count;
                        }
                    }
                    if (count == 0) {
                        return TEE_ERROR_BAD_PARAMETERS;
                    }
                    dst[((n * output->data.shape.dims[1] + c) * output->data.shape.dims[2] + oh) *
                        output->data.shape.dims[3] + ow] = sum / (float)count;
                }
            }
        }
    }
    return TEE_SUCCESS;
}

TEE_Result ta_execute_adaptive_maxpool2d_fp32(ta_layer_exec_ctx_t *ctx)
{
    const confinfer_model_image_adaptive_pool_attr_t *attr = NULL;
    ta_value_t *input = NULL;
    ta_value_t *output = NULL;
    float *src = NULL;
    float *dst = NULL;
    uint32_t n = 0, c = 0, oh = 0, ow = 0, ih = 0, iw = 0;

    if (TEE_SUCCESS != check_pool_io(ctx)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    attr = (const confinfer_model_image_adaptive_pool_attr_t *)ta_backend_attr(ctx, sizeof(*attr));
    input = ta_backend_input(ctx, 0);
    output = ta_backend_output(ctx, 0);
    if (!attr || !input || !output || !input->data.ptr || !output->data.ptr) {
        return TEE_ERROR_BAD_STATE;
    }
    if (attr->return_indices || attr->output_ndim == 0 || attr->output_ndim > 2 ||
        input->data.dtype != CONFINFER_DTYPE_FP32 || output->data.dtype != CONFINFER_DTYPE_FP32 ||
        input->data.shape.ndim != 4 || output->data.shape.ndim != 4) {
        return TEE_ERROR_NOT_SUPPORTED;
    }

    src = (float *)input->data.ptr;
    dst = (float *)output->data.ptr;
    for (n = 0; n < output->data.shape.dims[0]; ++n) {
        for (c = 0; c < output->data.shape.dims[1]; ++c) {
            for (oh = 0; oh < output->data.shape.dims[2]; ++oh) {
                uint32_t h_start = (oh * input->data.shape.dims[2]) / output->data.shape.dims[2];
                uint32_t h_end = ((oh + 1) * input->data.shape.dims[2] + output->data.shape.dims[2] - 1) /
                                 output->data.shape.dims[2];
                for (ow = 0; ow < output->data.shape.dims[3]; ++ow) {
                    uint32_t w_start = (ow * input->data.shape.dims[3]) / output->data.shape.dims[3];
                    uint32_t w_end = ((ow + 1) * input->data.shape.dims[3] + output->data.shape.dims[3] - 1) /
                                     output->data.shape.dims[3];
                    float best = -FLT_MAX;
                    for (ih = h_start; ih < h_end; ++ih) {
                        for (iw = w_start; iw < w_end; ++iw) {
                            float value = src[((n * input->data.shape.dims[1] + c) * input->data.shape.dims[2] + ih) *
                                              input->data.shape.dims[3] + iw];
                            if (value > best) {
                                best = value;
                            }
                        }
                    }
                    dst[((n * output->data.shape.dims[1] + c) * output->data.shape.dims[2] + oh) *
                        output->data.shape.dims[3] + ow] = best;
                }
            }
        }
    }
    return TEE_SUCCESS;
}
