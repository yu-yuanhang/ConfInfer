#include <tee_internal_api.h>

#include <backend/confinfer_ta_backend_common.h>

TEE_Result ta_execute_conv2d_fp32(ta_layer_exec_ctx_t *ctx)
{
    const confinfer_conv_attr_t *attr = NULL;
    ta_value_t *input = NULL;
    ta_value_t *output = NULL;
    ta_param_t *weight = NULL;
    ta_param_t *bias = NULL;
    float *src = NULL;
    float *dst = NULL;
    float *w = NULL;
    float *b = NULL;
    uint32_t batch = 0;
    uint32_t in_channels = 0;
    uint32_t in_h = 0;
    uint32_t in_w = 0;
    uint32_t out_channels = 0;
    uint32_t out_h = 0;
    uint32_t out_w = 0;
    uint32_t kernel_h = 0;
    uint32_t kernel_w = 0;
    uint32_t stride_h = 1;
    uint32_t stride_w = 1;
    int32_t pad_top = 0;
    int32_t pad_left = 0;
    uint32_t dil_h = 1;
    uint32_t dil_w = 1;
    uint32_t groups = 1;
    uint32_t out_per_group = 0;
    uint32_t in_per_group = 0;
    uint32_t n = 0;
    uint32_t g = 0;
    uint32_t oc = 0;
    uint32_t oh = 0;
    uint32_t ow = 0;
    uint32_t ic = 0;
    uint32_t kh = 0;
    uint32_t kw = 0;

    if (TEE_SUCCESS != ta_backend_require_layer_io(ctx, 1, 1)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    attr = (const confinfer_conv_attr_t *)ta_backend_attr(ctx, sizeof(*attr));
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
    if (attr->spatial_dim != 2 || input->data.shape.ndim != 4 ||
        output->data.shape.ndim != 4) {
        return TEE_ERROR_NOT_SUPPORTED;
    }

    batch = input->data.shape.dims[0];
    in_channels = input->data.shape.dims[1];
    in_h = input->data.shape.dims[2];
    in_w = input->data.shape.dims[3];
    out_channels = output->data.shape.dims[1];
    out_h = output->data.shape.dims[2];
    out_w = output->data.shape.dims[3];
    kernel_h = attr->kernel_size[0];
    kernel_w = attr->kernel_size[1];
    stride_h = attr->stride[0];
    stride_w = attr->stride[1];
    dil_h = attr->dilation[0];
    dil_w = attr->dilation[1];
    groups = attr->groups ? attr->groups : 1;

    if (attr->padding_count == 2) {
        pad_top = attr->padding[0];
        pad_left = attr->padding[1];
    } else if (attr->padding_count >= 4) {
        pad_top = attr->padding[0];
        pad_left = attr->padding[2];
    }

    if (groups == 0 || in_channels != attr->in_channels ||
        out_channels != attr->out_channels ||
        in_channels % groups != 0 || out_channels % groups != 0 ||
        kernel_h == 0 || kernel_w == 0) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    in_per_group = in_channels / groups;
    out_per_group = out_channels / groups;
    if (weight->data.shape.elem_count != out_channels * in_per_group * kernel_h * kernel_w) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    if (attr->has_bias) {
        if (!bias || !bias->data.ptr || bias->data.dtype != CONFINFER_DTYPE_FP32 ||
            bias->data.shape.elem_count != out_channels) {
            return TEE_ERROR_BAD_PARAMETERS;
        }
        b = (float *)bias->data.ptr;
    }

    src = (float *)input->data.ptr;
    dst = (float *)output->data.ptr;
    w = (float *)weight->data.ptr;

    for (n = 0; n < batch; ++n) {
        for (g = 0; g < groups; ++g) {
            for (oc = 0; oc < out_per_group; ++oc) {
                uint32_t global_oc = g * out_per_group + oc;
                for (oh = 0; oh < out_h; ++oh) {
                    for (ow = 0; ow < out_w; ++ow) {
                        float sum = b ? b[global_oc] : 0.0f;
                        for (ic = 0; ic < in_per_group; ++ic) {
                            uint32_t global_ic = g * in_per_group + ic;
                            for (kh = 0; kh < kernel_h; ++kh) {
                                int32_t ih = (int32_t)(oh * stride_h + kh * dil_h) - pad_top;
                                if (ih < 0 || ih >= (int32_t)in_h) {
                                    continue;
                                }
                                for (kw = 0; kw < kernel_w; ++kw) {
                                    int32_t iw = (int32_t)(ow * stride_w + kw * dil_w) - pad_left;
                                    float input_value;
                                    float weight_value;
                                    if (iw < 0 || iw >= (int32_t)in_w) {
                                        continue;
                                    }
                                    input_value = src[((n * in_channels + global_ic) * in_h + (uint32_t)ih) * in_w + (uint32_t)iw];
                                    weight_value = w[(((global_oc * in_per_group) + ic) * kernel_h + kh) * kernel_w + kw];
                                    sum += input_value * weight_value;
                                }
                            }
                        }
                        dst[((n * out_channels + global_oc) * out_h + oh) * out_w + ow] = sum;
                    }
                }
            }
        }
    }

    return TEE_SUCCESS;
}
