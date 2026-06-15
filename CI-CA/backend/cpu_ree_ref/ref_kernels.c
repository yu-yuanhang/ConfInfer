#include "ref_kernels.h"

#include <math.h>
#include <string.h>

void ref_copy_bytes(const void* src, void* dst, ref_uint32_t bytes) {
    if (bytes == 0) {
        return;
    }
    memcpy(dst, src, (size_t)bytes);
}

void ref_relu_fp32(const ref_float32_t* input,
                   ref_float32_t* output,
                   ref_uint32_t size) {
    ref_uint32_t i;
    for (i = 0; i < size; ++i) {
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }
}

void ref_linear_fp32(const ref_float32_t* input,
                     const ref_float32_t* weight,
                     const ref_float32_t* bias,
                     ref_float32_t* output,
                     ref_uint32_t outer,
                     ref_uint32_t out_features,
                     ref_uint32_t in_features,
                     ref_bool_t bias_enabled) {
    ref_uint32_t row;
    for (row = 0; row < outer; ++row) {
        ref_uint32_t col;
        for (col = 0; col < out_features; ++col) {
            ref_float32_t acc = 0.0f;
            ref_uint32_t k;
            for (k = 0; k < in_features; ++k) {
                acc += input[row * in_features + k] * weight[col * in_features + k];
            }
            if (bias_enabled) {
                acc += bias[col];
            }
            output[row * out_features + col] = acc;
        }
    }
}

void ref_batchnorm2d_eval_fp32(const ref_float32_t* input,
                               ref_float32_t* output,
                               const ref_float32_t* weight,
                               const ref_float32_t* bias,
                               const ref_float32_t* running_mean,
                               const ref_float32_t* running_var,
                               ref_uint32_t batch,
                               ref_uint32_t channels,
                               ref_uint32_t spatial,
                               ref_float32_t eps) {
    ref_uint32_t b;
    for (b = 0; b < batch; ++b) {
        ref_uint32_t c;
        for (c = 0; c < channels; ++c) {
            const ref_float32_t gamma = weight ? weight[c] : 1.0f;
            const ref_float32_t beta = bias ? bias[c] : 0.0f;
            const ref_float32_t mean = running_mean ? running_mean[c] : 0.0f;
            const ref_float32_t var = running_var ? running_var[c] : 1.0f;
            const ref_float32_t denom = sqrtf(var + eps);
            const ref_uint32_t base = b * channels * spatial + c * spatial;
            ref_uint32_t i;
            for (i = 0; i < spatial; ++i) {
                output[base + i] = ((input[base + i] - mean) / denom) * gamma + beta;
            }
        }
    }
}

void ref_adaptiveavgpool2d_fp32(const ref_float32_t* input,
                                ref_float32_t* output,
                                ref_uint32_t batch,
                                ref_uint32_t channels,
                                ref_uint32_t in_h,
                                ref_uint32_t in_w,
                                ref_uint32_t out_h,
                                ref_uint32_t out_w) {
    ref_uint32_t b;
    for (b = 0; b < batch; ++b) {
        ref_uint32_t c;
        for (c = 0; c < channels; ++c) {
            const ref_uint32_t in_base = (b * channels + c) * in_h * in_w;
            const ref_uint32_t out_base = (b * channels + c) * out_h * out_w;
            ref_uint32_t oh;
            for (oh = 0; oh < out_h; ++oh) {
                const ref_uint32_t h_start = (oh * in_h) / out_h;
                const ref_uint32_t h_end = ((oh + 1u) * in_h + out_h - 1u) / out_h;
                ref_uint32_t ow;
                for (ow = 0; ow < out_w; ++ow) {
                    const ref_uint32_t w_start = (ow * in_w) / out_w;
                    const ref_uint32_t w_end = ((ow + 1u) * in_w + out_w - 1u) / out_w;
                    ref_float32_t sum = 0.0f;
                    ref_uint32_t count = 0;
                    ref_uint32_t ih;
                    for (ih = h_start; ih < h_end; ++ih) {
                        ref_uint32_t iw;
                        for (iw = w_start; iw < w_end; ++iw) {
                            sum += input[in_base + ih * in_w + iw];
                            ++count;
                        }
                    }
                    output[out_base + oh * out_w + ow] =
                        (count > 0u) ? (sum / (ref_float32_t)count) : 0.0f;
                }
            }
        }
    }
}

void ref_conv2d_nchw_fp32(const ref_float32_t* input,
                          const ref_float32_t* weight,
                          const ref_float32_t* bias,
                          ref_float32_t* output,
                          ref_uint32_t batch,
                          ref_uint32_t in_c,
                          ref_uint32_t in_h,
                          ref_uint32_t in_w,
                          ref_uint32_t out_c,
                          ref_uint32_t out_h,
                          ref_uint32_t out_w,
                          ref_uint32_t groups,
                          ref_uint32_t in_c_pg,
                          ref_uint32_t out_c_pg,
                          ref_uint32_t k_h,
                          ref_uint32_t k_w,
                          ref_uint32_t s_h,
                          ref_uint32_t s_w,
                          ref_uint32_t d_h,
                          ref_uint32_t d_w,
                          ref_int32_t pad_t,
                          ref_int32_t pad_l,
                          ref_bool_t bias_enabled) {
    const ref_uint32_t input_batch_stride = in_c * in_h * in_w;
    const ref_uint32_t output_batch_stride = out_c * out_h * out_w;
    const ref_uint32_t kernel_size = in_c_pg * k_h * k_w;
    ref_uint32_t n;

    for (n = 0; n < batch; ++n) {
        ref_uint32_t g;
        for (g = 0; g < groups; ++g) {
            ref_uint32_t ocg;
            for (ocg = 0; ocg < out_c_pg; ++ocg) {
                const ref_uint32_t oc = g * out_c_pg + ocg;
                ref_uint32_t oh;
                for (oh = 0; oh < out_h; ++oh) {
                    ref_uint32_t ow;
                    for (ow = 0; ow < out_w; ++ow) {
                        ref_float32_t acc = bias_enabled ? bias[oc] : 0.0f;
                        ref_uint32_t icg;
                        for (icg = 0; icg < in_c_pg; ++icg) {
                            const ref_uint32_t ic = g * in_c_pg + icg;
                            ref_uint32_t kh;
                            for (kh = 0; kh < k_h; ++kh) {
                                const ref_int32_t ih =
                                    (ref_int32_t)(oh * s_h) - pad_t + (ref_int32_t)(kh * d_h);
                                if (ih < 0 || ih >= (ref_int32_t)in_h) {
                                    continue;
                                }
                                ref_uint32_t kw;
                                for (kw = 0; kw < k_w; ++kw) {
                                    const ref_int32_t iw =
                                        (ref_int32_t)(ow * s_w) - pad_l + (ref_int32_t)(kw * d_w);
                                    if (iw < 0 || iw >= (ref_int32_t)in_w) {
                                        continue;
                                    }
                                    const ref_uint32_t input_index =
                                        n * input_batch_stride + ic * in_h * in_w
                                        + (ref_uint32_t)ih * in_w + (ref_uint32_t)iw;
                                    const ref_uint32_t weight_index =
                                        oc * kernel_size + icg * k_h * k_w + kh * k_w + kw;
                                    acc += input[input_index] * weight[weight_index];
                                }
                            }
                        }
                        output[n * output_batch_stride + oc * out_h * out_w + oh * out_w + ow] = acc;
                    }
                }
            }
        }
    }
}
