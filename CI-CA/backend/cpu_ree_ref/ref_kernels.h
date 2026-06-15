#ifndef __REF_KERNELS_C_H__
#define __REF_KERNELS_C_H__

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef float ref_float32_t;
typedef uint32_t ref_uint32_t;
typedef int32_t ref_int32_t;
typedef int8_t ref_bool_t;

void ref_copy_bytes(const void* src, void* dst, ref_uint32_t bytes);

void ref_relu_fp32(const ref_float32_t* input,
                   ref_float32_t* output,
                   ref_uint32_t size);

void ref_linear_fp32(const ref_float32_t* input,
                     const ref_float32_t* weight,
                     const ref_float32_t* bias,
                     ref_float32_t* output,
                     ref_uint32_t outer,
                     ref_uint32_t out_features,
                     ref_uint32_t in_features,
                     ref_bool_t bias_enabled);

void ref_batchnorm2d_eval_fp32(const ref_float32_t* input,
                               ref_float32_t* output,
                               const ref_float32_t* weight,
                               const ref_float32_t* bias,
                               const ref_float32_t* running_mean,
                               const ref_float32_t* running_var,
                               ref_uint32_t batch,
                               ref_uint32_t channels,
                               ref_uint32_t spatial,
                               ref_float32_t eps);

void ref_adaptiveavgpool2d_fp32(const ref_float32_t* input,
                                ref_float32_t* output,
                                ref_uint32_t batch,
                                ref_uint32_t channels,
                                ref_uint32_t in_h,
                                ref_uint32_t in_w,
                                ref_uint32_t out_h,
                                ref_uint32_t out_w);

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
                          ref_bool_t bias_enabled);

#ifdef __cplusplus
}
#endif

#endif
