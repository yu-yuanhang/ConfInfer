#ifndef __MATH_UTILS_CPU_H_CA__
#define __MATH_UTILS_CPU_H_CA__

#include <core/Param.h>

namespace Kernel {
namespace backend {
namespace cpu {
namespace math {

void fill_zero(core::Data_t &data);
void add_bias_channelwise(FLOAT* out,
                          const FLOAT* bias,
                          UINT channels,
                          UINT spatial_size);
void gemm_nn(const FLOAT* A,
             const FLOAT* B,
             FLOAT* C,
             UINT M,
             UINT N,
             UINT K);
void gemm_nt(const FLOAT* A,
             const FLOAT* B,
             FLOAT* C,
             UINT M,
             UINT N,
             UINT K);
void im2col_nchw(const FLOAT* input,
                 UINT in_channels,
                 UINT in_h,
                 UINT in_w,
                 UINT kernel_h,
                 UINT kernel_w,
                 UINT stride_h,
                 UINT stride_w,
                 INT pad_t,
                 INT pad_l,
                 UINT dilation_h,
                 UINT dilation_w,
                 UINT out_h,
                 UINT out_w,
                 FLOAT* col);
FLOAT mean_fp32(const FLOAT* data, UINT size);
FLOAT variance_fp32(const FLOAT* data, UINT size, FLOAT mean);
void normalize_affine_fp32(const FLOAT* input,
                           FLOAT* output,
                           UINT size,
                           FLOAT mean,
                           FLOAT var,
                           FLOAT eps,
                           const FLOAT* gamma,
                           const FLOAT* beta);
void normalize_affine_scalar_fp32(const FLOAT* input,
                                  FLOAT* output,
                                  UINT size,
                                  FLOAT mean,
                                  FLOAT var,
                                  FLOAT eps,
                                  FLOAT gamma,
                                  FLOAT beta);
void relu_fp32(const FLOAT* input, FLOAT* output, UINT size);
void sigmoid_fp32(const FLOAT* input, FLOAT* output, UINT size);
void add_fp32(const FLOAT* input,
              const FLOAT* other,
              FLOAT* output,
              UINT size,
              FLOAT alpha);
void add_bias_axis_fp32(const FLOAT* input,
                        FLOAT* output,
                        const FLOAT* bias,
                        const core::DataShape_t& shape,
                        UINT axis);
void softmax_axis_fp32(const FLOAT* input,
                       FLOAT* output,
                       const core::DataShape_t& shape,
                       UINT axis);
void concat_axis_fp32(const std::vector<const FLOAT*>& inputs,
                      FLOAT* output,
                      const std::vector<core::DataShape_t>& shapes,
                      UINT axis);
void adaptive_avgpool2d_nchw(const FLOAT* input,
                             FLOAT* output,
                             UINT batch,
                             UINT channels,
                             UINT in_h,
                             UINT in_w,
                             UINT out_h,
                             UINT out_w);
void adaptive_maxpool2d_nchw(const FLOAT* input,
                             FLOAT* output,
                             UINT batch,
                             UINT channels,
                             UINT in_h,
                             UINT in_w,
                             UINT out_h,
                             UINT out_w);

} // namespace math
} // namespace cpu
} // namespace backend
} // namespace Kernel

#endif
