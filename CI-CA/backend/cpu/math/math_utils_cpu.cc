#include "math_utils_cpu.h"

#include <cmath>
#include <cstring>
#include <vector>

namespace Kernel {
namespace backend {
namespace cpu {
namespace math {

void fill_zero(core::Data_t &data) {
    if (!data.ptr || data.shape.size == 0) return;
    std::memset(data.ptr, 0, data.shape.size * data.getTypeSize());
}

void add_bias_channelwise(FLOAT* out,
                          const FLOAT* bias,
                          UINT channels,
                          UINT spatial_size) {
    if (nullptr == out || nullptr == bias) {
        return;
    }
    for (UINT c = 0; c < channels; ++c) {
        FLOAT b = bias[c];
        UINT base = c * spatial_size;
        for (UINT s = 0; s < spatial_size; ++s) {
            out[base + s] += b;
        }
    }
}

void gemm_nn(const FLOAT* A,
             const FLOAT* B,
             FLOAT* C,
             UINT M,
             UINT N,
             UINT K) {
    std::memset(C, 0, M * N * sizeof(FLOAT));
    for (UINT m = 0; m < M; ++m) {
        FLOAT* c_row = C + m * N;
        const FLOAT* a_row = A + m * K;
        for (UINT k = 0; k < K; ++k) {
            const FLOAT a = a_row[k];
            const FLOAT* b_row = B + k * N;
            for (UINT n = 0; n < N; ++n) {
                c_row[n] += a * b_row[n];
            }
        }
    }
}

void gemm_nt(const FLOAT* A,
             const FLOAT* B,
             FLOAT* C,
             UINT M,
             UINT N,
             UINT K) {
    std::memset(C, 0, M * N * sizeof(FLOAT));
    for (UINT m = 0; m < M; ++m) {
        FLOAT* c_row = C + m * N;
        const FLOAT* a_row = A + m * K;
        for (UINT n = 0; n < N; ++n) {
            const FLOAT* b_row = B + n * K;
            FLOAT acc = 0.0f;
            for (UINT k = 0; k < K; ++k) {
                acc += a_row[k] * b_row[k];
            }
            c_row[n] = acc;
        }
    }
}

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
                 FLOAT* col) {
    const UINT channel_size = in_h * in_w;
    const UINT out_spatial = out_h * out_w;

    for (UINT c = 0; c < in_channels; ++c) {
        const FLOAT* input_c = input + c * channel_size;
        const UINT c_row_base = c * kernel_h * kernel_w;
        for (UINT kh = 0; kh < kernel_h; ++kh) {
            const INT kh_offset = static_cast<INT>(kh * dilation_h);
            const UINT kh_row_base = (c_row_base + kh * kernel_w) * out_spatial;
            for (UINT kw = 0; kw < kernel_w; ++kw) {
                const INT kw_offset = static_cast<INT>(kw * dilation_w);
                FLOAT* col_row = col + kh_row_base + kw * out_spatial;
                for (UINT oh = 0; oh < out_h; ++oh) {
                    const INT ih = static_cast<INT>(oh * stride_h) - pad_t + kh_offset;
                    FLOAT* col_row_oh = col_row + oh * out_w;

                    if (ih < 0 || ih >= static_cast<INT>(in_h)) {
                        std::memset(col_row_oh, 0, out_w * sizeof(FLOAT));
                        continue;
                    }

                    const FLOAT* input_row = input_c + static_cast<UINT>(ih) * in_w;
                    for (UINT ow = 0; ow < out_w; ++ow) {
                        const INT iw = static_cast<INT>(ow * stride_w) - pad_l + kw_offset;
                        col_row_oh[ow] = (iw < 0 || iw >= static_cast<INT>(in_w))
                            ? 0.0f
                            : input_row[static_cast<UINT>(iw)];
                    }
                }
            }
        }
    }
}

FLOAT mean_fp32(const FLOAT* data, UINT size) {
    FLOAT sum = 0.0f;
    for (UINT i = 0; i < size; ++i) {
        sum += data[i];
    }
    return sum / static_cast<FLOAT>(size);
}

FLOAT variance_fp32(const FLOAT* data, UINT size, FLOAT mean) {
    FLOAT var = 0.0f;
    for (UINT i = 0; i < size; ++i) {
        FLOAT diff = data[i] - mean;
        var += diff * diff;
    }
    return var / static_cast<FLOAT>(size);
}

void normalize_affine_fp32(const FLOAT* input,
                           FLOAT* output,
                           UINT size,
                           FLOAT mean,
                           FLOAT var,
                           FLOAT eps,
                           const FLOAT* gamma,
                           const FLOAT* beta) {
    FLOAT denom = std::sqrt(var + eps);
    for (UINT i = 0; i < size; ++i) {
        FLOAT g = gamma ? gamma[i] : 1.0f;
        FLOAT b = beta ? beta[i] : 0.0f;
        output[i] = ((input[i] - mean) / denom) * g + b;
    }
}

void normalize_affine_scalar_fp32(const FLOAT* input,
                                  FLOAT* output,
                                  UINT size,
                                  FLOAT mean,
                                  FLOAT var,
                                  FLOAT eps,
                                  FLOAT gamma,
                                  FLOAT beta) {
    FLOAT denom = std::sqrt(var + eps);
    for (UINT i = 0; i < size; ++i) {
        output[i] = ((input[i] - mean) / denom) * gamma + beta;
    }
}

void relu_fp32(const FLOAT* input, FLOAT* output, UINT size) {
    for (UINT i = 0; i < size; ++i) {
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }
}

void sigmoid_fp32(const FLOAT* input, FLOAT* output, UINT size) {
    for (UINT i = 0; i < size; ++i) {
        output[i] = 1.0f / (1.0f + std::exp(-input[i]));
    }
}

void add_fp32(const FLOAT* input,
              const FLOAT* other,
              FLOAT* output,
              UINT size,
              FLOAT alpha) {
    for (UINT i = 0; i < size; ++i) {
        output[i] = input[i] + alpha * other[i];
    }
}

void add_bias_axis_fp32(const FLOAT* input,
                        FLOAT* output,
                        const FLOAT* bias,
                        const core::DataShape_t& shape,
                        UINT axis) {
    UINT outer = 1;
    for (UINT i = 0; i < axis; ++i) {
        outer *= shape.dims[i];
    }

    const UINT axis_size = shape.dims[axis];

    UINT inner = 1;
    for (UINT i = axis + 1; i < shape.ndim; ++i) {
        inner *= shape.dims[i];
    }

    for (UINT o = 0; o < outer; ++o) {
        const UINT outer_base = o * axis_size * inner;
        for (UINT a = 0; a < axis_size; ++a) {
            const FLOAT b = bias[a];
            const UINT base = outer_base + a * inner;
            for (UINT i = 0; i < inner; ++i) {
                output[base + i] = input[base + i] + b;
            }
        }
    }
}

void softmax_axis_fp32(const FLOAT* input,
                       FLOAT* output,
                       const core::DataShape_t& shape,
                       UINT axis) {
    UINT outer = 1;
    for (UINT i = 0; i < axis; ++i) {
        outer *= shape.dims[i];
    }
    const UINT axis_size = shape.dims[axis];
    UINT inner = 1;
    for (UINT i = axis + 1; i < shape.ndim; ++i) {
        inner *= shape.dims[i];
    }

    for (UINT o = 0; o < outer; ++o) {
        const UINT outer_base = o * axis_size * inner;
        for (UINT i = 0; i < inner; ++i) {
            FLOAT max_v = input[outer_base + i];
            for (UINT a = 1; a < axis_size; ++a) {
                const FLOAT v = input[outer_base + a * inner + i];
                if (v > max_v) {
                    max_v = v;
                }
            }
            FLOAT sum = 0.0f;
            for (UINT a = 0; a < axis_size; ++a) {
                const FLOAT e = std::exp(input[outer_base + a * inner + i] - max_v);
                output[outer_base + a * inner + i] = e;
                sum += e;
            }
            for (UINT a = 0; a < axis_size; ++a) {
                output[outer_base + a * inner + i] /= sum;
            }
        }
    }
}

void concat_axis_fp32(const std::vector<const FLOAT*>& inputs,
                      FLOAT* output,
                      const std::vector<core::DataShape_t>& shapes,
                      UINT axis) {
    if (inputs.empty()) {
        return;
    }
    UINT outer = 1;
    for (UINT i = 0; i < axis; ++i) {
        outer *= shapes[0].dims[i];
    }
    UINT inner = 1;
    for (UINT i = axis + 1; i < shapes[0].ndim; ++i) {
        inner *= shapes[0].dims[i];
    }
    UINT total_axis = 0;
    for (UINT t = 0; t < shapes.size(); ++t) {
        total_axis += shapes[t].dims[axis];
    }

    for (UINT o = 0; o < outer; ++o) {
        const UINT dst_base = o * total_axis * inner;
        UINT local_offset = 0;
        for (UINT t = 0; t < inputs.size(); ++t) {
            const UINT axis_size = shapes[t].dims[axis];
            const UINT copy_size = axis_size * inner;
            const UINT in_offset = o * copy_size;
            std::memcpy(output + dst_base + local_offset,
                        inputs[t] + in_offset,
                        copy_size * sizeof(FLOAT));
            local_offset += copy_size;
        }
    }
}

void adaptive_avgpool2d_nchw(const FLOAT* input,
                             FLOAT* output,
                             UINT batch,
                             UINT channels,
                             UINT in_h,
                             UINT in_w,
                             UINT out_h,
                             UINT out_w) {
    const UINT in_channel_stride = in_h * in_w;
    const UINT out_channel_stride = out_h * out_w;
    const UINT in_batch_stride = channels * in_channel_stride;
    const UINT out_batch_stride = channels * out_channel_stride;

    for (UINT n = 0; n < batch; ++n) {
        const FLOAT* in_n = input + n * in_batch_stride;
        FLOAT* out_n = output + n * out_batch_stride;
        for (UINT c = 0; c < channels; ++c) {
            const FLOAT* in_c = in_n + c * in_channel_stride;
            FLOAT* out_c = out_n + c * out_channel_stride;
            for (UINT oh = 0; oh < out_h; ++oh) {
                const UINT h_start = static_cast<UINT>(std::floor(static_cast<double>(oh * in_h) / out_h));
                const UINT h_end = static_cast<UINT>(std::ceil(static_cast<double>((oh + 1) * in_h) / out_h));
                for (UINT ow = 0; ow < out_w; ++ow) {
                    const UINT w_start = static_cast<UINT>(std::floor(static_cast<double>(ow * in_w) / out_w));
                    const UINT w_end = static_cast<UINT>(std::ceil(static_cast<double>((ow + 1) * in_w) / out_w));
                    FLOAT sum = 0.0f;
                    UINT count = 0;
                    for (UINT ih = h_start; ih < h_end; ++ih) {
                        const FLOAT* row = in_c + ih * in_w;
                        for (UINT iw = w_start; iw < w_end; ++iw) {
                            sum += row[iw];
                            ++count;
                        }
                    }
                    out_c[oh * out_w + ow] = sum / static_cast<FLOAT>(count);
                }
            }
        }
    }
}

void adaptive_maxpool2d_nchw(const FLOAT* input,
                             FLOAT* output,
                             UINT batch,
                             UINT channels,
                             UINT in_h,
                             UINT in_w,
                             UINT out_h,
                             UINT out_w) {
    const UINT in_channel_stride = in_h * in_w;
    const UINT out_channel_stride = out_h * out_w;
    const UINT in_batch_stride = channels * in_channel_stride;
    const UINT out_batch_stride = channels * out_channel_stride;

    for (UINT n = 0; n < batch; ++n) {
        const FLOAT* in_n = input + n * in_batch_stride;
        FLOAT* out_n = output + n * out_batch_stride;
        for (UINT c = 0; c < channels; ++c) {
            const FLOAT* in_c = in_n + c * in_channel_stride;
            FLOAT* out_c = out_n + c * out_channel_stride;
            for (UINT oh = 0; oh < out_h; ++oh) {
                const UINT h_start = static_cast<UINT>(std::floor(static_cast<double>(oh * in_h) / out_h));
                const UINT h_end = static_cast<UINT>(std::ceil(static_cast<double>((oh + 1) * in_h) / out_h));
                for (UINT ow = 0; ow < out_w; ++ow) {
                    const UINT w_start = static_cast<UINT>(std::floor(static_cast<double>(ow * in_w) / out_w));
                    const UINT w_end = static_cast<UINT>(std::ceil(static_cast<double>((ow + 1) * in_w) / out_w));
                    FLOAT max_v = input[0];
                    bool init = false;
                    for (UINT ih = h_start; ih < h_end; ++ih) {
                        const FLOAT* row = in_c + ih * in_w;
                        for (UINT iw = w_start; iw < w_end; ++iw) {
                            const FLOAT v = row[iw];
                            if (!init || v > max_v) {
                                max_v = v;
                                init = true;
                            }
                        }
                    }
                    out_c[oh * out_w + ow] = max_v;
                }
            }
        }
    }
}

} // namespace math
} // namespace cpu
} // namespace backend
} // namespace Kernel
