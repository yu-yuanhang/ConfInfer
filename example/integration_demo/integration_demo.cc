#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#include <core/Network.h>
#include <ops.h>
#include <trustinfer.h>

using namespace Kernel;
using namespace Kernel::core;

namespace {

bool close_enough(FLOAT a, FLOAT b, FLOAT eps = 1e-4f) {
    return std::fabs(a - b) <= eps;
}

void fill_fp32(Value_t& value, const std::vector<FLOAT>& data) {
    EXIT_ERROR_CHECK_NE(value.data.shape.size, data.size(), "fill_fp32 size mismatch");
    if (nullptr == value.data.ptr) {
        value.alloc();
    }
    FLOAT* ptr = static_cast<FLOAT*>(value.data.ptr);
    for (UINT i = 0; i < data.size(); ++i) {
        ptr[i] = data[i];
    }
}

void check_fp32(const Value_t& value, const std::vector<FLOAT>& expect, const char* name) {
    EXIT_ERROR_CHECK_EQ(nullptr, value.data.ptr, "output ptr is nullptr");
    EXIT_ERROR_CHECK_NE(value.data.shape.size, expect.size(), "check_fp32 size mismatch");
    const FLOAT* ptr = static_cast<const FLOAT*>(value.data.ptr);
    for (UINT i = 0; i < expect.size(); ++i) {
        if (!close_enough(ptr[i], expect[i])) {
            std::cerr << name << " mismatch at " << i
                      << " got=" << ptr[i] << " expect=" << expect[i] << std::endl;
            std::exit(1);
        }
    }
}

void check_i32(const Value_t& value, const std::vector<int32_t>& expect, const char* name) {
    EXIT_ERROR_CHECK_EQ(nullptr, value.data.ptr, "output ptr is nullptr");
    EXIT_ERROR_CHECK_NE(value.data.shape.size, expect.size(), "check_i32 size mismatch");
    const int32_t* ptr = static_cast<const int32_t*>(value.data.ptr);
    for (UINT i = 0; i < expect.size(); ++i) {
        if (ptr[i] != expect[i]) {
            std::cerr << name << " mismatch at " << i
                      << " got=" << ptr[i] << " expect=" << expect[i] << std::endl;
            std::exit(1);
        }
    }
}

void ref_conv2d_1x1_nchw(const std::vector<FLOAT>& input,
                         std::vector<FLOAT>& output,
                         const std::vector<FLOAT>& weight,
                         const std::vector<FLOAT>& bias,
                         UINT batch,
                         UINT in_channels,
                         UINT out_channels,
                         UINT height,
                         UINT width) {
    output.assign(batch * out_channels * height * width, 0.0f);
    const UINT in_ch_stride = height * width;
    const UINT out_ch_stride = height * width;
    const UINT in_batch_stride = in_channels * in_ch_stride;
    const UINT out_batch_stride = out_channels * out_ch_stride;
    for (UINT n = 0; n < batch; ++n) {
        for (UINT oc = 0; oc < out_channels; ++oc) {
            for (UINT h = 0; h < height; ++h) {
                for (UINT w = 0; w < width; ++w) {
                    FLOAT sum = bias[oc];
                    for (UINT ic = 0; ic < in_channels; ++ic) {
                        const UINT in_idx = n * in_batch_stride + ic * in_ch_stride + h * width + w;
                        const UINT w_idx = oc * in_channels + ic;
                        sum += input[in_idx] * weight[w_idx];
                    }
                    const UINT out_idx = n * out_batch_stride + oc * out_ch_stride + h * width + w;
                    output[out_idx] = sum;
                }
            }
        }
    }
}

void ref_relu(std::vector<FLOAT>& data) {
    for (UINT i = 0; i < data.size(); ++i) {
        if (data[i] < 0.0f) {
            data[i] = 0.0f;
        }
    }
}

void ref_maxpool2d_nchw(const std::vector<FLOAT>& input,
                        std::vector<FLOAT>& output,
                        std::vector<int32_t>& indices,
                        UINT batch,
                        UINT channels,
                        UINT in_h,
                        UINT in_w,
                        UINT kernel_h,
                        UINT kernel_w,
                        UINT stride_h,
                        UINT stride_w) {
    const UINT out_h = (in_h - kernel_h) / stride_h + 1;
    const UINT out_w = (in_w - kernel_w) / stride_w + 1;
    output.assign(batch * channels * out_h * out_w, 0.0f);
    indices.assign(batch * channels * out_h * out_w, -1);
    const UINT in_ch_stride = in_h * in_w;
    const UINT out_ch_stride = out_h * out_w;
    const UINT in_batch_stride = channels * in_ch_stride;
    const UINT out_batch_stride = channels * out_ch_stride;
    for (UINT n = 0; n < batch; ++n) {
        for (UINT c = 0; c < channels; ++c) {
            const FLOAT* in_c = input.data() + n * in_batch_stride + c * in_ch_stride;
            FLOAT* out_c = output.data() + n * out_batch_stride + c * out_ch_stride;
            int32_t* idx_c = indices.data() + n * out_batch_stride + c * out_ch_stride;
            for (UINT oh = 0; oh < out_h; ++oh) {
                for (UINT ow = 0; ow < out_w; ++ow) {
                    FLOAT max_v = -1e30f;
                    int32_t max_idx = -1;
                    for (UINT kh = 0; kh < kernel_h; ++kh) {
                        const UINT ih = oh * stride_h + kh;
                        for (UINT kw = 0; kw < kernel_w; ++kw) {
                            const UINT iw = ow * stride_w + kw;
                            const UINT flat = ih * in_w + iw;
                            const FLOAT v = in_c[flat];
                            if (v > max_v) {
                                max_v = v;
                                max_idx = static_cast<int32_t>(flat);
                            }
                        }
                    }
                    out_c[oh * out_w + ow] = max_v;
                    idx_c[oh * out_w + ow] = max_idx;
                }
            }
        }
    }
}

void ref_avgpool2d_nchw(const std::vector<FLOAT>& input,
                        std::vector<FLOAT>& output,
                        UINT batch,
                        UINT channels,
                        UINT in_h,
                        UINT in_w,
                        UINT kernel_h,
                        UINT kernel_w,
                        UINT stride_h,
                        UINT stride_w) {
    const UINT out_h = (in_h - kernel_h) / stride_h + 1;
    const UINT out_w = (in_w - kernel_w) / stride_w + 1;
    output.assign(batch * channels * out_h * out_w, 0.0f);
    const UINT in_ch_stride = in_h * in_w;
    const UINT out_ch_stride = out_h * out_w;
    const UINT in_batch_stride = channels * in_ch_stride;
    const UINT out_batch_stride = channels * out_ch_stride;
    const FLOAT denom = static_cast<FLOAT>(kernel_h * kernel_w);
    for (UINT n = 0; n < batch; ++n) {
        for (UINT c = 0; c < channels; ++c) {
            const FLOAT* in_c = input.data() + n * in_batch_stride + c * in_ch_stride;
            FLOAT* out_c = output.data() + n * out_batch_stride + c * out_ch_stride;
            for (UINT oh = 0; oh < out_h; ++oh) {
                for (UINT ow = 0; ow < out_w; ++ow) {
                    FLOAT sum = 0.0f;
                    for (UINT kh = 0; kh < kernel_h; ++kh) {
                        const UINT ih = oh * stride_h + kh;
                        for (UINT kw = 0; kw < kernel_w; ++kw) {
                            const UINT iw = ow * stride_w + kw;
                            sum += in_c[ih * in_w + iw];
                        }
                    }
                    out_c[oh * out_w + ow] = sum / denom;
                }
            }
        }
    }
}

void ref_adaptive_pool2d_nchw(const std::vector<FLOAT>& input,
                              std::vector<FLOAT>& output,
                              std::vector<int32_t>* indices,
                              UINT batch,
                              UINT channels,
                              UINT in_h,
                              UINT in_w,
                              UINT out_h,
                              UINT out_w,
                              bool max_mode) {
    output.assign(batch * channels * out_h * out_w, 0.0f);
    if (nullptr != indices) {
        indices->assign(batch * channels * out_h * out_w, -1);
    }
    const UINT in_ch_stride = in_h * in_w;
    const UINT out_ch_stride = out_h * out_w;
    const UINT in_batch_stride = channels * in_ch_stride;
    const UINT out_batch_stride = channels * out_ch_stride;
    for (UINT n = 0; n < batch; ++n) {
        for (UINT c = 0; c < channels; ++c) {
            const FLOAT* in_c = input.data() + n * in_batch_stride + c * in_ch_stride;
            FLOAT* out_c = output.data() + n * out_batch_stride + c * out_ch_stride;
            int32_t* idx_c = (nullptr == indices) ? nullptr : (indices->data() + n * out_batch_stride + c * out_ch_stride);
            for (UINT oh = 0; oh < out_h; ++oh) {
                const UINT h_start = (oh * in_h) / out_h;
                const UINT h_end = ((oh + 1) * in_h + out_h - 1) / out_h;
                for (UINT ow = 0; ow < out_w; ++ow) {
                    const UINT w_start = (ow * in_w) / out_w;
                    const UINT w_end = ((ow + 1) * in_w + out_w - 1) / out_w;
                    if (max_mode) {
                        FLOAT max_v = -1e30f;
                        int32_t max_idx = -1;
                        for (UINT ih = h_start; ih < h_end; ++ih) {
                            for (UINT iw = w_start; iw < w_end; ++iw) {
                                const UINT flat = ih * in_w + iw;
                                const FLOAT v = in_c[flat];
                                if (v > max_v) {
                                    max_v = v;
                                    max_idx = static_cast<int32_t>(flat);
                                }
                            }
                        }
                        out_c[oh * out_w + ow] = max_v;
                        if (nullptr != idx_c) {
                            idx_c[oh * out_w + ow] = max_idx;
                        }
                    } else {
                        FLOAT sum = 0.0f;
                        UINT count = 0;
                        for (UINT ih = h_start; ih < h_end; ++ih) {
                            for (UINT iw = w_start; iw < w_end; ++iw) {
                                sum += in_c[ih * in_w + iw];
                                ++count;
                            }
                        }
                        out_c[oh * out_w + ow] = sum / static_cast<FLOAT>(count);
                    }
                }
            }
        }
    }
}

void ref_concat_axis1_nchw(const std::vector<std::vector<FLOAT>>& inputs,
                           const std::vector<UINT>& channels_per_input,
                           std::vector<FLOAT>& output,
                           UINT batch,
                           UINT height,
                           UINT width) {
    UINT total_channels = 0;
    for (UINT i = 0; i < channels_per_input.size(); ++i) {
        total_channels += channels_per_input[i];
    }
    output.assign(batch * total_channels * height * width, 0.0f);
    const UINT spatial = height * width;
    UINT channel_base = 0;
    for (UINT src_idx = 0; src_idx < inputs.size(); ++src_idx) {
        const UINT src_channels = channels_per_input[src_idx];
        for (UINT n = 0; n < batch; ++n) {
            const FLOAT* src_n = inputs[src_idx].data() + n * src_channels * spatial;
            FLOAT* dst_n = output.data() + n * total_channels * spatial + channel_base * spatial;
            std::memcpy(dst_n, src_n, sizeof(FLOAT) * src_channels * spatial);
        }
        channel_base += src_channels;
    }
}

void ref_biasadd_channel_nchw(std::vector<FLOAT>& data,
                              const std::vector<FLOAT>& bias,
                              UINT batch,
                              UINT channels,
                              UINT height,
                              UINT width) {
    const UINT spatial = height * width;
    for (UINT n = 0; n < batch; ++n) {
        for (UINT c = 0; c < channels; ++c) {
            FLOAT* ptr = data.data() + (n * channels + c) * spatial;
            for (UINT i = 0; i < spatial; ++i) {
                ptr[i] += bias[c];
            }
        }
    }
}

void ref_batchnorm2d_inference(std::vector<FLOAT>& data,
                               const std::vector<FLOAT>& gamma,
                               const std::vector<FLOAT>& beta,
                               const std::vector<FLOAT>& mean,
                               const std::vector<FLOAT>& var,
                               FLOAT eps,
                               UINT batch,
                               UINT channels,
                               UINT height,
                               UINT width) {
    const UINT spatial = height * width;
    for (UINT n = 0; n < batch; ++n) {
        for (UINT c = 0; c < channels; ++c) {
            FLOAT* ptr = data.data() + (n * channels + c) * spatial;
            const FLOAT inv_std = 1.0f / std::sqrt(var[c] + eps);
            for (UINT i = 0; i < spatial; ++i) {
                ptr[i] = ((ptr[i] - mean[c]) * inv_std) * gamma[c] + beta[c];
            }
        }
    }
}

void ref_softmax_channel_nchw(std::vector<FLOAT>& data,
                              UINT batch,
                              UINT channels,
                              UINT height,
                              UINT width) {
    const UINT spatial = height * width;
    for (UINT n = 0; n < batch; ++n) {
        for (UINT s = 0; s < spatial; ++s) {
            FLOAT max_v = -1e30f;
            for (UINT c = 0; c < channels; ++c) {
                const FLOAT v = data[(n * channels + c) * spatial + s];
                if (v > max_v) {
                    max_v = v;
                }
            }
            FLOAT sum = 0.0f;
            for (UINT c = 0; c < channels; ++c) {
                FLOAT& v = data[(n * channels + c) * spatial + s];
                v = std::exp(v - max_v);
                sum += v;
            }
            for (UINT c = 0; c < channels; ++c) {
                FLOAT& v = data[(n * channels + c) * spatial + s];
                v /= sum;
            }
        }
    }
}

void ref_linear_2d(const std::vector<FLOAT>& input,
                   std::vector<FLOAT>& output,
                   const std::vector<FLOAT>& weight,
                   const std::vector<FLOAT>& bias,
                   UINT batch,
                   UINT in_features,
                   UINT out_features) {
    output.assign(batch * out_features, 0.0f);
    for (UINT n = 0; n < batch; ++n) {
        for (UINT o = 0; o < out_features; ++o) {
            FLOAT sum = bias[o];
            for (UINT i = 0; i < in_features; ++i) {
                sum += input[n * in_features + i] * weight[o * in_features + i];
            }
            output[n * out_features + o] = sum;
        }
    }
}

void ref_biasadd_dim1_2d(std::vector<FLOAT>& data,
                         const std::vector<FLOAT>& bias,
                         UINT batch,
                         UINT features) {
    for (UINT n = 0; n < batch; ++n) {
        for (UINT f = 0; f < features; ++f) {
            data[n * features + f] += bias[f];
        }
    }
}

void ref_matmul_2d(const std::vector<FLOAT>& lhs,
                   const std::vector<FLOAT>& rhs,
                   std::vector<FLOAT>& out,
                   UINT m,
                   UINT k,
                   UINT n) {
    out.assign(m * n, 0.0f);
    for (UINT i = 0; i < m; ++i) {
        for (UINT j = 0; j < n; ++j) {
            FLOAT sum = 0.0f;
            for (UINT kk = 0; kk < k; ++kk) {
                sum += lhs[i * k + kk] * rhs[kk * n + j];
            }
            out[i * n + j] = sum;
        }
    }
}

void ref_add_inplace(std::vector<FLOAT>& lhs,
                     const std::vector<FLOAT>& rhs,
                     FLOAT alpha) {
    EXIT_ERROR_CHECK_NE(lhs.size(), rhs.size(), "ref_add_inplace size mismatch");
    for (UINT i = 0; i < lhs.size(); ++i) {
        lhs[i] += alpha * rhs[i];
    }
}

void ref_layernorm_lastdim(std::vector<FLOAT>& data,
                           const std::vector<FLOAT>& gamma,
                           const std::vector<FLOAT>& beta,
                           FLOAT eps,
                           UINT outer,
                           UINT inner) {
    for (UINT o = 0; o < outer; ++o) {
        FLOAT mean = 0.0f;
        for (UINT i = 0; i < inner; ++i) {
            mean += data[o * inner + i];
        }
        mean /= static_cast<FLOAT>(inner);

        FLOAT var = 0.0f;
        for (UINT i = 0; i < inner; ++i) {
            const FLOAT diff = data[o * inner + i] - mean;
            var += diff * diff;
        }
        var /= static_cast<FLOAT>(inner);

        const FLOAT inv_std = 1.0f / std::sqrt(var + eps);
        for (UINT i = 0; i < inner; ++i) {
            FLOAT norm = (data[o * inner + i] - mean) * inv_std;
            data[o * inner + i] = norm * gamma[i] + beta[i];
        }
    }
}

void ref_sigmoid(std::vector<FLOAT>& data) {
    for (UINT i = 0; i < data.size(); ++i) {
        data[i] = 1.0f / (1.0f + std::exp(-data[i]));
    }
}

void test_complex_cnn_graph() {
    Value_t graph_input({1, 1, 4, 4});

    Conv2d conv(1, 2, {1, 1}, {1, 1}, {0, 0, 0, 0}, {1, 1}, 1, true);
    ReLU relu(false);
    MaxPool2d max_pool({2, 2}, {2, 2}, {0, 0}, {1, 1}, true);
    AvgPool2d avg_pool({2, 2}, {2, 2});
    AdaptiveMaxPool2d adaptive_max_pool({2, 2}, true);
    AdaptiveAvgPool2d adaptive_avg_pool({2, 2});
    Concat concat(1);
    BiasAdd biasadd(8, 1);
    BatchNorm2d bn(8, 0.0f, 0.1f, true, true);
    AdaptiveAvgPool2d final_pool({1, 1});
    Softmax softmax(1);

    Layer& l1 = conv(graph_input);
    Layer& l2 = relu(l1.output());
    Layer& l3 = max_pool(l2.output());
    Layer& l4 = avg_pool(l2.output());
    Layer& l5 = adaptive_max_pool(l2.output());
    Layer& l6 = adaptive_avg_pool(l2.output());
    Layer& l7 = concat(
        l3.output(OutputKind::Default),
        l4.output(),
        l5.output(OutputKind::Default),
        l6.output()
    );
    Layer& l8 = biasadd(l7.output());
    Layer& l9 = bn(l8.output());
    Layer& l10 = final_pool(l9.output());
    Layer& l11 = softmax(l10.output());

    {
        FLOAT* w = static_cast<FLOAT*>(l1.param(ParamRole::WEIGHT)->ptr);
        FLOAT* b = static_cast<FLOAT*>(l1.param(ParamRole::BIAS)->ptr);
        w[0] = 1.0f;
        w[1] = -1.0f;
        b[0] = 0.0f;
        b[1] = 20.0f;
    }
    {
        FLOAT* bias = static_cast<FLOAT*>(l8.param(ParamRole::BIAS)->ptr);
        for (UINT i = 0; i < 8; ++i) {
            bias[i] = static_cast<FLOAT>(i);
        }
    }
    {
        FLOAT* gamma = static_cast<FLOAT*>(l9.param(ParamRole::WEIGHT)->ptr);
        FLOAT* beta = static_cast<FLOAT*>(l9.param(ParamRole::BIAS)->ptr);
        FLOAT* mean = static_cast<FLOAT*>(l9.param(ParamRole::RUNNING_MEAN)->ptr);
        FLOAT* var = static_cast<FLOAT*>(l9.param(ParamRole::RUNNING_VAR)->ptr);
        for (UINT i = 0; i < 8; ++i) {
            gamma[i] = 1.0f;
            beta[i] = 0.0f;
            mean[i] = static_cast<FLOAT>(i);
            var[i] = 1.0f;
        }
    }

    Graph graph(
        { GraphInputSlot("input", graph_input) },
        {
            GraphOutputSlot("softmax", l11.output()),
            GraphOutputSlot("max_idx", l3.output(OutputKind::Indices)),
            GraphOutputSlot("adaptive_max_idx", l5.output(OutputKind::Indices))
        }
    );
    Network net(graph);
    net.prepare();

    Value_t runtime_input({1, 1, 4, 4});
    fill_fp32(runtime_input, {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    });

    Value_t softmax_output;
    Value_t max_indices;
    Value_t adaptive_max_indices;
    net.run({ &runtime_input }, { &softmax_output, &max_indices, &adaptive_max_indices });

    const std::vector<FLOAT> input = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    };
    std::vector<FLOAT> conv_out;
    ref_conv2d_1x1_nchw(input, conv_out, {1.0f, -1.0f}, {0.0f, 20.0f}, 1, 1, 2, 4, 4);
    ref_relu(conv_out);

    std::vector<FLOAT> max_out;
    std::vector<int32_t> max_idx_ref;
    ref_maxpool2d_nchw(conv_out, max_out, max_idx_ref, 1, 2, 4, 4, 2, 2, 2, 2);

    std::vector<FLOAT> avg_out;
    ref_avgpool2d_nchw(conv_out, avg_out, 1, 2, 4, 4, 2, 2, 2, 2);

    std::vector<FLOAT> adaptive_max_out;
    std::vector<int32_t> adaptive_max_idx_ref;
    ref_adaptive_pool2d_nchw(conv_out, adaptive_max_out, &adaptive_max_idx_ref, 1, 2, 4, 4, 2, 2, true);

    std::vector<FLOAT> adaptive_avg_out;
    ref_adaptive_pool2d_nchw(conv_out, adaptive_avg_out, nullptr, 1, 2, 4, 4, 2, 2, false);

    std::vector<FLOAT> concat_out;
    ref_concat_axis1_nchw(
        { max_out, avg_out, adaptive_max_out, adaptive_avg_out },
        { 2, 2, 2, 2 },
        concat_out, 1, 2, 2);
    ref_biasadd_channel_nchw(concat_out, {0, 1, 2, 3, 4, 5, 6, 7}, 1, 8, 2, 2);
    ref_batchnorm2d_inference(concat_out,
        {1, 1, 1, 1, 1, 1, 1, 1},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 1, 2, 3, 4, 5, 6, 7},
        {1, 1, 1, 1, 1, 1, 1, 1},
        0.0f, 1, 8, 2, 2);

    std::vector<FLOAT> pooled;
    ref_adaptive_pool2d_nchw(concat_out, pooled, nullptr, 1, 8, 2, 2, 1, 1, false);
    ref_softmax_channel_nchw(pooled, 1, 8, 1, 1);

    check_fp32(softmax_output, pooled, "complex_cnn_softmax");
    check_i32(max_indices, max_idx_ref, "complex_cnn_max_indices");
    check_i32(adaptive_max_indices, adaptive_max_idx_ref, "complex_cnn_adaptive_max_indices");
}

void test_dense_residual_graph() {
    Value_t graph_x({2, 4});
    Value_t graph_w({4, 3});

    Linear linear(4, 3, true);
    ReLU relu(false);
    BiasAdd biasadd(3, 1);
    MatMul matmul;
    Add add(0.25f);
    LayerNorm layernorm({3}, 0.0f, true);
    Sigmoid sigmoid(false);
    Dropout dropout(0.2f, false);

    Layer& l1 = linear(graph_x);
    Layer& l2 = relu(l1.output());
    Layer& l3 = biasadd(l2.output());
    Layer& l4 = matmul(graph_x, graph_w);
    Layer& l5 = add(l3.output(), l4.output());
    Layer& l6 = layernorm(l5.output());
    Layer& l7 = sigmoid(l6.output());
    Layer& l8 = dropout(l7.output());

    {
        FLOAT* w = static_cast<FLOAT*>(l1.param(ParamRole::WEIGHT)->ptr);
        FLOAT* b = static_cast<FLOAT*>(l1.param(ParamRole::BIAS)->ptr);
        const FLOAT custom_weight[12] = {
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 1.0f
        };
        const FLOAT custom_bias[3] = {0.5f, -1.0f, 2.0f};
        std::memcpy(w, custom_weight, sizeof(custom_weight));
        std::memcpy(b, custom_bias, sizeof(custom_bias));
    }
    {
        FLOAT* bias = static_cast<FLOAT*>(l3.param(ParamRole::BIAS)->ptr);
        const FLOAT custom_bias[3] = {1.0f, -2.0f, 0.5f};
        std::memcpy(bias, custom_bias, sizeof(custom_bias));
    }
    {
        FLOAT* gamma = static_cast<FLOAT*>(l6.param(ParamRole::WEIGHT)->ptr);
        FLOAT* beta = static_cast<FLOAT*>(l6.param(ParamRole::BIAS)->ptr);
        const FLOAT custom_gamma[3] = {1.0f, 0.5f, 2.0f};
        const FLOAT custom_beta[3] = {0.0f, 1.0f, -1.0f};
        std::memcpy(gamma, custom_gamma, sizeof(custom_gamma));
        std::memcpy(beta, custom_beta, sizeof(custom_beta));
    }

    Graph graph(
        { GraphInputSlot("x", graph_x), GraphInputSlot("w", graph_w) },
        { GraphOutputSlot("output", l8.output()) }
    );
    Network net(graph);
    net.prepare();

    Value_t runtime_x({2, 4});
    Value_t runtime_w({4, 3});
    fill_fp32(runtime_x, {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f
    });
    fill_fp32(runtime_w, {
        1.0f, 2.0f, 3.0f,
        0.0f, 1.0f, 0.0f,
        1.0f, 0.0f, 1.0f,
        0.0f, -1.0f, 2.0f
    });

    Value_t runtime_output;
    net.run({ &runtime_x, &runtime_w }, { &runtime_output });

    std::vector<FLOAT> linear_out;
    ref_linear_2d(
        {
            1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f
        },
        linear_out,
        {
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 1.0f
        },
        {0.5f, -1.0f, 2.0f},
        2, 4, 3);
    ref_relu(linear_out);
    ref_biasadd_dim1_2d(linear_out, {1.0f, -2.0f, 0.5f}, 2, 3);

    std::vector<FLOAT> matmul_out;
    ref_matmul_2d(
        {
            1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f
        },
        {
            1.0f, 2.0f, 3.0f,
            0.0f, 1.0f, 0.0f,
            1.0f, 0.0f, 1.0f,
            0.0f, -1.0f, 2.0f
        },
        matmul_out,
        2, 4, 3);
    ref_add_inplace(linear_out, matmul_out, 0.25f);
    ref_layernorm_lastdim(linear_out, {1.0f, 0.5f, 2.0f}, {0.0f, 1.0f, -1.0f}, 0.0f, 2, 3);
    ref_sigmoid(linear_out);

    check_fp32(runtime_output, linear_out, "dense_residual_output");
}

} // namespace

int main() {
    test_complex_cnn_graph();
    test_dense_residual_graph();
    std::cout << "integration demo ok" << std::endl;
    return 0;
}
