#include <cmath>
#include <cstring>
#include <iostream>

#include <core/Network.h>
#include <ops.h>
#include <trustinfer.h>

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

void test_relu_sigmoid_dropout() {
    Value_t graph_input({2, 3});
    ReLU relu(false);
    Sigmoid sigmoid(false);
    Dropout dropout(0.25f, false);

    Layer& l1 = relu(graph_input);
    Layer& l2 = sigmoid(l1.output());
    Layer& l3 = dropout(l2.output());

    Graph graph(
        { GraphInputSlot("input", graph_input) },
        { GraphOutputSlot("output", l3.output()) }
    );
    Network net(graph);
    net.prepare();

    Value_t runtime_input({2, 3});
    fill_fp32(runtime_input, {-1.0f, 0.0f, 2.0f, 3.0f, -4.0f, 5.0f});
    Value_t runtime_output;
    net.run({ &runtime_input }, { &runtime_output });

    std::vector<FLOAT> expect(6);
    const std::vector<FLOAT> relu_out = {0.0f, 0.0f, 2.0f, 3.0f, 0.0f, 5.0f};
    for (UINT i = 0; i < expect.size(); ++i) {
        expect[i] = 1.0f / (1.0f + std::exp(-relu_out[i]));
    }
    check_fp32(runtime_output, expect, "relu_sigmoid_dropout");
}

void test_add() {
    Value_t lhs_input({2, 2});
    Value_t rhs_input({2, 2});
    Add add(0.5f);
    Layer& out = add(lhs_input, rhs_input);

    Graph graph(
        { GraphInputSlot("lhs", lhs_input), GraphInputSlot("rhs", rhs_input) },
        { GraphOutputSlot("output", out.output()) }
    );
    Network net(graph);
    net.prepare();

    Value_t lhs({2, 2});
    Value_t rhs({2, 2});
    fill_fp32(lhs, {1.0f, 2.0f, 3.0f, 4.0f});
    fill_fp32(rhs, {10.0f, 20.0f, 30.0f, 40.0f});
    Value_t output;
    net.run({ &lhs, &rhs }, { &output });
    check_fp32(output, {6.0f, 12.0f, 18.0f, 24.0f}, "add");
}

void test_matmul() {
    Value_t lhs_input({2, 3});
    Value_t rhs_input({3, 2});
    MatMul matmul;
    Layer& out = matmul(lhs_input, rhs_input);

    Graph graph(
        { GraphInputSlot("lhs", lhs_input), GraphInputSlot("rhs", rhs_input) },
        { GraphOutputSlot("output", out.output()) }
    );
    Network net(graph);
    net.prepare();

    Value_t lhs({2, 3});
    Value_t rhs({3, 2});
    fill_fp32(lhs, {1.0f, 2.0f, 3.0f,
                    4.0f, 5.0f, 6.0f});
    fill_fp32(rhs, {7.0f, 8.0f,
                    9.0f, 10.0f,
                    11.0f, 12.0f});
    Value_t output;
    net.run({ &lhs, &rhs }, { &output });
    check_fp32(output, {58.0f, 64.0f, 139.0f, 154.0f}, "matmul");
}

void test_biasadd() {
    Value_t graph_input({2, 3, 2});
    BiasAdd biasadd(3, 1);
    Layer& out = biasadd(graph_input);

    const Data_t* bias = out.param(ParamRole::BIAS);
    EXIT_ERROR_CHECK_EQ(nullptr, bias, "BiasAdd bias missing");
    FLOAT* bias_ptr = static_cast<FLOAT*>(bias->ptr);
    const FLOAT custom_bias[3] = {1.0f, -2.0f, 3.0f};
    std::memcpy(bias_ptr, custom_bias, sizeof(custom_bias));

    Graph graph(
        { GraphInputSlot("input", graph_input) },
        { GraphOutputSlot("output", out.output()) }
    );
    Network net(graph);
    net.prepare();

    Value_t runtime_input({2, 3, 2});
    fill_fp32(runtime_input, {
        1.0f, 2.0f,  3.0f, 4.0f,  5.0f, 6.0f,
        7.0f, 8.0f,  9.0f, 10.0f, 11.0f, 12.0f
    });
    Value_t runtime_output;
    net.run({ &runtime_input }, { &runtime_output });
    check_fp32(runtime_output, {
        2.0f, 3.0f,  1.0f, 2.0f,  8.0f, 9.0f,
        8.0f, 9.0f,  7.0f, 8.0f, 14.0f, 15.0f
    }, "biasadd");
}

void test_linear() {
    Value_t graph_input({2, 4});
    Linear linear(4, 3, true);
    Layer& out = linear(graph_input);

    const Data_t* weight = out.param(ParamRole::WEIGHT);
    const Data_t* bias = out.param(ParamRole::BIAS);
    EXIT_ERROR_CHECK_EQ(nullptr, weight, "Linear weight missing");
    EXIT_ERROR_CHECK_EQ(nullptr, bias, "Linear bias missing");
    FLOAT* w_ptr = static_cast<FLOAT*>(weight->ptr);
    FLOAT* b_ptr = static_cast<FLOAT*>(bias->ptr);
    const FLOAT custom_weight[12] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.5f, 1.0f
    };
    const FLOAT custom_bias[3] = {0.5f, -1.0f, 2.0f};
    std::memcpy(w_ptr, custom_weight, sizeof(custom_weight));
    std::memcpy(b_ptr, custom_bias, sizeof(custom_bias));

    Graph graph(
        { GraphInputSlot("input", graph_input) },
        { GraphOutputSlot("output", out.output()) }
    );
    Network net(graph);
    net.prepare();

    Value_t runtime_input({2, 4});
    fill_fp32(runtime_input, {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f
    });
    Value_t runtime_output;
    net.run({ &runtime_input }, { &runtime_output });
    check_fp32(runtime_output, {
        1.5f, 4.0f, 7.5f,
        5.5f, 12.0f, 13.5f
    }, "linear");
}

void test_softmax() {
    Value_t graph_input({2, 3});
    Softmax softmax(1);
    Layer& out = softmax(graph_input);

    Graph graph(
        { GraphInputSlot("input", graph_input) },
        { GraphOutputSlot("output", out.output()) }
    );
    Network net(graph);
    net.prepare();

    Value_t runtime_input({2, 3});
    fill_fp32(runtime_input, {1.0f, 2.0f, 3.0f, 1.0f, 1.0f, 1.0f});
    Value_t runtime_output;
    net.run({ &runtime_input }, { &runtime_output });

    const FLOAT e1 = std::exp(1.0f);
    const FLOAT e2 = std::exp(2.0f);
    const FLOAT e3 = std::exp(3.0f);
    const FLOAT s = e1 + e2 + e3;
    check_fp32(runtime_output, {
        e1 / s, e2 / s, e3 / s,
        1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f
    }, "softmax");
}

void test_concat() {
    Value_t input_a({1, 2, 2});
    Value_t input_b({1, 1, 2});
    Concat concat(1);
    Layer& out = concat(input_a, input_b);

    Graph graph(
        { GraphInputSlot("a", input_a), GraphInputSlot("b", input_b) },
        { GraphOutputSlot("output", out.output()) }
    );
    Network net(graph);
    net.prepare();

    Value_t a({1, 2, 2});
    Value_t b({1, 1, 2});
    fill_fp32(a, {1.0f, 2.0f, 3.0f, 4.0f});
    fill_fp32(b, {5.0f, 6.0f});
    Value_t output;
    net.run({ &a, &b }, { &output });
    check_fp32(output, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, "concat");
}

void test_adaptive_pool() {
    Value_t graph_input({1, 1, 4, 4});
    AdaptiveAvgPool2d avg_pool({2, 2});
    AdaptiveMaxPool2d max_pool({2, 2});
    Layer& avg_out = avg_pool(graph_input);
    Layer& max_out = max_pool(graph_input);

    Graph avg_graph(
        { GraphInputSlot("input", graph_input) },
        { GraphOutputSlot("output", avg_out.output()) }
    );
    Network avg_net(avg_graph);
    avg_net.prepare();

    Value_t runtime_input({1, 1, 4, 4});
    fill_fp32(runtime_input, {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    });
    Value_t avg_output;
    avg_net.run({ &runtime_input }, { &avg_output });
    check_fp32(avg_output, {3.5f, 5.5f, 11.5f, 13.5f}, "adaptive_avgpool2d");

    Graph max_graph(
        { GraphInputSlot("input", graph_input) },
        { GraphOutputSlot("output", max_out.output()) }
    );
    Network max_net(max_graph);
    max_net.prepare();
    Value_t max_output;
    max_net.run({ &runtime_input }, { &max_output });
    check_fp32(max_output, {6.0f, 8.0f, 14.0f, 16.0f}, "adaptive_maxpool2d");
}

} // namespace

int main() {
    test_relu_sigmoid_dropout();
    test_add();
    test_biasadd();
    test_matmul();
    test_linear();
    test_softmax();
    test_concat();
    test_adaptive_pool();
    std::cout << "basic ops demo ok" << std::endl;
    return 0;
}
