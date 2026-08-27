#include <cmath>
#include <iostream>

#include <core/Network.h>
#include <ops.h>
#include <trustinfer.h>

static bool nearly_equal(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) <= eps;
}

static bool check_tensor(const Kernel::core::Value_t& value,
                         const float* expected,
                         Kernel::UINT size,
                         const char* name) {
    const float* ptr = static_cast<const float*>(value.data.ptr);
    if (nullptr == ptr) {
        std::cerr << name << " ptr is nullptr" << std::endl;
        return false;
    }
    for (Kernel::UINT i = 0; i < size; ++i) {
        if (!nearly_equal(ptr[i], expected[i])) {
            std::cerr << name << " mismatch at " << i
                      << " expected=" << expected[i]
                      << " actual=" << ptr[i] << std::endl;
            return false;
        }
    }
    return true;
}

static bool check_tensor_i32(const Kernel::core::Value_t& value,
                             const int32_t* expected,
                             Kernel::UINT size,
                             const char* name) {
    const int32_t* ptr = static_cast<const int32_t*>(value.data.ptr);
    if (nullptr == ptr) {
        std::cerr << name << " ptr is nullptr" << std::endl;
        return false;
    }
    for (Kernel::UINT i = 0; i < size; ++i) {
        if (ptr[i] != expected[i]) {
            std::cerr << name << " mismatch at " << i
                      << " expected=" << expected[i]
                      << " actual=" << ptr[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    using namespace Kernel::core;


    Value_t graph_input({1, 1, 4, 4});

    MaxPool2d max_pool({2, 2}, {2, 2}, {0, 0}, {1, 1}, true);
    AvgPool2d avg_pool({2, 2}, {2, 2});
    AdaptiveMaxPool2d adaptive_max_pool({2, 2}, true);

    Layer& max_layer = max_pool(graph_input);
    Layer& avg_layer = avg_pool(graph_input);
    Layer& adaptive_max_layer = adaptive_max_pool(graph_input);

    Graph graph(
        { GraphInputSlot("input", graph_input) },
        {
            GraphOutputSlot("max_out", max_layer.output(OutputKind::Default)),
            GraphOutputSlot("max_idx", max_layer.output(OutputKind::Indices)),
            GraphOutputSlot("avg_out", avg_layer.output()),
            GraphOutputSlot("adaptive_max_out", adaptive_max_layer.output(OutputKind::Default)),
            GraphOutputSlot("adaptive_max_idx", adaptive_max_layer.output(OutputKind::Indices))
        }
    );
    Network network(graph);

    Value_t runtime_input({1, 1, 4, 4});
    runtime_input.alloc();
    float* in_ptr = static_cast<float*>(runtime_input.data.ptr);
    for (UINT i = 0; i < 16; ++i) {
        in_ptr[i] = static_cast<float>(i + 1);
    }

    Value_t max_output;
    Value_t max_indices;
    Value_t avg_output;
    Value_t adaptive_max_output;
    Value_t adaptive_max_indices;

    network.prepare();
    network.run({ &runtime_input }, {
        &max_output, &max_indices, &avg_output, &adaptive_max_output, &adaptive_max_indices
    });

    const float expected_max[] = {
        6.0f, 8.0f,
        14.0f, 16.0f
    };
    const float expected_avg[] = {
        3.5f, 5.5f,
        11.5f, 13.5f
    };
    const int32_t expected_max_idx[] = {
        5, 7,
        13, 15
    };

    if (!check_tensor(max_output, expected_max, 4, "max_output")) {
        return 1;
    }
    if (!check_tensor_i32(max_indices, expected_max_idx, 4, "max_indices")) {
        return 1;
    }
    if (!check_tensor(avg_output, expected_avg, 4, "avg_output")) {
        return 1;
    }
    if (!check_tensor(adaptive_max_output, expected_max, 4, "adaptive_max_output")) {
        return 1;
    }
    if (!check_tensor_i32(adaptive_max_indices, expected_max_idx, 4, "adaptive_max_indices")) {
        return 1;
    }
    if (!(max_output.data.flags & PARAM_OUTPUT) || !(max_output.data.flags & PARAM_OWN_DATA)) {
        std::cerr << "max_output flags invalid" << std::endl;
        return 1;
    }
    if (DataType::INT32 != max_indices.data.dtype) {
        std::cerr << "max_indices dtype invalid" << std::endl;
        return 1;
    }
    if (!(avg_output.data.flags & PARAM_OUTPUT) || !(avg_output.data.flags & PARAM_OWN_DATA)) {
        std::cerr << "avg_output flags invalid" << std::endl;
        return 1;
    }

    std::cout << "pool demo ok" << std::endl;
    return 0;
}
