#include <cmath>
#include <iostream>

#include <core/Network.h>
#include <ops.h>
#include <trustinfer.h>

static bool shape_eq(const Kernel::core::DataShape_t& shape,
                     std::initializer_list<Kernel::UINT> dims) {
    if (shape.ndim != dims.size()) {
        return false;
    }
    Kernel::UINT idx = 0;
    for (auto it = dims.begin(); it != dims.end(); ++it, ++idx) {
        if (shape.dims[idx] != *it) {
            return false;
        }
    }
    return true;
}

static bool has_nonzero_diff_between_batches(const Kernel::core::Value_t& value) {
    if (4 != value.data.shape.ndim) {
        return false;
    }
    const Kernel::UINT batch = value.data.shape.dims[0];
    if (batch < 2) {
        return false;
    }
    const Kernel::UINT per_batch = value.data.shape.size / batch;
    const float* ptr = static_cast<const float*>(value.data.ptr);
    for (Kernel::UINT i = 0; i < per_batch; ++i) {
        if (std::fabs(ptr[i] - ptr[per_batch + i]) > 1e-6f) {
            return true;
        }
    }
    return false;
}

int main() {
    using namespace Kernel::core;


    Value_t graph_input({2, 3, 6, 6});

    Conv2d conv(3, 4, {3, 3}, {1, 1}, {1, 1, 1, 1});
    MaxPool2d pool({2, 2}, {2, 2});
    BatchNorm2d bn(4);

    Layer& l1 = conv(graph_input);
    Layer& l2 = pool(l1.output());
    Layer& l3 = bn(l2.output());

    Graph graph(
        { GraphInputSlot("input", graph_input) },
        { GraphOutputSlot("output", l3.output()) }
    );
    Network network(graph);

    Value_t runtime_input({2, 3, 6, 6});
    runtime_input.alloc();
    fill_random(runtime_input.data.ptr,
                runtime_input.data.dtype,
                runtime_input.data.shape.size,
                TIMESEED);

    Value_t runtime_output;

    network.prepare();
    network.run({ &runtime_input }, { &runtime_output });

    if (!shape_eq(l1.output().data.shape, {2, 4, 6, 6})) {
        std::cerr << "conv output shape mismatch" << std::endl;
        return 1;
    }
    if (!shape_eq(l2.output().data.shape, {2, 4, 3, 3})) {
        std::cerr << "pool output shape mismatch" << std::endl;
        return 1;
    }
    if (!shape_eq(l3.output().data.shape, {2, 4, 3, 3})) {
        std::cerr << "batchnorm output shape mismatch" << std::endl;
        return 1;
    }
    if (!shape_eq(runtime_output.data.shape, {2, 4, 3, 3})) {
        std::cerr << "network output shape mismatch" << std::endl;
        return 1;
    }
    if (print_zero_check("batch_cnn_output", runtime_output)) {
        std::cerr << "batch_cnn output should not be zero" << std::endl;
        return 1;
    }
    if (!has_nonzero_diff_between_batches(runtime_output)) {
        std::cerr << "different batch samples should not collapse to identical outputs" << std::endl;
        return 1;
    }
    if (!(runtime_output.data.flags & PARAM_OUTPUT) || !(runtime_output.data.flags & PARAM_OWN_DATA)) {
        std::cerr << "runtime_output flags invalid" << std::endl;
        return 1;
    }

    std::cout << "batch cnn demo ok" << std::endl;
    return 0;
}
