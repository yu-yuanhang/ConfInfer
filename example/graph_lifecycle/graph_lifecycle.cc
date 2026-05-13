#include <iostream>
#include <vector>

#include <core/Network.h>
#include <pool.h>
#include <trustinfer.h>
#include <ops.h>

using namespace Kernel::core;

static bool all_zero(Value_t& value) {
    if (nullptr == value.data.ptr) {
        std::cerr << "value.data.ptr is nullptr" << std::endl;
        return false;
    }
    if (DataType::FP32 != value.data.dtype) {
        EXIT_ERROR("graph_lifecycle test only supports FP32");
    }

    float32* ptr = static_cast<float32*>(value.data.ptr);
    for (uint32_t i = 0; i < value.data.shape.size; ++i) {
        if (0.0f != ptr[i]) {
            return false;
        }
    }
    return true;
}

int main() {
    RUNTIME->setThreadsNum(1);

    Conv2d conv(3, 3, {3, 3}, {1, 1}, {1, 1, 1, 1});
    MaxPool2d pool({2, 2});

    Value_t graph_input({1, 3, 8, 8});
    Layer& l1 = conv(graph_input);
    Layer& l2 = pool(l1.output());
    Layer& l3 = conv(l2.output());

    Value_t runtime_output;
    void* output_ptr_in_scope = nullptr;
    uint32_t output_size_in_scope = 0;
    uint32_t output_flags_in_scope = 0;

    {
        Graph graph(
            { GraphInputSlot("input", graph_input) },
            { GraphOutputSlot("output", l3.output()) }
        );
        Network network(graph, RUNTIME);

        Value_t runtime_input({1, 3, 8, 8});
        runtime_input.alloc();
        fill_random(runtime_input.data.ptr,
                    runtime_input.data.dtype,
                    runtime_input.data.shape.size,
                    TIMESEED);

        network.prepare();
        network.run({ &runtime_input }, { &runtime_output });

        if (&l1.input(0) != &graph.inputBoundary()->output(0)) {
            std::cerr << "check failed: layer input not rewired inside graph lifetime" << std::endl;
            return 11;
        }
        if (!(runtime_output.data.flags & PARAM_OUTPUT)) {
            std::cerr << "check failed: runtime_output missing PARAM_OUTPUT" << std::endl;
            return 12;
        }
        if (!(runtime_output.data.flags & PARAM_OWN_DATA)) {
            std::cerr << "check failed: runtime_output missing PARAM_OWN_DATA" << std::endl;
            return 13;
        }

        output_ptr_in_scope = runtime_output.data.ptr;
        output_size_in_scope = runtime_output.data.shape.size;
        output_flags_in_scope = runtime_output.data.flags;
    }

    if (&l1.input(0) != &graph_input) {
        std::cerr << "check failed: layer input not restored after graph destruction" << std::endl;
        return 21;
    }
    if (nullptr == runtime_output.data.ptr) {
        std::cerr << "check failed: runtime_output ptr lost after graph destruction" << std::endl;
        return 22;
    }
    if (output_ptr_in_scope != runtime_output.data.ptr) {
        std::cerr << "check failed: runtime_output ptr changed after graph destruction" << std::endl;
        return 23;
    }
    if (output_size_in_scope != runtime_output.data.shape.size) {
        std::cerr << "check failed: runtime_output shape changed after graph destruction" << std::endl;
        return 24;
    }
    if (output_flags_in_scope != runtime_output.data.flags) {
        std::cerr << "check failed: runtime_output flags changed after graph destruction" << std::endl;
        return 25;
    }
    if (all_zero(runtime_output)) {
        std::cerr << "check failed: runtime_output should not remain zero after real execution" << std::endl;
        return 26;
    }

    std::cout << "graph lifecycle test ok" << std::endl;
    return 0;
}
