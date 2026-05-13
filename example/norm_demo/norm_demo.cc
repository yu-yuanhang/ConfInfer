#include <iostream>

#include <core/Network.h>
#include <ops.h>
#include <trustinfer.h>

static void print_shape_line(const char* name, const Kernel::core::DataShape_t& shape) {
    std::cout << name << " ndim=" << shape.ndim << " dims=[";
    for (Kernel::UINT i = 0; i < shape.ndim; ++i) {
        if (i) std::cout << ", ";
        std::cout << shape.dims[i];
    }
    std::cout << "] size=" << shape.size << std::endl;
}

int main() {
    using namespace Kernel::core;

    RUNTIME->setThreadsNum(1);

    Value_t graph_input({2, 3, 4, 4});

    BatchNorm2d bn(3);
    GroupNorm gn(3, 3);
    LayerNorm ln({4, 4});

    Layer& l1 = bn(graph_input);
    Layer& l2 = gn(l1.output());
    Layer& l3 = ln(l2.output());

    Graph graph(
        { GraphInputSlot("input", graph_input) },
        { GraphOutputSlot("output", l3.output()) }
    );
    Network network(graph, RUNTIME);

    Value_t runtime_input({2, 3, 4, 4});
    runtime_input.alloc();
    fill_random(runtime_input.data.ptr,
                runtime_input.data.dtype,
                runtime_input.data.shape.size,
                TIMESEED);

    Value_t runtime_output;

    network.prepare();
    network.run({ &runtime_input }, { &runtime_output });

    print_shape_line("input", runtime_input.data.shape);
    print_shape_line("bn_out", l1.output().data.shape);
    print_shape_line("gn_out", l2.output().data.shape);
    print_shape_line("ln_out", l3.output().data.shape);
    print_shape_line("network_output", runtime_output.data.shape);

    if (runtime_output.data.ptr == nullptr) {
        std::cerr << "runtime_output ptr is nullptr" << std::endl;
        return 1;
    }
    if (!(runtime_output.data.flags & PARAM_OUTPUT)) {
        std::cerr << "runtime_output missing PARAM_OUTPUT" << std::endl;
        return 1;
    }
    if (!(runtime_output.data.flags & PARAM_OWN_DATA)) {
        std::cerr << "runtime_output missing PARAM_OWN_DATA" << std::endl;
        return 1;
    }

    std::cout << "norm demo ok" << std::endl;
    return 0;
}
