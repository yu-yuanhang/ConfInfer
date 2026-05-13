#include <iostream>

#include <core/Network.h>
#include <pool.h>
#include <trustinfer.h>
#include <ops.h>

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    RUNTIME->setThreadsNum(1);

    Conv2d shared_conv(3, 3, {3, 3}, {1, 1}, {1, 1, 1, 1});
    MaxPool2d pool1({2, 2});

    Value_t graph_input({1, 3, 8, 8});

    std::cout << "build layer chain..." << std::endl;
    Layer &l1 = shared_conv(graph_input);
    Layer &l2 = pool1(l1.output());
    Layer &l3 = shared_conv(l2.output());

    std::cout << "build graph..." << std::endl;
    Graph graph(
        { GraphInputSlot("input", graph_input) },
        { GraphOutputSlot("output", l3.output()) }
    );
    std::cout << "build network..." << std::endl;
    Network network(graph, RUNTIME);

    Value_t runtime_input({1, 3, 8, 8});
    runtime_input.alloc();
    fill_random(runtime_input.data.ptr, runtime_input.data.dtype,
                runtime_input.data.shape.size, TIMESEED);

    Value_t runtime_output;

    print_shape("input", runtime_input.data.shape);
    print_shape("shared_conv_1_out", l1.output().data.shape);
    print_shape("pool1_out", l2.output().data.shape);
    print_shape("shared_conv_2_out", l3.output().data.shape);

    std::cout << "graph layers: " << graph._layersNum << std::endl;
    std::cout << "exec order size: " << graph._execOrder.size() << std::endl;
    const bool relation_ok =
        print_layer_relation("shared_conv_1", l1, "shared_conv_2", l3);
    if (!relation_ok) {
        std::cerr << "OpSignature/Layer relation check failed" << std::endl;
        return 1;
    }

    std::cout << "prepare network..." << std::endl;
    network.prepare();
    std::cout << "run network..." << std::endl;
    network.run({ &runtime_input }, { &runtime_output });

    print_shape("network_output", runtime_output.data.shape);

    const bool conv1_zero = print_zero_check("shared_conv_1_out", l1.output());
    const bool pool_zero = print_zero_check("pool1_out", l2.output());
    const bool conv2_zero = print_zero_check("shared_conv_2_out", l3.output());
    if (conv1_zero) {
        std::cerr << "conv1 output should not remain zero after Conv2d execution" << std::endl;
        return 1;
    }
    if (pool_zero) {
        std::cerr << "pool output should not remain zero after MaxPool2d execution" << std::endl;
        return 1;
    }
    if (conv2_zero) {
        std::cerr << "conv2 output should not remain zero after Conv2d execution" << std::endl;
        return 1;
    }

    std::cout << "demo run ok" << std::endl;

    return 0;
}
