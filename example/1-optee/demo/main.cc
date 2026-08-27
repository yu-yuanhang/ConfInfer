#include <iostream>
#include <vector>

#include <core/Network.h>
#include <ops.h>
#include <trustinfer.h>

using namespace Kernel;
using namespace Kernel::core;

namespace {

const char *exec_domain_name(ExecutionDomain domain) {
    switch (domain) {
    case ExecutionDomain::ED_CPU_REE:
        return "cpu_ree";
    case ExecutionDomain::ED_CPU_TEE:
        return "cpu_tee";
    case ExecutionDomain::ED_DEFAULT:
    default:
        return "default";
    }
}

bool is_boundary_layer(const Layer *layer) {
    if (nullptr == layer) {
        return false;
    }
    return layer->type() == LayerType::GRAPH_INPUT ||
           layer->type() == LayerType::GRAPH_OUTPUT;
}

bool has_compute_layer(const ExecPartition& part) {
    for (Layer *layer : part.layers()) {
        if (!is_boundary_layer(layer)) {
            return true;
        }
    }
    return false;
}

void print_partition_summary(const Network& network) {
    const PartitionGraph& part_graph = network.partGraph();
    const std::vector<ExecPartition>& parts = part_graph.parts();

    std::cout << "[partition] count: " << parts.size() << std::endl;
    for (const ExecPartition& part : parts) {
        std::cout << "  part[" << part.id() << "]"
                  << " domain=" << exec_domain_name(part.domain())
                  << " layers=" << part.layers().size()
                  << " inputs=" << part.inputs().size()
                  << " outputs=" << part.outputs().size()
                  << " internals=" << part.internals().size()
                  << std::endl;
    }
}

void validate_plan(const Network& network) {
    const std::vector<ExecPartition>& parts = network.partGraph().parts();

    EXIT_ERROR_CHECK_EQ(true, parts.empty(), "ExecPartition list is empty");
    EXIT_ERROR_CHECK_EQ(false,
                        EXECUTOR->supports(ExecutionDomain::ED_CPU_TEE),
                        "TEE execution backend is not installed");

    UINT tee_parts = 0;
    UINT compute_parts = 0;
    for (const ExecPartition& part : parts) {
        if (has_compute_layer(part)) {
            ++compute_parts;
        }
        if (part.domain() == ExecutionDomain::ED_CPU_TEE) {
            ++tee_parts;
        }
    }

    EXIT_ERROR_CHECK_NE(2u, compute_parts,
                        "Expected exactly 2 compute partitions: REE + TEE");
    EXIT_ERROR_CHECK_NE(1u, tee_parts, "Expected exactly 1 TEE partition");
}

} // namespace

int main() {

    Conv2d conv(1, 2, {3, 3}, {1, 1}, {1, 1}, {1, 1}, 1, true);
    BatchNorm2d bn(2);
    ReLU relu(false);
    AvgPool2d avgpool({2, 2}, {2, 2}, {0, 0}, false, true, 0);
    Flatten flatten(1, -1);
    Linear fc(8, 4, true);
    Softmax softmax(1);

    Value_t graph_input({1, 1, 4, 4});

    Layer& l1 = conv(graph_input);
    Layer& l2 = bn(l1.output());
    Layer& l3 = relu(l2.output());
    Layer& l4 = avgpool(l3.output());
    Layer& l5 = flatten(l4.output());
    Layer& l6 = fc(l5.output());
    Layer& l7 = softmax(l6.output());

    l2.setExecDomain(ExecutionDomain::ED_CPU_TEE);
    l3.setExecDomain(ExecutionDomain::ED_CPU_TEE);
    l4.setExecDomain(ExecutionDomain::ED_CPU_TEE);
    l5.setExecDomain(ExecutionDomain::ED_CPU_TEE);
    l6.setExecDomain(ExecutionDomain::ED_CPU_TEE);
    l7.setExecDomain(ExecutionDomain::ED_CPU_TEE);

    Graph graph(
        { GraphInputSlot("input", graph_input) },
        { GraphOutputSlot("output", l7.output()) }
    );
    Network network(graph);

    Value_t runtime_input({1, 1, 4, 4});
    runtime_input.alloc();
    fill_random(runtime_input.data.ptr,
                runtime_input.data.dtype,
                runtime_input.data.shape.size,
                TIMESEED);

    Value_t runtime_output;

    std::cout << "[demo] prepare network" << std::endl;
    network.prepare();
    validate_plan(network);
    print_partition_summary(network);

    std::cout << "[demo] run network" << std::endl;
    network.run({ &runtime_input }, { &runtime_output });

    EXIT_ERROR_CHECK_EQ(nullptr, runtime_output.data.ptr, "runtime output is nullptr");
    EXIT_ERROR_CHECK_NE(2u, runtime_output.data.shape.ndim, "runtime output ndim mismatch");
    EXIT_ERROR_CHECK_NE(1u, runtime_output.data.shape.dims[0], "runtime output batch mismatch");
    EXIT_ERROR_CHECK_NE(4u, runtime_output.data.shape.dims[1], "runtime output feature mismatch");

    std::cout << "[demo] run ok" << std::endl;
    std::cout << "[demo] TEE bridge communication path executed successfully" << std::endl;
    return 0;
}
