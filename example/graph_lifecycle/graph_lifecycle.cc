#include <iostream>
#include <vector>
#include <cstring>

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

static bool has_flag(int argc, char* argv[], const char* flag) {
    for (int i = 1; i < argc; ++i) {
        if (0 == std::strcmp(argv[i], flag)) {
            return true;
        }
    }
    return false;
}

static const char* exec_domain_name(ExecutionDomain domain) {
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

static void print_partition_summary(const Network& network) {
    const PartitionGraph& part_graph = network.partGraph();
    const std::vector<ExecPartition>& parts = part_graph.parts();

    std::cout << "[partition] count=" << parts.size() << std::endl;
    for (const ExecPartition& part : parts) {
        std::cout << "  part[" << part.id() << "]"
                  << " domain=" << exec_domain_name(part.domain())
                  << " layers=" << part.layers().size()
                  << " inputs=" << part.inputs().size()
                  << " outputs=" << part.outputs().size()
                  << " internals=" << part.internals().size()
                  << std::endl;
    }

    std::cout << "[part-graph] nodes=" << part_graph.size()
              << " edges=" << part_graph.edges().size() << std::endl;
    for (const PartitionEdge& edge : part_graph.edges()) {
        std::cout << "  edge " << edge.from << " -> " << edge.to
                  << " values=" << edge.values.size() << std::endl;
    }
}

static void validate_partition_pipeline(const Network& network) {
    const PartitionGraph& part_graph = network.partGraph();
    const std::vector<ExecPartition>& parts = part_graph.parts();

    EXIT_ERROR_CHECK_EQ(true, parts.empty(), "ExecPartition list is empty");
    EXIT_ERROR_CHECK_NE(part_graph.size(), parts.size(), "PartitionGraph node size mismatch");

    UINT tee_parts = 0;
    UINT total_part_layers = 0;

    for (const ExecPartition& part : parts) {
        EXIT_ERROR_CHECK_EQ(true, part.empty(), "ExecPartition must not be empty");
        EXIT_ERROR_CHECK_NE(part.topo().size(), part.layers().size(),
            "ExecPartition topo size mismatch");
        total_part_layers += static_cast<UINT>(part.topo().size());
        if (ExecutionDomain::ED_CPU_TEE == part.domain()) {
            ++tee_parts;
        }
    }
    EXIT_ERROR_CHECK_EQ(true, tee_parts == 0, "Expected at least one TEE partition");
    EXIT_ERROR_CHECK_EQ(true, total_part_layers == 0,
        "ExecPartition total layer coverage mismatch");
}

int main(int argc, char *argv[]) {
    const bool partition_only = has_flag(argc, argv, "--partition-only");

    Conv2d conv(3, 3, {3, 3}, {1, 1}, {1, 1, 1, 1});
    MaxPool2d pool({2, 2});

    Value_t graph_input({1, 3, 8, 8});
    Layer& l1 = conv(graph_input);
    Layer& l2 = pool(l1.output());
    Layer& l3 = conv(l2.output());
    l2.requireTEE(true);

    Value_t runtime_output;
    void* output_ptr_in_scope = nullptr;
    uint32_t output_size_in_scope = 0;
    uint32_t output_flags_in_scope = 0;

    {
        Graph graph(
            { GraphInputSlot("input", graph_input) },
            { GraphOutputSlot("output", l3.output()) }
        );
        Network network(graph);

        Value_t runtime_input({1, 3, 8, 8});
        runtime_input.alloc();
        fill_random(runtime_input.data.ptr,
                    runtime_input.data.dtype,
                    runtime_input.data.shape.size,
                    TIMESEED);

        network.prepare();
        validate_partition_pipeline(network);
        if (partition_only) {
            print_partition_summary(network);
            std::cout << "partition validation ok" << std::endl;
            return 0;
        }
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
