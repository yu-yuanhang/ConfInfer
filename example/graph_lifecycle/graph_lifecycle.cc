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

static const char* exec_unit_type_name(ExecUnitType type) {
    switch (type) {
    case ExecUnitType::EU_LAYER:
        return "layer";
    case ExecUnitType::EU_PARTITION:
        return "partition";
    default:
        return "unknown";
    }
}

static void print_partition_summary(const Network& network) {
    const std::vector<ExecutionPartition>& parts = network.execPartitions();
    const PartitionGraph& part_graph = network.partGraph();
    const ExecutionPlan& plan = network.execPlan();

    std::cout << "[partition] count=" << parts.size() << std::endl;
    for (const ExecutionPartition& part : parts) {
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

    std::cout << "[exec-plan] units=" << plan.size() << std::endl;
    UINT idx = 0;
    for (const ExecUnit& unit : plan.units()) {
        const ExecutionPartition *part = unit.part();
        std::cout << "  unit[" << idx++ << "]"
                  << " type=" << exec_unit_type_name(unit.type())
                  << " domain=" << exec_domain_name(unit.domain())
                  << " slices=" << unit.slices().size();
        if (part) {
            std::cout << " part=" << part->id();
        }
        std::cout << std::endl;
    }
}

static void validate_partition_pipeline(const Network& network) {
    const std::vector<ExecutionPartition>& parts = network.execPartitions();
    const PartitionGraph& part_graph = network.partGraph();
    const ExecutionPlan& plan = network.execPlan();

    EXIT_ERROR_CHECK_EQ(true, parts.empty(), "ExecutionPartition list is empty");
    EXIT_ERROR_CHECK_NE(part_graph.size(), parts.size(), "PartitionGraph node size mismatch");
    EXIT_ERROR_CHECK_EQ(true, plan.empty(), "ExecutionPlan is empty");

    UINT tee_parts = 0;
    UINT tee_units = 0;
    UINT total_part_layers = 0;
    UINT total_unit_slices = 0;

    for (const ExecutionPartition& part : parts) {
        EXIT_ERROR_CHECK_EQ(true, part.empty(), "ExecutionPartition must not be empty");
        EXIT_ERROR_CHECK_NE(part.topo().size(), part.layers().size(),
            "ExecutionPartition topo size mismatch");
        total_part_layers += static_cast<UINT>(part.topo().size());
        if (ExecutionDomain::ED_CPU_TEE == part.domain()) {
            ++tee_parts;
        }
    }

    for (const ExecUnit& unit : plan.units()) {
        EXIT_ERROR_CHECK_EQ(nullptr, unit.part(), "ExecUnit partition is nullptr");
        EXIT_ERROR_CHECK_EQ(true, unit.slices().empty(), "ExecUnit slices must not be empty");
        total_unit_slices += static_cast<UINT>(unit.slices().size());
        if (ExecutionDomain::ED_CPU_TEE == unit.domain()) {
            EXIT_ERROR_CHECK_NE(unit.type(), ExecUnitType::EU_PARTITION,
                "TEE ExecUnit must be partition type");
            ++tee_units;
        }
    }

    EXIT_ERROR_CHECK_NE(total_part_layers, total_unit_slices,
        "ExecutionPlan total slice coverage mismatch");
    EXIT_ERROR_CHECK_EQ(true, tee_parts == 0, "Expected at least one TEE partition");
    EXIT_ERROR_CHECK_NE(tee_parts, tee_units,
        "TEE partition count and TEE ExecUnit count mismatch");
}

int main(int argc, char *argv[]) {
    RUNTIME->setThreadsNum(1);
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
        Network network(graph, RUNTIME);

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
