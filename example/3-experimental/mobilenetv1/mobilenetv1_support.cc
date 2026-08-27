#include "mobilenetv1_support.h"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>

#include <core/model_param_loader.h>

namespace mobilenetv1_demo {

using namespace Kernel;
using namespace Kernel::core;

namespace {

std::string default_manifest_path() {
    const std::filesystem::path source_dir = std::filesystem::path(__FILE__).parent_path();
    return (source_dir / "export" / "params_best" / "params.json").string();
}

std::string default_dataset_root() {
    const std::filesystem::path source_dir = std::filesystem::path(__FILE__).parent_path();
    return (source_dir.parent_path() / "dataset" / "cifar100" / "cifar-100-python").string();
}

bool has_flag(int argc, char* argv[], const std::string& flag) {
    for (int i = 1; i < argc; ++i) {
        if (flag == argv[i]) {
            return true;
        }
    }
    return false;
}

size_t resolve_sample_index(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--sample-index=", 15) == 0) {
            return static_cast<size_t>(std::stoul(argv[i] + 15));
        }
    }
    return 0;
}

BackendKind resolve_backend_kind(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--backend=", 10) != 0) {
            continue;
        }

        const char* backend = argv[i] + 10;
        if (std::strcmp(backend, "cpu_ree") == 0) {
            return BackendKind::BK_CPU_REE;
        }
        if (std::strcmp(backend, "cpu_ree_ref") == 0) {
            return BackendKind::BK_CPU_REE_REF;
        }
        if (std::strcmp(backend, "cpu_tee") == 0) {
            return BackendKind::BK_CPU_TEE;
        }
        EXIT_ERROR("unsupported backend: %s", backend);
    }
    return BackendKind::BK_CPU_REE;
}

size_t resolve_max_samples(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--max-samples=", 14) == 0) {
            return static_cast<size_t>(std::stoul(argv[i] + 14));
        }
    }
    return 0;
}

std::string resolve_manifest_path(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--manifest=", 11) == 0) {
            return std::string(argv[i] + 11);
        }
    }
    return default_manifest_path();
}

std::string resolve_dataset_root(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--data-root=", 12) == 0) {
            return std::string(argv[i] + 12);
        }
    }
    return default_dataset_root();
}

CIFAR100Dataset::Split resolve_dataset_split(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--train") == 0) {
            return CIFAR100Dataset::Split::TRAIN;
        }
    }
    return CIFAR100Dataset::Split::TEST;
}

} // namespace

MobileNetRunOptions parse_options(int argc, char* argv[]) {
    MobileNetRunOptions opts{};
    // 默认值: example/3-experimental/mobilenetv1/export/params_best/params.json
    opts.manifest_path = resolve_manifest_path(argc, argv);
    // 默认值: example/dataset/cifar100/cifar-100-python
    opts.dataset_root = resolve_dataset_root(argc, argv);
    // 默认值: BK_CPU_REE
    opts.backend_kind = resolve_backend_kind(argc, argv);
    // 默认值: false
    opts.load_only = has_flag(argc, argv, "--load-only");
    // 默认值: false
    opts.dataset_only = has_flag(argc, argv, "--dataset-only");
    // 默认值: false
    opts.eval_only = has_flag(argc, argv, "--eval");
    // 默认值: false
    opts.print_logits_only = has_flag(argc, argv, "--print-logits");
    // 默认值: false
    opts.print_partition_only = has_flag(argc, argv, "--print-partitions");
    // 默认值: false
    opts.partition_only = has_flag(argc, argv, "--partition-only");
    // 默认值: CIFAR100Dataset::Split::TEST
    opts.dataset_split = resolve_dataset_split(argc, argv);
    // 默认值: 0, 表示不额外限制样本数
    opts.max_samples = resolve_max_samples(argc, argv);
    // 默认值: 0
    opts.sample_index = resolve_sample_index(argc, argv);
    return opts;
}

const char* backend_kind_name(BackendKind kind) {
    switch (kind) {
    case BackendKind::BK_CPU_REE:
        return "cpu_ree";
    case BackendKind::BK_CPU_REE_REF:
        return "cpu_ree_ref";
    case BackendKind::BK_CPU_REE_ALT0:
        return "cpu_ree_alt0";
    case BackendKind::BK_CPU_REE_ALT1:
        return "cpu_ree_alt1";
    case BackendKind::BK_CPU_TEE:
        return "cpu_tee";
    default:
        return "unknown";
    }
}

const char* exec_domain_name(ExecutionDomain domain) {
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

void expect_shape(const Value_t& value,
                  std::initializer_list<UINT> dims,
                  const char* name) {
    EXIT_ERROR_CHECK_NE(value.data.shape.ndim, dims.size(), "%s ndim mismatch", name);
    UINT idx = 0;
    for (UINT dim : dims) {
        EXIT_ERROR_CHECK_NE(value.data.shape.dims[idx], dim, "%s dim mismatch", name);
        ++idx;
    }
}

void fill_sequential_fp32(Value_t& value) {
    if (nullptr == value.data.ptr) {
        value.alloc();
    }
    FLOAT* ptr = static_cast<FLOAT*>(value.data.ptr);
    for (UINT i = 0; i < value.data.shape.size; ++i) {
        const INT centered = static_cast<INT>(i % 23) - 11;
        ptr[i] = static_cast<FLOAT>(centered) / 11.0f;
    }
}

bool is_all_zero(const Value_t& value, FLOAT eps) {
    EXIT_ERROR_CHECK_EQ(nullptr, value.data.ptr, "value ptr is nullptr");
    const FLOAT* ptr = static_cast<const FLOAT*>(value.data.ptr);
    for (UINT i = 0; i < value.data.shape.size; ++i) {
        if (std::fabs(ptr[i]) > eps) {
            return false;
        }
    }
    return true;
}

UINT top1_index(const Value_t& logits) {
    EXIT_ERROR_CHECK_EQ(nullptr, logits.data.ptr, "logits ptr is nullptr");
    EXIT_ERROR_CHECK_NE(logits.data.shape.ndim, 2, "logits ndim must be 2");
    EXIT_ERROR_CHECK_NE(logits.data.shape.dims[0], 1, "logits batch size must be 1");

    const UINT class_count = logits.data.shape.dims[1];
    const FLOAT* ptr = static_cast<const FLOAT*>(logits.data.ptr);
    UINT best_index = 0;
    FLOAT best_value = ptr[0];
    for (UINT i = 1; i < class_count; ++i) {
        if (ptr[i] > best_value) {
            best_value = ptr[i];
            best_index = i;
        }
    }
    return best_index;
}

bool in_topk(const Value_t& logits, UINT label, UINT k) {
    EXIT_ERROR_CHECK_EQ(nullptr, logits.data.ptr, "logits ptr is nullptr");
    EXIT_ERROR_CHECK_NE(logits.data.shape.ndim, 2, "logits ndim must be 2");
    EXIT_ERROR_CHECK_NE(logits.data.shape.dims[0], 1, "logits batch size must be 1");

    const UINT class_count = logits.data.shape.dims[1];
    if (k > class_count) {
        k = class_count;
    }

    const FLOAT* ptr = static_cast<const FLOAT*>(logits.data.ptr);
    std::vector<UINT> top_indices;
    top_indices.reserve(k);

    for (UINT i = 0; i < class_count; ++i) {
        UINT insert_pos = 0;
        while (insert_pos < top_indices.size() && ptr[top_indices[insert_pos]] >= ptr[i]) {
            ++insert_pos;
        }
        if (insert_pos < k) {
            top_indices.insert(top_indices.begin() + static_cast<std::ptrdiff_t>(insert_pos), i);
            if (top_indices.size() > k) {
                top_indices.pop_back();
            }
        }
    }

    for (UINT index : top_indices) {
        if (index == label) {
            return true;
        }
    }
    return false;
}

void print_logits(const Value_t& logits) {
    EXIT_ERROR_CHECK_EQ(nullptr, logits.data.ptr, "logits ptr is nullptr");
    EXIT_ERROR_CHECK_NE(logits.data.shape.ndim, 2, "logits ndim must be 2");
    EXIT_ERROR_CHECK_NE(logits.data.shape.dims[0], 1, "logits batch size must be 1");

    const UINT class_count = logits.data.shape.dims[1];
    const FLOAT* ptr = static_cast<const FLOAT*>(logits.data.ptr);
    const std::streamsize old_precision = std::cout.precision();
    std::cout << "logits: [";
    std::cout << std::setprecision(std::numeric_limits<FLOAT>::max_digits10);
    for (UINT i = 0; i < class_count; ++i) {
        std::cout << ptr[i];
        if (i + 1 != class_count) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
    std::cout << std::setprecision(old_precision);
}

void verify_model_structure(const MobileNetV1Model& model,
                            const ParamBindingTable& bindings) {
    const std::vector<MobileNetOpInfo>& ops = model.ops();

    EXIT_ERROR_CHECK_NE(67, ops.size(), "MobileNetV1 op count mismatch");
    EXIT_ERROR_CHECK_EQ(0, bindings.size(), "MobileNetV1 param binding table is empty");

    expect_shape(ops[2].layer->output(), {1, 32, 32, 32}, "stem");
    expect_shape(ops[8].layer->output(), {1, 64, 32, 32}, "stage1");
    expect_shape(ops[20].layer->output(), {1, 128, 16, 16}, "stage2");
    expect_shape(ops[38].layer->output(), {1, 256, 8, 8}, "stage3");
    expect_shape(ops[62].layer->output(), {1, 512, 4, 4}, "stage4");
    expect_shape(ops[63].layer->output(), {1, 512, 1, 1}, "avgpool");
    expect_shape(ops[64].layer->output(), {1, 512}, "flatten");
    expect_shape(ops[65].layer->output(), {1, 512}, "dropout");
    expect_shape(ops[66].layer->output(), {1, 100}, "classifier");
}

ParamLoadReport load_model_params_or_die(const std::string& manifest_path,
                                         const ParamBindingTable& bindings) {
    ParamLoadOptions load_options;
    const ParamLoadReport load_report = loadModelParams(manifest_path, bindings, load_options);
    EXIT_ERROR_CHECK_NE(static_cast<UINT>(bindings.size()), load_report.loaded_param_count,
        "Loaded param count mismatch");
    EXIT_ERROR_CHECK_EQ(0, load_report.loaded_bytes, "Loaded bytes must be > 0");
    return load_report;
}

void print_op_summary(const std::vector<MobileNetOpInfo>& ops) {
    std::cout << "================================================================================\n";
    std::cout << "idx  name                    type                output_shape\n";
    std::cout << "================================================================================\n";
    for (UINT i = 0; i < ops.size(); ++i) {
        const DataShape_t& shape = ops[i].layer->output().data.shape;
        std::cout << i << "    " << ops[i].name << "    " << ops[i].type << "    [";
        for (UINT d = 0; d < shape.ndim; ++d) {
            std::cout << shape.dims[d];
            if (d + 1 != shape.ndim) {
                std::cout << ",";
            }
        }
        std::cout << "]\n";
    }
    std::cout << "================================================================================\n";
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

    std::cout << "[part-graph] nodes: " << part_graph.size()
              << " edges: " << part_graph.edges().size() << std::endl;
    for (const PartitionEdge& edge : part_graph.edges()) {
        std::cout << "  edge " << edge.from << " -> " << edge.to
                  << " values=" << edge.values.size() << std::endl;
    }
}

void validate_partition_pipeline(const Network& network) {
    const PartitionGraph& part_graph = network.partGraph();
    const std::vector<ExecPartition>& parts = part_graph.parts();

    EXIT_ERROR_CHECK_EQ(true, parts.empty(), "ExecPartition list is empty");
    EXIT_ERROR_CHECK_NE(part_graph.size(), parts.size(), "PartitionGraph node size mismatch");

    UINT total_part_layers = 0;
    for (const ExecPartition& part : parts) {
        EXIT_ERROR_CHECK_EQ(true, part.empty(), "ExecPartition must not be empty");
        EXIT_ERROR_CHECK_NE(part.topo().size(), part.layers().size(),
            "ExecPartition topo size mismatch");
        total_part_layers += static_cast<UINT>(part.topo().size());
    }

    UINT tee_partitions = 0;
    for (const ExecPartition& part : parts) {
        if (ExecutionDomain::ED_CPU_TEE == part.domain()) {
            ++tee_partitions;
        }
    }

    EXIT_ERROR_CHECK_EQ(true, total_part_layers == 0,
        "ExecPartition total layer coverage mismatch");
    EXIT_ERROR_CHECK_EQ(true, tee_partitions == 0,
        "Expected at least one TEE partition");
}

} // namespace mobilenetv1_demo
