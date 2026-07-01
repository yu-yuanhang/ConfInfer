#include <iostream>

#include <core/Network.h>
#include <core/cifar100_dataset.h>
#include <trustinfer.h>

#include "mobilenetv1_model.h"
#include "mobilenetv1_support.h"

using namespace Kernel;
using namespace Kernel::core;
using namespace mobilenetv1_demo;

namespace mobilenetv1_demo {

using namespace Kernel;
using namespace Kernel::core;

void print_dataset_sample(const MobileNetRunOptions& opts) {
    CIFAR100Dataset dataset(opts.dataset_root, opts.dataset_split, true);
    Value_t sample_input({1, 3, 32, 32});
    dataset.loadInput(0, sample_input);
    const FLOAT* ptr = static_cast<const FLOAT*>(sample_input.data.ptr);
    EXIT_ERROR_CHECK_EQ(nullptr, ptr, "dataset sample ptr is nullptr");

    std::cout << "dataset load ok" << std::endl;
    std::cout << "data_root: " << opts.dataset_root << std::endl;
    std::cout << "split: "
              << (opts.dataset_split == CIFAR100Dataset::Split::TRAIN ? "train" : "test")
              << std::endl;
    std::cout << "sample_count: " << dataset.size() << std::endl;
    std::cout << "sample0_filename: " << dataset.filename(0) << std::endl;
    std::cout << "sample0_fine_label: " << dataset.fineLabel(0) << std::endl;
    std::cout << "sample0_coarse_label: " << dataset.coarseLabel(0) << std::endl;
    std::cout << "sample0_value0: " << ptr[0] << std::endl;
    std::cout << "sample0_value1: " << ptr[1] << std::endl;
    std::cout << "sample0_value2: " << ptr[2] << std::endl;
}

void print_param_load_report(const MobileNetRunOptions& opts,
                             const ParamLoadReport& load_report) {
    std::cout << "param load ok" << std::endl;
    std::cout << "manifest: " << opts.manifest_path << std::endl;
    std::cout << "backend: " << backend_kind_name(opts.backend_kind) << std::endl;
    std::cout << "loaded_param_count: " << load_report.loaded_param_count << std::endl;
    std::cout << "skipped_manifest_entry_count: "
              << load_report.skipped_manifest_entry_count << std::endl;
    std::cout << "loaded_bytes: " << load_report.loaded_bytes << std::endl;
}

void run_partition_stage(const MobileNetRunOptions& opts, Network& network) {
    network.prepare();
    validate_partition_pipeline(network);
    if (opts.print_partition_only || opts.partition_only) {
        print_partition_summary(network);
    }
}

void run_logits_mode(const MobileNetRunOptions& opts, Network& network) {
    CIFAR100Dataset dataset(opts.dataset_root, opts.dataset_split, true);
    EXIT_ERROR_CHECK_EQ(false, opts.sample_index < dataset.size(), "sample_index out of range");

    Value_t runtime_input({1, 3, 32, 32});
    Value_t runtime_output;
    dataset.loadInput(opts.sample_index, runtime_input);
    network.run({ &runtime_input }, { &runtime_output });

    std::cout << "logits ok" << std::endl;
    std::cout << "backend: " << backend_kind_name(opts.backend_kind) << std::endl;
    std::cout << "split: "
              << (opts.dataset_split == CIFAR100Dataset::Split::TRAIN ? "train" : "test")
              << std::endl;
    std::cout << "sample_index: " << opts.sample_index << std::endl;
    std::cout << "label: " << dataset.fineLabel(opts.sample_index) << std::endl;
    print_logits(runtime_output);
}

void run_eval_mode(const MobileNetRunOptions& opts, Network& network) {
    CIFAR100Dataset dataset(opts.dataset_root, opts.dataset_split, true);
    const size_t total_samples = dataset.size();
    const size_t eval_samples =
        (0 == opts.max_samples || opts.max_samples > total_samples) ? total_samples : opts.max_samples;
    Value_t runtime_input({1, 3, 32, 32});
    Value_t runtime_output;
    size_t correct_top1 = 0;
    size_t correct_top5 = 0;

    for (size_t sample_index = 0; sample_index < eval_samples; ++sample_index) {
        dataset.loadInput(sample_index, runtime_input);
        network.run({ &runtime_input }, { &runtime_output });

        const UINT label = dataset.fineLabel(sample_index);
        if (top1_index(runtime_output) == label) {
            ++correct_top1;
        }
        if (in_topk(runtime_output, label, 5)) {
            ++correct_top5;
        }
    }

    const double top1 = 100.0 * static_cast<double>(correct_top1) / static_cast<double>(eval_samples);
    const double top5 = 100.0 * static_cast<double>(correct_top5) / static_cast<double>(eval_samples);

    std::cout << "eval ok" << std::endl;
    std::cout << "data_root: " << opts.dataset_root << std::endl;
    std::cout << "backend: " << backend_kind_name(opts.backend_kind) << std::endl;
    std::cout << "split: "
              << (opts.dataset_split == CIFAR100Dataset::Split::TRAIN ? "train" : "test")
              << std::endl;
    std::cout << "samples: " << eval_samples << std::endl;
    std::cout << "top1: " << top1 << "%" << std::endl;
    std::cout << "top5: " << top5 << "%" << std::endl;
}

void run_closed_loop(Network& network, const std::vector<MobileNetOpInfo>& ops) {
    print_op_summary(ops);

    Value_t runtime_input({1, 3, 32, 32});
    fill_sequential_fp32(runtime_input);
    Value_t runtime_output;
    network.run({ &runtime_input }, { &runtime_output });

    expect_shape(runtime_output, {1, 100}, "network_output");
    if (is_all_zero(runtime_output)) {
        std::cerr << "mobilenetv1 output should not be all zero" << std::endl;
        std::exit(1);
    }

    std::cout << "mobilenetv1 closed-loop test ok" << std::endl;
}

} // namespace mobilenetv1_demo

int main(int argc, char *argv[]) {

    const MobileNetRunOptions opts = parse_options(argc, argv);

    if (!opts.load_only && !opts.dataset_only) {
        RUNTIME->setThreadsNum(1);
        std::cout << "[stage] runtime ready" << std::endl;
    }

    EXECUTOR->setBackendKind(opts.backend_kind);

    if (opts.dataset_only) {
        print_dataset_sample(opts);
        return 0;
    }

    // Stage 1: build model structure only.
    MobileNetV1Model model;
    const std::vector<MobileNetOpInfo>& ops = model.ops();
    const ParamBindingTable bindings = model.build_param_bindings();
    std::cout << "[stage] model built" << std::endl;

    // Stage 2: instantiate graph and runtime network from the model.
    Graph graph(
        { GraphInputSlot("input", model.graph_input()) },
        { GraphOutputSlot("output", model.output_layer().output()) }
    );
    Network network(graph, RUNTIME);
    std::cout << "[stage] graph built" << std::endl;
    std::cout << "[stage] network created" << std::endl;

    // Stage 3: validate graph-visible structure and load weights.
    verify_model_structure(model, bindings);
    std::cout << "[stage] shape checks ok" << std::endl;
    std::cout << "[stage] start param load" << std::endl;
    const ParamLoadReport load_report = load_model_params_or_die(opts.manifest_path, bindings);
    print_param_load_report(opts, load_report);

    if (opts.load_only) {
        return 0;
    }

    // Stage 4: prepare runtime execution views, including partitions.
    run_partition_stage(opts, network);
    if (opts.partition_only) {
        std::cout << "partition validation ok" << std::endl;
        return 0;
    }

    if (opts.print_logits_only) {
        run_logits_mode(opts, network);
        return 0;
    }

    if (opts.eval_only) {
        run_eval_mode(opts, network);
        return 0;
    }

    // Stage 5: local closed-loop execution.
    run_closed_loop(network, ops);
    return 0;
}
