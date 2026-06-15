#include <cmath>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <core/Network.h>
#include <core/cifar100_dataset.h>
#include <core/model_param_loader.h>
#include <ops.h>
#include <trustinfer.h>

using namespace Kernel;
using namespace Kernel::core;

namespace {

UINT make_divisible(FLOAT value, UINT divisor = 8) {
    INT rounded = static_cast<INT>(value + static_cast<FLOAT>(divisor) / 2.0f);
    rounded = (rounded / static_cast<INT>(divisor)) * static_cast<INT>(divisor);
    if (rounded < static_cast<INT>(divisor)) {
        rounded = static_cast<INT>(divisor);
    }
    return static_cast<UINT>(rounded);
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

bool is_all_zero(const Value_t& value, FLOAT eps = 1e-7f) {
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
            return BackendKind::CPU_REE;
        }
        if (std::strcmp(backend, "cpu_ree_ref") == 0) {
            return BackendKind::CPU_REE_REF;
        }
        if (std::strcmp(backend, "cpu_tee") == 0) {
            return BackendKind::CPU_TEE;
        }
        EXIT_ERROR("unsupported backend: %s", backend);
    }
    return BackendKind::CPU_REE;
}

const char* backend_kind_name(BackendKind kind) {
    switch (kind) {
        case BackendKind::CPU_REE:
            return "cpu_ree";
        case BackendKind::CPU_REE_REF:
            return "cpu_ree_ref";
        case BackendKind::CPU_REE_ALT0:
            return "cpu_ree_alt0";
        case BackendKind::CPU_REE_ALT1:
            return "cpu_ree_alt1";
        case BackendKind::CPU_TEE:
            return "cpu_tee";
        default:
            return "unknown";
    }
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

struct MobileNetOpInfo {
    std::string name;
    std::string type;
    Layer* layer;
};

class MobileNetV1Graph {
public:
    MobileNetV1Graph(UINT num_classes = 100,
                     FLOAT width_multiplier = 1.0f,
                     FLOAT dropout_rate = 0.2f)
        : num_classes_(num_classes),
          width_multiplier_(width_multiplier),
          dropout_rate_(dropout_rate),
          graph_input_({1, 3, 32, 32}),
          output_(nullptr) {
        build();
    }

    Value_t& graph_input() { return graph_input_; }
    const std::vector<MobileNetOpInfo>& ops() const { return ops_; }
    Layer& output_layer() const { return *output_; }
    ParamBindingTable build_param_bindings() const {
        ParamBindingTable bindings;

        for (UINT op_index = 0; op_index < ops_.size(); ++op_index) {
            const MobileNetOpInfo& op_info = ops_[op_index];
            const Layer* layer = op_info.layer;
            EXIT_ERROR_CHECK_EQ(nullptr, layer, "MobileNet op layer is nullptr");

            add_param_binding(bindings, *layer, op_index, ParamRole::WEIGHT, "weight");
            add_param_binding(bindings, *layer, op_index, ParamRole::BIAS, "bias");
            add_param_binding(bindings, *layer, op_index, ParamRole::RUNNING_MEAN, "running_mean");
            add_param_binding(bindings, *layer, op_index, ParamRole::RUNNING_VAR, "running_var");
        }

        return bindings;
    }

private:
    static void add_param_binding(ParamBindingTable& bindings,
                                  const Layer& layer,
                                  UINT op_index,
                                  ParamRole role,
                                  const std::string& suffix) {
        const Data_t* param = layer.param(role);
        if (nullptr == param) {
            return;
        }

        const std::string external_name =
            "ops." + std::to_string(op_index) + "." + suffix;
        bindings.add(external_name,
                     const_cast<Data_t*>(param),
                     layer_debug_name(op_index, suffix));
    }

    static std::string layer_debug_name(UINT op_index, const std::string& suffix) {
        return "ops[" + std::to_string(op_index) + "]." + suffix;
    }

    template <typename OpT, typename Fn>
    Layer& add_op(const std::string& name,
                  const std::string& type,
                  std::unique_ptr<OpT> op,
                  Fn&& invoke) {
        OpT* raw = op.get();
        Layer& layer = invoke(*raw);
        owned_ops_.push_back(std::move(op));
        ops_.push_back(MobileNetOpInfo{name, type, &layer});
        return layer;
    }

    Layer& add_conv_bn_act(Value_t& input,
                           UINT in_channels,
                           UINT out_channels,
                           UINT stride,
                           const std::string& block_id) {
        Layer& conv = add_op(
            "conv_" + block_id,
            "conv",
            std::make_unique<Conv2d>(in_channels, out_channels,
                                     std::vector<UINT>{3, 3},
                                     std::vector<UINT>{stride, stride},
                                     std::vector<INT>{1, 1, 1, 1},
                                     std::vector<UINT>{1, 1},
                                     1,
                                     false),
            [&](Conv2d& op) -> Layer& { return op(input); });

        Layer& bn = add_op(
            "bn_" + block_id,
            "bn",
            std::make_unique<BatchNorm2d>(out_channels),
            [&](BatchNorm2d& op) -> Layer& { return op(conv.output()); });

        Layer& act = add_op(
            "act_" + block_id,
            "relu",
            std::make_unique<ReLU>(false),
            [&](ReLU& op) -> Layer& { return op(bn.output()); });

        return act;
    }

    Layer& add_depthwise_separable_conv(Value_t& input,
                                        UINT in_channels,
                                        UINT out_channels,
                                        UINT stride,
                                        const std::string& block_id) {
        Layer& dw_conv = add_op(
            "dw_conv_" + block_id,
            "depthwise_conv",
            std::make_unique<Conv2d>(in_channels, in_channels,
                                     std::vector<UINT>{3, 3},
                                     std::vector<UINT>{stride, stride},
                                     std::vector<INT>{1, 1, 1, 1},
                                     std::vector<UINT>{1, 1},
                                     in_channels,
                                     false),
            [&](Conv2d& op) -> Layer& { return op(input); });

        Layer& dw_bn = add_op(
            "dw_bn_" + block_id,
            "bn",
            std::make_unique<BatchNorm2d>(in_channels),
            [&](BatchNorm2d& op) -> Layer& { return op(dw_conv.output()); });

        Layer& dw_act = add_op(
            "dw_act_" + block_id,
            "relu",
            std::make_unique<ReLU>(false),
            [&](ReLU& op) -> Layer& { return op(dw_bn.output()); });

        Layer& pw_conv = add_op(
            "pw_conv_" + block_id,
            "pointwise_conv",
            std::make_unique<Conv2d>(in_channels, out_channels,
                                     std::vector<UINT>{1, 1},
                                     std::vector<UINT>{1, 1},
                                     std::vector<INT>{0, 0, 0, 0},
                                     std::vector<UINT>{1, 1},
                                     1,
                                     false),
            [&](Conv2d& op) -> Layer& { return op(dw_act.output()); });

        Layer& pw_bn = add_op(
            "pw_bn_" + block_id,
            "bn",
            std::make_unique<BatchNorm2d>(out_channels),
            [&](BatchNorm2d& op) -> Layer& { return op(pw_conv.output()); });

        Layer& pw_act = add_op(
            "pw_act_" + block_id,
            "relu",
            std::make_unique<ReLU>(false),
            [&](ReLU& op) -> Layer& { return op(pw_bn.output()); });

        return pw_act;
    }

    void build() {
        UINT in_channels = 3;

        UINT out_channels = make_divisible(32.0f * width_multiplier_);
        Layer& stem = add_conv_bn_act(graph_input_, in_channels, out_channels, 1, "stem");
        in_channels = out_channels;

        out_channels = make_divisible(64.0f * width_multiplier_);
        Layer& stage1 = add_depthwise_separable_conv(stem.output(), in_channels, out_channels, 1, "1_0");
        in_channels = out_channels;

        out_channels = make_divisible(128.0f * width_multiplier_);
        Layer& stage2_0 = add_depthwise_separable_conv(stage1.output(), in_channels, out_channels, 2, "2_0");
        Layer& stage2_1 = add_depthwise_separable_conv(stage2_0.output(), out_channels, out_channels, 1, "2_1");
        in_channels = out_channels;

        out_channels = make_divisible(256.0f * width_multiplier_);
        Layer& stage3_0 = add_depthwise_separable_conv(stage2_1.output(), in_channels, out_channels, 2, "3_0");
        Layer& stage3_1 = add_depthwise_separable_conv(stage3_0.output(), out_channels, out_channels, 1, "3_1");
        Layer& stage3_2 = add_depthwise_separable_conv(stage3_1.output(), out_channels, out_channels, 1, "3_2");
        in_channels = out_channels;

        out_channels = make_divisible(512.0f * width_multiplier_);
        Layer& stage4_0 = add_depthwise_separable_conv(stage3_2.output(), in_channels, out_channels, 2, "4_0");
        Layer& stage4_1 = add_depthwise_separable_conv(stage4_0.output(), out_channels, out_channels, 1, "4_1");
        Layer& stage4_2 = add_depthwise_separable_conv(stage4_1.output(), out_channels, out_channels, 1, "4_2");
        Layer& stage4_3 = add_depthwise_separable_conv(stage4_2.output(), out_channels, out_channels, 1, "4_3");
        in_channels = out_channels;

        Layer& avgpool = add_op(
            "avgpool",
            "avgpool",
            std::make_unique<AdaptiveAvgPool2d>(std::vector<UINT>{1}),
            [&](AdaptiveAvgPool2d& op) -> Layer& { return op(stage4_3.output()); });

        Layer& flatten = add_op(
            "flatten",
            "flatten",
            std::make_unique<Flatten>(1, -1),
            [&](Flatten& op) -> Layer& { return op(avgpool.output()); });

        Layer& dropout = add_op(
            "dropout",
            "dropout",
            std::make_unique<Dropout>(dropout_rate_, false),
            [&](Dropout& op) -> Layer& { return op(flatten.output()); });

        Layer& classifier = add_op(
            "classifier",
            "linear",
            std::make_unique<Linear>(in_channels, num_classes_, true),
            [&](Linear& op) -> Layer& { return op(dropout.output()); });

        output_ = &classifier;
    }

private:
    UINT num_classes_;
    FLOAT width_multiplier_;
    FLOAT dropout_rate_;
    Value_t graph_input_;
    std::vector<std::unique_ptr<OpSignature>> owned_ops_;
    std::vector<MobileNetOpInfo> ops_;
    Layer* output_;
};

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

} // namespace

int main(int argc, char *argv[]) {
    const std::string manifest_path = resolve_manifest_path(argc, argv);
    const std::string dataset_root = resolve_dataset_root(argc, argv);
    const BackendKind backend_kind = resolve_backend_kind(argc, argv);
    const bool load_only = has_flag(argc, argv, "--load-only");
    const bool dataset_only = has_flag(argc, argv, "--dataset-only");
    const bool eval_only = has_flag(argc, argv, "--eval");
    const bool print_logits_only = has_flag(argc, argv, "--print-logits");
    const CIFAR100Dataset::Split dataset_split = resolve_dataset_split(argc, argv);
    const size_t max_samples = resolve_max_samples(argc, argv);
    const size_t sample_index = resolve_sample_index(argc, argv);

    if (!load_only && !dataset_only) {
        RUNTIME->setThreadsNum(1);
        std::cout << "[stage] runtime ready" << std::endl;
    }

    EXECUTOR->setBackendKind(backend_kind);

    if (dataset_only) {
        CIFAR100Dataset dataset(dataset_root, dataset_split, true);
        Value_t sample_input({1, 3, 32, 32});
        dataset.loadInput(0, sample_input);
        const FLOAT* ptr = static_cast<const FLOAT*>(sample_input.data.ptr);
        EXIT_ERROR_CHECK_EQ(nullptr, ptr, "dataset sample ptr is nullptr");

        std::cout << "dataset load ok" << std::endl;
        std::cout << "data_root: " << dataset_root << std::endl;
        std::cout << "split: "
                  << (dataset_split == CIFAR100Dataset::Split::TRAIN ? "train" : "test")
                  << std::endl;
        std::cout << "sample_count: " << dataset.size() << std::endl;
        std::cout << "sample0_filename: " << dataset.filename(0) << std::endl;
        std::cout << "sample0_fine_label: " << dataset.fineLabel(0) << std::endl;
        std::cout << "sample0_coarse_label: " << dataset.coarseLabel(0) << std::endl;
        std::cout << "sample0_value0: " << ptr[0] << std::endl;
        std::cout << "sample0_value1: " << ptr[1] << std::endl;
        std::cout << "sample0_value2: " << ptr[2] << std::endl;
        return 0;
    }

    MobileNetV1Graph model;
    std::cout << "[stage] model built" << std::endl;
    const std::vector<MobileNetOpInfo>& ops = model.ops();
    const ParamBindingTable bindings = model.build_param_bindings();
    std::cout << "[stage] bindings built" << std::endl;
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
    std::cout << "[stage] shape checks ok" << std::endl;

    ParamLoadOptions load_options;
    std::cout << "[stage] start param load" << std::endl;
    const ParamLoadReport load_report = loadModelParams(manifest_path, bindings, load_options);
    EXIT_ERROR_CHECK_NE(static_cast<UINT>(bindings.size()), load_report.loaded_param_count,
        "Loaded param count mismatch");
    EXIT_ERROR_CHECK_EQ(0, load_report.loaded_bytes, "Loaded bytes must be > 0");

    std::cout << "param load ok" << std::endl;
    std::cout << "manifest: " << manifest_path << std::endl;
    std::cout << "backend: " << backend_kind_name(backend_kind) << std::endl;
    std::cout << "loaded_param_count: " << load_report.loaded_param_count << std::endl;
    std::cout << "skipped_manifest_entry_count: "
              << load_report.skipped_manifest_entry_count << std::endl;
    std::cout << "loaded_bytes: " << load_report.loaded_bytes << std::endl;

    if (load_only) {
        return 0;
    }

    Graph graph(
        { GraphInputSlot("input", model.graph_input()) },
        { GraphOutputSlot("output", model.output_layer().output()) }
    );
    Network network(graph, RUNTIME);
    network.prepare();

// =================================================================================
    // ./mobilenetv1 --print-logits --backend=cpu_ree_ref --sample-index=0
    if (print_logits_only) {
        CIFAR100Dataset dataset(dataset_root, dataset_split, true);
        EXIT_ERROR_CHECK_EQ(false, sample_index < dataset.size(), "sample_index out of range");

        Value_t runtime_input({1, 3, 32, 32});
        Value_t runtime_output;
        dataset.loadInput(sample_index, runtime_input);
        network.run({ &runtime_input }, { &runtime_output });

        std::cout << "logits ok" << std::endl;
        std::cout << "backend: " << backend_kind_name(backend_kind) << std::endl;
        std::cout << "split: "
                  << (dataset_split == CIFAR100Dataset::Split::TRAIN ? "train" : "test")
                  << std::endl;
        std::cout << "sample_index: " << sample_index << std::endl;
        std::cout << "label: " << dataset.fineLabel(sample_index) << std::endl;
        print_logits(runtime_output);
        return 0;
    }

    //./mobilenetv1 --eval --max-samples=1000 --backend=cpu_ree_ref
    if (eval_only) { // 精度测试
        CIFAR100Dataset dataset(dataset_root, dataset_split, true);
        const size_t total_samples = dataset.size();
        const size_t eval_samples =
            (0 == max_samples || max_samples > total_samples) ? total_samples : max_samples;
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
        std::cout << "data_root: " << dataset_root << std::endl;
        std::cout << "backend: " << backend_kind_name(backend_kind) << std::endl;
        std::cout << "split: "
                  << (dataset_split == CIFAR100Dataset::Split::TRAIN ? "train" : "test")
                  << std::endl;
        std::cout << "samples: " << eval_samples << std::endl;
        std::cout << "top1: " << top1 << "%" << std::endl;
        std::cout << "top5: " << top5 << "%" << std::endl;
        return 0;
    }
// =================================================================================

    print_op_summary(ops);

    Value_t runtime_input({1, 3, 32, 32});
    fill_sequential_fp32(runtime_input);
    Value_t runtime_output;
    network.run({ &runtime_input }, { &runtime_output });

    expect_shape(runtime_output, {1, 100}, "network_output");
    if (is_all_zero(runtime_output)) {
        std::cerr << "mobilenetv1 output should not be all zero" << std::endl;
        return 1;
    }

    std::cout << "mobilenetv1 closed-loop test ok" << std::endl;
    return 0;
}
