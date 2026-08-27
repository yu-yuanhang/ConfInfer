#ifndef __MOBILENETV1_SUPPORT_H__
#define __MOBILENETV1_SUPPORT_H__

#include <string>
#include <vector>

#include <core/Network.h>
#include <core/cifar100_dataset.h>

#include "mobilenetv1_model.h"

namespace mobilenetv1_demo {

struct MobileNetRunOptions {
    std::string manifest_path;
    std::string dataset_root;
    Kernel::backend::BackendKind backend_kind;
    bool load_only;
    bool dataset_only;
    bool eval_only;
    bool print_logits_only;
    bool print_partition_only;
    bool partition_only;
    Kernel::core::CIFAR100Dataset::Split dataset_split;
    size_t max_samples;
    size_t sample_index;
};

// 命令行参数解析:
// 将 demo 的运行模式、数据集路径、backend 选择等收敛到一个结构里。
MobileNetRunOptions parse_options(int argc, char* argv[]);

// 名称格式化:
// 用于把 backend / execution domain 转成便于打印的字符串。
const char* backend_kind_name(Kernel::backend::BackendKind kind);
const char* exec_domain_name(Kernel::core::ExecutionDomain domain);

// 张量辅助:
// 这些函数服务于 demo 内部的输入填充、shape 检查和简单结果判定，
// 不属于模型结构本身，只是运行时验证工具。
void expect_shape(const Kernel::core::Value_t& value,
                  std::initializer_list<Kernel::UINT> dims,
                  const char* name);
void fill_sequential_fp32(Kernel::core::Value_t& value);
bool is_all_zero(const Kernel::core::Value_t& value,
                 Kernel::FLOAT eps = 1e-7f);

// 推理结果辅助:
// 用于分类结果检查与 logits 打印。
Kernel::UINT top1_index(const Kernel::core::Value_t& logits);
bool in_topk(const Kernel::core::Value_t& logits,
             Kernel::UINT label,
             Kernel::UINT k);
void print_logits(const Kernel::core::Value_t& logits);

// 模型结构与参数加载:
// 在真正执行前，先检查模型输出 shape 是否符合预期，
// 并完成参数文件到模型参数槽位的装载。
void verify_model_structure(const MobileNetV1Model& model,
                            const Kernel::core::ParamBindingTable& bindings);
Kernel::core::ParamLoadReport load_model_params_or_die(
    const std::string& manifest_path,
    const Kernel::core::ParamBindingTable& bindings);

// 分区观测:
// 用于打印和验证 network.prepare() 之后生成的
// ExecPartition / PartitionGraph。
void print_op_summary(const std::vector<MobileNetOpInfo>& ops);
void print_partition_summary(const Kernel::core::Network& network);
void validate_partition_pipeline(const Kernel::core::Network& network);

} // namespace mobilenetv1_demo

#endif
