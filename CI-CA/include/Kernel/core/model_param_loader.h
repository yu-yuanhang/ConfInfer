#ifndef __MODEL_PARAM_LOADER_H_CA__
#define __MODEL_PARAM_LOADER_H_CA__

#include <All.h>
#include <core/Param.h>

namespace Kernel {
namespace core {

/*
    这里假设前提 外部导出的参数文件 必须给每个参数一个稳定名字，不然文件系统里没法区分
    所以现在 manifest 里是这种形式：
    {
    "param_count": 128,
    "params": {
        "ops.0.weight": { ... },
        "ops.1.weight": { ... },
        "ops.1.bias": { ... }
    }
    }
    
    ConfInfer 本身不对参数进行命名 或着说其中的参数是没有命名的
    所以需要 参数绑定表 把外部参数名，例如 "ops.1.weight"
    翻译成 ConfInfer 内部某个 Data_t*

    但是绑定表 具体的 对应关系 是模型自己构建

 */

// 一条外部参数项到框架内部 Data_t 的绑定关系。
// external_name 对应 params.json 中的参数名；
// data 指向框架内真实承载参数数据的目标缓冲区；
// target_name 仅用于调试和报错定位，不参与实际匹配。
struct ParamBinding {
    std::string external_name;  // 外部 manifest 里的参数名
    Data_t* data;               // 指向框架内部真实参数存储位置
    std::string target_name;

    ParamBinding(const std::string& external_name,
                 Data_t* data,
                 const std::string& target_name = "")
        : external_name(external_name),
          data(data),
          target_name(target_name) {}
};

// 参数绑定表由模型侧构建。
// 加载器本身不推断“某个外部参数应该属于哪一层”，
// 只按这张表完成 external_name -> Data_t* 的装载。
class ParamBindingTable {
public:
    void add(const std::string& external_name,
             Data_t* data,
             const std::string& target_name = "");

    const ParamBinding* find(const std::string& external_name) const;
    const std::vector<ParamBinding>& items() const { return bindings_; }
    size_t size() const { return bindings_.size(); }

private:
    std::vector<ParamBinding> bindings_;
};

// 参数加载时的行为选项。
struct ParamLoadOptions {
    // manifest 中允许存在当前未绑定的参数项。
    // 例如某些仅训练阶段使用的 buffer，可先跳过。
    bool allow_unused_manifest_entries;
    // 绑定表中的每一项都必须在 manifest 中找到。
    bool require_all_bindings;
    // 当目标 Data_t 还没有 shape/dtype 描述时，
    // 是否允许用 manifest 中的元信息补齐。
    bool init_empty_target_meta;

    ParamLoadOptions()
        : allow_unused_manifest_entries(true),
          require_all_bindings(true),
          init_empty_target_meta(false) {}
};

// 一次参数加载完成后的统计信息。
struct ParamLoadReport {
    UINT loaded_param_count;
    UINT skipped_manifest_entry_count;
    size_t loaded_bytes;

    ParamLoadReport()
        : loaded_param_count(0),
          skipped_manifest_entry_count(0),
          loaded_bytes(0) {}
};

// 读取 params.json 及其对应的二进制参数文件，
// 并按 bindings 中提供的映射关系写入框架内部参数区。
ParamLoadReport loadModelParams(const std::string& manifest_path,
                                const ParamBindingTable& bindings,
                                const ParamLoadOptions& options = ParamLoadOptions());

} // namespace core
} // namespace Kernel

#endif
