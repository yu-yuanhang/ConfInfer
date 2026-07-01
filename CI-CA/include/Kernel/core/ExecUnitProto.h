#ifndef __EXEC_UNIT_PROTO_H_CA__
#define __EXEC_UNIT_PROTO_H_CA__

#include <core/ExecutionPlan.h>
#include <confinfer_protocol.h>

namespace Kernel {
namespace core {

// 表示一个协议视角下的 Value 描述信息 desc
struct ExecValueProto {
    confinfer_value_desc_t desc;

    ExecValueProto() : desc{} {}
};

// 用来表示 某个执行单元的数据面描述 数据关系图
// 本质上 我们需要把 REE 内模型内的 Layer 之间输入输出关系依赖关系传递给 TEE
// 这些信息在 REE 中就简单的用 指针来表示就行 
// 但是要静态化到协议中就需要 值表 引用表 索引区间
// confinfer_layer_io_desc_t 它是在保存某个 layer 去哪些全局引用表里取自己的输入、输出、参数
// 一个 partition 里 关系有三类：
// Layer -> Value
// Layer -> Param
// Value -> 被哪个 Layer 生产 / 被哪些 Layer 消费
struct ExecDataProto {
    confinfer_partition_data_req_t req;     // 数据面总头 记录各种数组有多少项
    // inputs outputs internals 不按 layer 分 而是按“分区数据边界”分
    // inputs / outputs / internals 是 partition 里出现过的所有 value 描述表
    std::vector<ExecValueProto> inputs;     // 表示分区外部输入
    std::vector<ExecValueProto> outputs;    // 表示分区外部输出
    std::vector<ExecValueProto> internals;  // 表示分区内部中间值
    // 每个 layer 在这些引用表中的索引目录
    std::vector<confinfer_layer_io_desc_t> layer_ios;       // 每个元素对应一个 layer 的 IO 描述
    // 基于 layer_ios 信息 查找 ref 引用表
    // input_refs / output_refs 这是 layer 到 value 的引用表
    std::vector<confinfer_layer_value_ref_t> input_refs;
    std::vector<confinfer_layer_value_ref_t> output_refs;
    // 这是 layer 到 param 的引用表
    std::vector<confinfer_layer_param_ref_t> param_refs;

    ExecDataProto() : req{}, inputs(), outputs(), internals(), layer_ios(), input_refs(), output_refs(), param_refs() {}
};

// 这里需要把 CA 内部的 ExecUnit 
// 转换成一份合适的协议中间表示 支持通过协议中提供的信息 
// 能支持 TEE 内计算
struct ExecUnitProto {
    confinfer_partition_req_t req;              // 作为协议头 描述这个执行单元整体信息
    std::vector<confinfer_layer_desc_t> layers; // 协议层的 layer 描述数组
    // 比如 conv / pool / norm / flatten 这些算子的具体属性 不同算子的 attr 大小不一样
    std::vector<uint8_t> layer_attrs;           // layer 属性 blob 
    ExecDataProto data;                         // 数据面描述 不承载真实 buffer

    ExecUnitProto() // 初始化就先置空
        : req{}, layers(), layer_attrs(), data() {}
};

// 把内部执行域枚举映射到协议枚举
uint32_t exec_domain_to_proto(ExecutionDomain domain);
// 把内部调度单元类型映射到协议枚举
uint32_t exec_unit_type_to_proto(ExecUnitType type);
// 构造并填充一个 ExecUnitProto
ExecUnitProto make_exec_unit_proto(const ExecUnit& unit);

} // namespace core
} // namespace Kernel

#endif
