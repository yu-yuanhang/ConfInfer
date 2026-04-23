#ifndef __TENSOR_H_CA__
#define __TENSOR_H_CA__

#include <All.h>

namespace Kernel {
namespace core {

#define PARAM_MAX_DIMS  6

// 强类型枚举 (enum class)
// 枚举值会直接暴露到外层作用域 必须通过枚举类型名访问
enum class DataType : int8_t {
    FP32,   // default
    FP16,   
    INT8,   // 量化推理
};

enum class DataLocation : int8_t {
    CPU,    // default
    TEE,
};

typedef struct DataShape_s {
    uint32_t    size;   // numbers 没有算上 sizeof()
    uint32_t    ndim;
    uint32_t    dims[PARAM_MAX_DIMS];
} DataShape_t;

// default (PARAM_CONST | PARAM_OWN_DATA)
enum ParamFlags : uint32_t {
    PARAM_CONST         = 1u << 0,  // 权重 / 常量
    PARAM_INTERMEDIATE  = 1u << 1,  // 计算中间结果 (Value_t 构造)
    PARAM_READONLY      = 1u << 2,  
    PARAM_SECURE        = 1u << 3,  // 位于 TEE
    PARAM_OWN_DATA      = 1u << 4,  // 是否负责释放
};

// 用于表示模型参数 或是 输入输出数据 
// (最底层表示 在计算图上的不具有任何语义)
typedef struct Data_s {
    DataShape_t     shape;    // c_out | c_in/g | h | w
    DataType        dtype;        
    DataLocation    location; // 所对应执行域
    uint32_t        flags;
    void*           ptr;

    // 默认构造函数 : 一般在 Layer 初始化列表中被调用
    // 被作用为 算子参数 (权重信息)
    Data_s(uint32_t flags):
        shape(), 
        dtype(DataType::FP32), 
        location(DataLocation::CPU), 
        flags(flags), 
        ptr(nullptr) {}
    ~Data_s() 
    { if ((flags | PARAM_OWN_DATA) && ptr) delete[] (char*)ptr; }
    Data_s(const Data_s &rhs) = delete;
    Data_s &operator=(const Data_s &rhs) = delete;
    uint32_t getTypeSize() {
        switch (dtype) {
            case DataType::INT8:
                return sizeof(int8_t);
            case DataType::FP16:
                // 这个目前也不考虑量化的事情
                return sizeof(fp16_t);
            case DataType::FP32:
                return sizeof(float32);
            EXIT_ERROR("error Data_s.dtype");
        }
        return 0;
    }
} Data_t;

// ========================================= value
enum class OutputKind : uint8_t {
    Default = 0,
    // ......
};

class Layer;
// 表示计算图上的节点
// Value 不需要引用计数 因为 Value 的归属问题比较简单
// 计算图上 Value 仅仅属于一个具体的 Layer (_outputs)
typedef struct Value_s {
    Data_t      data;
    UINT        id;             // 应该也是全局唯一 (目前没有全局管理所以也不太需要)
    Layer*      producer;       // 产生它的 Layer
    UINT        output_index;   // 在该 Layer 内是第几个输出
    OutputKind  kind;           // 输出语义

    // link->bind_inputs() 过程中填充
    std::vector<Layer *> consumers;

    Value_s(uint32_t flag = PARAM_INTERMEDIATE | PARAM_OWN_DATA):
        data(flag),
        id(INVALID_VALUE_U),
        producer(nullptr),
        output_index(INVALID_VALUE_U),
        kind(OutputKind::Default) {}
    ~Value_s() {}
    // Value_s 的拷贝都是深度拷贝
    Value_s(const Value_s &rhs) = delete;
    Value_s &operator=(const Value_s &rhs) = delete;
} Value_t;


} // namespace end of core
} // namespace end of Kernel 
#endif