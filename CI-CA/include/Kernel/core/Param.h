#ifndef __TENSOR_H_CA__
#define __TENSOR_H_CA__

#include <All.h>
#include <cstring>

namespace Kernel {
namespace core {

#define PARAM_MAX_DIMS  6

// 强类型枚举 (enum class)
// 枚举值会直接暴露到外层作用域 必须通过枚举类型名访问
enum class DataType : int8_t {
    FP32,   // default
    FP16,   
    INT8,   // 量化推理
    INT32,
};

enum class DataLocation : int8_t {
    CPU,    // default
    TEE,
};

typedef struct DataShape_s {
    uint32_t    size;   // numbers 没有算上 sizeof()
    uint32_t    ndim;
    uint32_t    dims[PARAM_MAX_DIMS];

    DataShape_s(): size(0), ndim(0), dims{0} {}
    DataShape_s(std::initializer_list<uint32_t> shape_dims): size(1), ndim(0), dims{0} {
        EXIT_ERROR_CHECK_EQ(true, shape_dims.size() > PARAM_MAX_DIMS,
            "DataShape dims exceed PARAM_MAX_DIMS");

        ndim = static_cast<uint32_t>(shape_dims.size());
        if (0 == ndim) {
            size = 0;
            return;
        }

        uint32_t idx = 0;
        for (auto it = shape_dims.begin(); it != shape_dims.end(); ++it, ++idx) {
            dims[idx] = *it;
            size *= *it;
        }
    }
} DataShape_t;

enum ParamFlags : uint32_t {
    PARAM_NONE          = 0,
    PARAM_CONST         = 1u << 0,  // 权重 / 常量
    PARAM_INPUT         = 1u << 1,  // 网络输入
    PARAM_OUTPUT        = 1u << 2,  // 网络输出
    PARAM_INTERMEDIATE  = 1u << 3,  // 计算中间结果
    PARAM_READONLY      = 1u << 4,
    PARAM_SECURE        = 1u << 5,  // 位于 TEE
    PARAM_OWN_DATA      = 1u << 6,  // 是否负责释放
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
    Data_s(uint32_t flags, std::initializer_list<uint32_t> shape_dims,
           DataType dtype = DataType::FP32,
           DataLocation location = DataLocation::CPU):
        shape(shape_dims),
        dtype(dtype),
        location(location),
        flags(flags),
        ptr(nullptr) {}
    ~Data_s() { release(); }
    Data_s(const Data_s &rhs) = delete;
    Data_s &operator=(const Data_s &rhs) = delete;
    uint32_t getTypeSize() {
        switch (dtype) {
            case DataType::INT8:
                return sizeof(int8_t);
            case DataType::INT32:
                return sizeof(int32_t);
            case DataType::FP16:
                // 这个目前也不考虑量化的事情
                return sizeof(fp16_t);
            case DataType::FP32:
                return sizeof(float32);
            EXIT_ERROR("error Data_s.dtype");
        }
        return 0;
    }
    void copyMetaFrom(const Data_s& rhs) {
        shape = rhs.shape;
        dtype = rhs.dtype;
        location = rhs.location;
        flags = rhs.flags;
        ptr = rhs.ptr;
    }
    void copyShapeFrom(const Data_s& rhs) {
        shape = rhs.shape;
    }
    void copyTypeFrom(const Data_s& rhs) {
        dtype = rhs.dtype;
        location = rhs.location;
    }
    void copyDescFrom(const Data_s& rhs) {
        copyShapeFrom(rhs);
        copyTypeFrom(rhs);
    }
    void borrowFrom(const Data_s& rhs, uint32_t extra_flags = PARAM_NONE) {
        copyDescFrom(rhs);
        ptr = rhs.ptr;
        flags = extra_flags;
    }
    void deepCopyFrom(const Data_s& rhs, uint32_t extra_flags = PARAM_NONE) {
        EXIT_ERROR_CHECK_EQ(nullptr, rhs.ptr, "Source data buffer is nullptr");

        const bool can_reuse = (nullptr != ptr)
            && (flags & PARAM_OWN_DATA)
            && (shape.size == rhs.shape.size)
            && (dtype == rhs.dtype);

        if (!can_reuse) {
            release();
            copyDescFrom(rhs);
            flags = extra_flags;
            alloc();
        } else {
            copyDescFrom(rhs);
            flags = extra_flags | PARAM_OWN_DATA;
        }

        std::memcpy(ptr, rhs.ptr, shape.size * getTypeSize());
    }
    void alloc() {
        EXIT_ERROR_CHECK_NE(nullptr, ptr, "Data buffer already allocated");
        EXIT_ERROR_CHECK_EQ(0, shape.size, "Data shape size is zero");
        flags |= PARAM_OWN_DATA;
        ptr = new char[shape.size * getTypeSize()];
    }
    void release() {
        if ((flags & PARAM_OWN_DATA) && ptr) {
            delete[] static_cast<char*>(ptr);
        }
        ptr = nullptr;
        flags &= ~PARAM_OWN_DATA;
    }
} Data_t;

// ========================================= value
enum class OutputKind : uint8_t {
    Default = 0,
    Indices,    // 用来支持 pool return_indices=true
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

    // 默认构造为占位 / 外部托管语义，不自动拥有数据
    Value_s(uint32_t flag = PARAM_NONE):
        data(flag),
        id(INVALID_VALUE_U),
        producer(nullptr),
        output_index(INVALID_VALUE_U),
        kind(OutputKind::Default) {}
    Value_s(std::initializer_list<uint32_t> shape_dims,
            DataType dtype = DataType::FP32,
            DataLocation location = DataLocation::CPU,
            uint32_t flag = PARAM_NONE):
        data(flag, shape_dims, dtype, location),
        id(INVALID_VALUE_U),
        producer(nullptr),
        output_index(INVALID_VALUE_U),
        kind(OutputKind::Default) {}
    ~Value_s() {}
    // Value_s 的拷贝都是深度拷贝
    Value_s(const Value_s &rhs) = delete;
    Value_s &operator=(const Value_s &rhs) = delete;

    // alloc 操作会自动添加 PARAM_OWN_DATA
    void alloc() { data.alloc(); }
    void copyMetaFrom(const Value_s& rhs) {
        data.copyMetaFrom(rhs.data);
    }
    void copyDescFrom(const Value_s& rhs) {
        data.copyDescFrom(rhs.data);
    }
    void borrowFrom(const Value_s& rhs, uint32_t extra_flags = PARAM_NONE) {
        data.borrowFrom(rhs.data, extra_flags);
    }
    void deepCopyFrom(const Value_s& rhs, uint32_t extra_flags = PARAM_NONE) {
        data.deepCopyFrom(rhs.data, extra_flags);
    }
} Value_t;


} // namespace end of core
} // namespace end of Kernel 
#endif
