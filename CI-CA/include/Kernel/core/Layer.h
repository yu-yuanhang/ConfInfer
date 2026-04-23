#ifndef __LAYER_H_CA__
#define __LAYER_H_CA__

#include <core/Param.h>
#include <generic/templateList.h>
#include <generic/utils.h>

namespace Kernel {

namespace backend { class Backend; } // namespace end of backend

namespace core {
using Kernel::backend::Backend;
class Executor;
class network;
class Graph;
class OpSignature;
class Layer;
class LayerSlice;

enum LayerFlags : uint32_t {
    LF_NONE             = 0,
    LF_PREFER_CPU       = 1 << 1,
    LF_PREFER_GPU       = 1 << 2,
    LF_PREFER_NPU       = 1 << 3,
    
    // 安全域相关
    LF_REQUIRE_TEE      = 1 << 4,   // 必须在 TEE 中执行

    // ......
};
#define LF_DEFAULT (LF_NONE | LF_PREFER_CPU)

enum class ParamRole : uint8_t {
    WEIGHT,
    BIAS,
    RUNNING_MEAN,
    RUNNING_VAR,
    // ......
    UNKNOWN
};

enum PADDING_MODE : uint32_t {
    ZEROS_PADDING = 200,
    // ......
};

// ADD / MUL / CONCAT 是 "结构级合并" 的最小完备集合
enum class LayerType : UINT {
    CONV2D,
    MAXPOOL2D,
    LINEAR,
    SOFTMAX,
    // ...

    // 合并算子 
    ADD,
    MUL,
    CONCAT,
};

// Params_t 生命周期归属问题 Layer 或是 OpSignature
class Params {
friend class OpSignature;
friend class Layer;
private:
    Params(): refcnt(1), params(), order() {}
    ~Params() {
        for (auto it = params.begin(); it != params.end(); ++it) {
            delete it->second;
        }
        params.clear();
    }
    Data_t* get(ParamRole role) const {
        auto it = params.find(role);
        return it == params.end() ? nullptr : it->second;
    }
    bool has(ParamRole role) const {
        return params.find(role) != params.end();
    }

    void retain() {
        // memory_order_relaxed 用于确保原子操作
        refcnt.fetch_add(1, std::memory_order_relaxed);
    }
    void release() {
        // 可能触发对象销毁 需要保证线程安全
        // fetch_sub 返回修改前的值
        if (refcnt.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            delete this;
        }
    }

public:
    // 对于重复插入的情况这里默认删除旧对象
    void insert(ParamRole role, Data_t *data) {
        // 如果该 role 已经存在 先释放旧的对象
        auto it = params.find(role);
        if (it != params.end()) {
            delete it->second;
            it->second = data;
            // 注意: order 不需要重复插入 因为 role 已经存在
        } else {
            // 新插入
            params[role] = data;
            order.push_back(role);
        }
    }

private:
    std::atomic<uint32_t> refcnt;

    // 这里的 *Data_t 为实际拥有 负责释放
    std::unordered_map<ParamRole, Data_t *> params;
    // 参数顺序 (执行 / 序列化用) 保留字段作用
    std::vector<ParamRole> order;
};

typedef struct SliceDesc_s {
    // 通用 slice 信息
    UINT sliceId;
    UINT sliceNum;

    // ===== 数据维度相关 =====
    // 输入 offset / size
    std::vector<UINT> inputOffset;
    std::vector<UINT> inputExtent;

    // 输出 offset / size
    std::vector<UINT> outputOffset;
    std::vector<UINT> outputExtent;

    // ===== 参数相关 =====
    // 比如 conv 的 weight / bias 偏移
    std::vector<UINT> paramOffset;
    std::vector<UINT> paramExtent;

    // workspace 偏移
    UINT workspaceOffset;
    UINT workspaceSize;

} SliceDesc_t;

class Layer {
// Graph 视角下只有 Layer 
friend class Backend;
friend class Executor;
friend class Graph;
friend class OpSignature;
friend class Node<Layer>;
public:

protected:
    // layer 作为计算图节点 其释放并不由自己管理
    virtual ~Layer();
    Layer() = delete;
    Layer(LayerType type, OpSignature *opSignature);
    Layer(const Layer &rhs);
    // virtual bool sliceable() const { return false; }
    inline uint32_t flags() { return _lf; }

    virtual void makeOutputs() = 0;
    virtual UINT calcWorkspaceSize() = 0;
    virtual void makeParams(Params *params) = 0;
    // 针对某些简单的算子 给出一个默认版本
    virtual LayerSlice *makeSliceDesc(UINT sliceId, UINT sliceNum);
    // 判断 Num == 0 / Num == 1 判断是否支持纵向切分 
    // 其他输入判断切分维度是否合法
    virtual bool sliceable(UINT Num = 0) const { return true;/* 后续补充 switch 基于类型简单判断*/}
private:
    void linkInit();    // link 过程调用派生类相关虚函数接口
    // bind_inputs 用于构建计算图结构 
    // 初始化 Layer 之间 / Layer 与 Value 之间的链接关系
    void bind_inputs(Value_t &value);
    inline void setParams(Params *params) { _params = params; }
public:
    // 这里需要取消拷贝构造带来的问题 
    // 理论上开发者的语义上下文中不该直接获取计算图中的具体节点
    // 所以这里拷贝构造直接 protect/private
    Layer &link() = delete;
    // 这里和可变模板函数并不冲突 单个参数会优先调用这里
    Layer &link(Value_t &value);     // 绑定某个输出
    // 这里对于多输入的情况 输入之间的顺序函数中并不检查
    template<typename... Args>
    Layer &link(Value_t &first, Args &... rest) {
        static_assert(
            (std::is_same_v<Value_t, std::remove_reference_t<Args>> && ...),
            "Layer::operator() only accepts Value_t& arguments"
        );

        // ...... inputs 数量正确性检查

        // 是否存在重复绑定的情况 
        // 判定是否存在重复绑定（同一原始数据）
        EXIT_ERROR_CHECK_EQ(true,
                            has_duplicate_address(&first, &rest...),  
                            "Duplicate Value binding detected")

        // Lambda 返回统一的 匿名函数对象
        auto bind_all = [this](Value_t& v) { bind_inputs(v); };
        bind_all(first);
        // 折叠表达式 一元右折叠 (unary right fold)
        (bind_all(rest), ...);
        
        linkInit();
        return *this;
    }
    // 取输出 (单个输出/全部同类输出)
    Value_t &output(OutputKind kind, uint32_t slot);
    Value_t &output(uint32_t idx);
    Value_t &output();

    std::vector<Value_t *> outputs(OutputKind kind);
    inline BOOL isInTEE() { return _inTEE; }

protected:
    // 从设计原则上 Layer ID 应该是 Graph 内唯一
    // 但是这样本质上并不利于 TEE 中的 Layer 对应
    // 我们的最终目标至少需要确保 CA TA 之间的 Layer 有明确的一一对应关系
    // 这里 ID 的分配必须全局唯一
    UINT _id;

    LayerType _type;
    BOOL _inTEE;
    // 其他运行时属性 LayerFlags 
    // 逻辑上应该和 OpSignature 完全对应
    uint32_t _lf;

    // _inputsL / _outputsL 主要用于辅助构建执行顺序
    // _inputs / _outputs 用于表示计算图的逻辑结构 (Value 为边 Layer 节点)
    UINT _inputsLNum;
    UINT _outputsLNum;
    // false 表示不拥有数据所有权
    List<Layer, false> _inputsL;
    List<Layer, false> _outputsL;
    // 输入输出 (Value 级)
    // 注意: 在多输入输出的情况下 输入输出 value 之间的顺序是需要有约定的
    // 输入顺序 : 在 Layer 创建后 由调用者通过参数顺序控制
    // 输出顺序 : 控制逻辑交给对应的派生类实现
    // 注意 : 这里用 vector (而不是 unordered_set) 是为了保存输入输出之间的顺序逻辑
    std::vector<Value_t*> _inputs;    // 来自前序 Layer 的输出
    // 这里容器中不能直接存储对象 会导致 项目中的相关 Value_t* 悬空
    // unique_ptr 智能指针从原理上依然支持移动语义
    // 但是从 项目逻辑简单 计算图结构完整 的角度出发 这里并不建议任何情况下转移 Value 所有权
    std::vector<std::unique_ptr<Value_t>>  _outputs;   // 本 Layer 产生的所有输出

    // ====================================================
    UINT _workspaceSize;    // workspace Layers 共用 (单位 / B)
    Params *_params;
    // ====================================================

    OpSignature *_opSignature; // 便于追溯
    // 这里要考虑是否让 Layer 持有 Graph
    // ......
    static std::atomic<UINT> _counter;
};

// 用于实现算子计算描述语义 与 计算图节点 之间解耦
// 防止语义上多个相同的算子复用 破坏 有向无环图结构
// Layer 作为计算图构成的节点
// 一个 OpSignature 可以对应结算图上的多个 Layer
class OpSignature {
public:
    OpSignature() = delete;
    OpSignature(LayerType type);
    ~OpSignature();
    inline uint32_t flags() { return _lf; }
protected:
    void dealParams(Layer *l);
protected:
    LayerType _type;
    BOOL _inTEE;
    // 其他运行时属性 LayerFlags
    uint32_t _lf;

    // layer 的释放目前最好的方式就是交给 OpSignature
    // Graph 的构建是在 Layer 构建之后 因此存在内存泄漏风险
    std::vector<Layer *> _layers;
    // 保存参数指针以获得理论上的最大化灵活性
    BOOL _ownParams;
    Params *_params;
};

class LayerSlice {
public:
    LayerSlice() {}
    LayerSlice(Layer* layer, SliceDesc_t desc)
        : _layer(layer), _desc(std::move(desc)) {}

    Layer* layer() const { return _layer; }
    const SliceDesc_t& desc() const { return _desc; }

private:
    Layer*   _layer;   // 原始 Layer
    SliceDesc_t _desc;   // 切片描述
};


} // namespace end of core
} // namespace end of Kernel 

#endif
