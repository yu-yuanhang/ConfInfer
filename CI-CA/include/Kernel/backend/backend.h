#ifndef __BACKEND_H_CA__
#define __BACKEND_H_CA__

#include <core/Layer.h>
#include <core/threads.h>

using namespace Kernel::core;

namespace Kernel {
namespace backend {

enum class BackendKind : uint8_t {
    BK_CPU_REE,        // 当前默认 CPU backend
    BK_CPU_REE_REF,    // 参考实现 / 基准实现
    BK_CPU_REE_ALT0,   // 预留给其他 CPU 实现
    BK_CPU_REE_ALT1,   // 预留给其他 CPU 实现
    BK_CPU_TEE,
    // GPU,
    // NPU,
};

class Backend {
public:
    Backend() = default;
    virtual ~Backend() = default;

    virtual BackendKind kind() const = 0;
    virtual void prepare(LayerSlice* ls) = 0;
    virtual void execute(LayerSlice* ls, ThreadCtx_t* ctx) = 0;
};

class Backend_CPU_REE:
virtual public Backend {
public:
    Backend_CPU_REE() = default;
    ~Backend_CPU_REE() override = default;

    BackendKind kind() const override { return BackendKind::BK_CPU_REE; }
    void prepare(LayerSlice* ls) override;
    void execute(LayerSlice* ls, ThreadCtx_t* ctx) override;
};
class Backend_CPU_REE_REF:
virtual public Backend {
public:
    Backend_CPU_REE_REF() = default;
    ~Backend_CPU_REE_REF() override = default;

    BackendKind kind() const override { return BackendKind::BK_CPU_REE_REF; }
    void prepare(LayerSlice* ls) override;
    void execute(LayerSlice* ls, ThreadCtx_t* ctx) override;
};
class Backend_CPU_TEE:
virtual public Backend {
public:
    Backend_CPU_TEE() = default;
    ~Backend_CPU_TEE() override = default;

    BackendKind kind() const override { return BackendKind::BK_CPU_TEE; }
    void prepare(LayerSlice* ls) override;
    void execute(LayerSlice* ls, ThreadCtx_t* ctx) override;
};


} // namespace end of backend
} // namespace end of Kernel 

#endif
