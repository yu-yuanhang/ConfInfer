#ifndef __BACKEND_H_CA__
#define __BACKEND_H_CA__

#include <core/Layer.h>
#include <core/threads.h>

using namespace Kernel::core;

namespace Kernel {
namespace backend {

enum class BackendKind : uint8_t {
    CPU_REE,
    CPU_TEE,
    // GPU,
    // NPU,
};

class Backend {
public:
    Backend() = default;
    virtual ~Backend() = default;

    virtual BackendKind kind() const = 0;
    // virtual bool supports(const LayerSlice* ls, ThreadCtx_t* ctx) const = 0;
    virtual void execute(LayerSlice* ls, ThreadCtx_t* ctx) = 0;
};

class CpuBackend: 
virtual public Backend {
public:
    CpuBackend() = default;
    ~CpuBackend() override = default;

    BackendKind kind() const override { return BackendKind::CPU_REE; }
    void execute(LayerSlice* ls, ThreadCtx_t* ctx) override;
};
class CpuBackend_TEE: 
virtual public Backend {
public:
    CpuBackend_TEE() = default;
    ~CpuBackend_TEE() override = default;

    BackendKind kind() const override { return BackendKind::CPU_TEE; }
    void execute(LayerSlice* ls, ThreadCtx_t* ctx) override;
};


} // namespace end of backend
} // namespace end of Kernel 

#endif