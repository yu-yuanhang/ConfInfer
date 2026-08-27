#ifndef __BACKEND_H_CA__
#define __BACKEND_H_CA__

#include <confinfer_protocol.h>
#include <core/ExecutionPartition.h>
#include <core/Layer.h>
#include <image/ModelImage.h>

#include <memory>

using namespace Kernel::core;

namespace Kernel {
namespace bridges { class ExecBridge_TEE; }
namespace backend {

enum class BackendKind : uint8_t {
    BK_CPU_REE,
    BK_CPU_REE_REF,
    BK_CPU_REE_ALT0,
    BK_CPU_REE_ALT1,
    BK_CPU_TEE,
};

class Backend {
public:
    Backend() = default;
    virtual ~Backend() = default;

    virtual BackendKind kind() const = 0;
    virtual bool supports(ExecutionDomain domain) const = 0;
    virtual void prepare(Layer* layer) = 0;
    virtual void execute(Layer* layer, ExecContext_t* ctx) = 0;
    virtual void prepare(const ExecPartition& part, ExecContext_t* ctx);
    virtual void execute(const ExecPartition& part, ExecContext_t* ctx);
    virtual void resetRuntime(ExecContext_t* ctx, bool strict);
};

class Backend_CPU_REE:
virtual public Backend {
public:
    Backend_CPU_REE() = default;
    ~Backend_CPU_REE() override = default;

    BackendKind kind() const override { return BackendKind::BK_CPU_REE; }
    bool supports(ExecutionDomain domain) const override {
        return domain == ExecutionDomain::ED_DEFAULT ||
               domain == ExecutionDomain::ED_CPU_REE;
    }
    void prepare(Layer* layer) override;
    void execute(Layer* layer, ExecContext_t* ctx) override;
};

class Backend_CPU_REE_REF:
virtual public Backend {
public:
    Backend_CPU_REE_REF() = default;
    ~Backend_CPU_REE_REF() override = default;

    BackendKind kind() const override { return BackendKind::BK_CPU_REE_REF; }
    bool supports(ExecutionDomain domain) const override {
        return domain == ExecutionDomain::ED_DEFAULT ||
               domain == ExecutionDomain::ED_CPU_REE;
    }
    void prepare(Layer* layer) override;
    void execute(Layer* layer, ExecContext_t* ctx) override;
};

class Backend_CPU_TEE:
virtual public Backend {
public:
    Backend_CPU_TEE();
    ~Backend_CPU_TEE() override;

    BackendKind kind() const override { return BackendKind::BK_CPU_TEE; }
    bool supports(ExecutionDomain domain) const override {
        return domain == ExecutionDomain::ED_CPU_TEE;
    }
    void prepare(Layer* layer) override;
    void execute(Layer* layer, ExecContext_t* ctx) override;
    void prepare(const ExecPartition& part, ExecContext_t* ctx) override;
    void execute(const ExecPartition& part, ExecContext_t* ctx) override;
    void resetRuntime(ExecContext_t* ctx, bool strict) override;

private:
    bool loadRuntime(confinfer_model_id_t model_id,
                     const std::vector<ExecPartition>& parts);
    bool unloadRuntime(confinfer_model_id_t model_id, bool strict);
    bool openBridge(uint32_t *err_origin = nullptr);

    bool hasRuntime(confinfer_model_id_t model_id) const;
    bool isBridgeOpened() const;

    std::unique_ptr<Kernel::bridges::ExecBridge_TEE> _bridge;
    Kernel::image::ModelImageBuilder _imageBuilder;
    Kernel::image::ModelImage _runtimeImage;
    bool _runtimeLoaded;
    confinfer_model_id_t _loadedModelId;
};

} // namespace backend
} // namespace Kernel

#endif
