#ifndef __EXEC_BRIDGE_TEE_H_CA__
#define __EXEC_BRIDGE_TEE_H_CA__

#include <confinfer_host.h>
#include <image/ModelImage.h>

#include <memory>
#include <vector>

namespace Kernel {
namespace bridges {

struct ExecIOBlob {
    const uint8_t *input_data;
    uint32_t input_bytes;
    uint32_t input_count;
    uint8_t *output_data;
    uint32_t output_bytes;
    uint32_t output_count;

    ExecIOBlob()
        : input_data(nullptr),
          input_bytes(0),
          input_count(0),
          output_data(nullptr),
          output_bytes(0),
          output_count(0) {}
};

class ExecBridge_TEE {
public:
    ExecBridge_TEE();
    virtual ~ExecBridge_TEE();

    // 控制声明周期
    bool open(uint32_t *err_origin = nullptr);
    bool ready() const;
    void close();

    // 处于解耦的设计 bridge 只认三类对象 model_id ModelImage ExecIOBlob
    // ExecIOBlob 是通过 backend 负责打包
    // bridge 不直接接触 ExecPartition Layer Value_t ......
    // 模型装载控制
    virtual bool loadModelImage(confinfer_model_id_t model_id,
                                const Kernel::image::ModelImage& image) = 0;
    virtual bool unloadModel(confinfer_model_id_t model_id, bool strict) = 0;
    // 执行控制
    virtual bool executePartition(confinfer_model_id_t model_id,
                            confinfer_partition_id_t partition_id,
                            const ExecIOBlob& io_blob) = 0;

protected:
    confinfer_teec_client_t *client() { return &_client; }
    const confinfer_teec_client_t *client() const { return &_client; }

    confinfer_model_id_t loadedModelId() const { return _loaded_model_id; }
    void markModelLoaded(confinfer_model_id_t model_id) { _loaded_model_id = model_id; }
    void clearLoadedModel() { _loaded_model_id = CONFINFER_INVALID_MODEL_ID; }

private:
    confinfer_teec_client_t _client;
    bool _opened;
    confinfer_model_id_t _loaded_model_id;
};

class ExecBridge_TEE_Default final : public ExecBridge_TEE {
public:
    ExecBridge_TEE_Default() = default;
    ~ExecBridge_TEE_Default() override = default;

    bool loadModelImage(confinfer_model_id_t model_id,
                        const Kernel::image::ModelImage& image) override;
    bool unloadModel(confinfer_model_id_t model_id, bool strict) override;
    bool executePartition(confinfer_model_id_t model_id,
                          confinfer_partition_id_t partition_id,
                          const ExecIOBlob& io_blob) override;
};

class ExecBridge_TEE_TrustSpan final : public ExecBridge_TEE {
public:
    ExecBridge_TEE_TrustSpan() = default;
    ~ExecBridge_TEE_TrustSpan() override = default;

    bool loadModelImage(confinfer_model_id_t model_id,
                        const Kernel::image::ModelImage& image) override;
    bool unloadModel(confinfer_model_id_t model_id, bool strict) override;
    bool executePartition(confinfer_model_id_t model_id,
                          confinfer_partition_id_t partition_id,
                          const ExecIOBlob& io_blob) override;
};

std::unique_ptr<ExecBridge_TEE> createCompiledTEEBridge();

} // namespace bridges
} // namespace Kernel

#endif
