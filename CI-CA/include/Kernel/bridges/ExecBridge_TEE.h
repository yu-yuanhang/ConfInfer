#ifndef __EXEC_BRIDGE_TEE_H_CA__
#define __EXEC_BRIDGE_TEE_H_CA__

#include <confinfer_host.h>
#include <image/ModelImage.h>

#include <memory>
#include <cstddef>
#include <cstdint>
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



struct TrustSpanModelRegion {
    int fd = -1;
    uint8_t *ree_addr = nullptr;
    uint64_t phys_addr = 0;
    uint32_t size = 0;
};

class ExecBridge_TEE_TrustSpan final : public ExecBridge_TEE {
public:
    ExecBridge_TEE_TrustSpan() = default;
    // ExecBridge_TEE_TrustSpan 的析构函数 比较特殊
    // 关键是涉及到各种情况下 操作失败后 程序推出后 ree 侧驱动申请的物理页释放问题
    // 这个对应的生命周期的关系 思来想去还是 交给 Bridge 最合适
    ~ExecBridge_TEE_TrustSpan() override;

    bool loadModelImage(confinfer_model_id_t model_id,
                        const Kernel::image::ModelImage& image) override;
    bool unloadModel(confinfer_model_id_t model_id, bool strict) override;
    bool executePartition(confinfer_model_id_t model_id,
                          confinfer_partition_id_t partition_id,
                          const ExecIOBlob& io_blob) override;

private:
    // 物理连续区属于 TrustSpan bridge 的传输资源
    // backend 仍只拥有通用的 ModelImage
    bool allocateRegion(size_t size);
    void releaseRegion();
    bool copyToRegion(const Kernel::image::ModelImage& image);

    // 关于 TrustSpan 的 REE 侧连续物理地址管理驱动维护的状态
    /*
	 * TRUSTSPAN_MEM_STATE_IDLE = 0,
	 * TRUSTSPAN_MEM_STATE_ALLOCATED = 1,	// 驱动已分配物理页 REE 可以 mmap 并读写
	 * TRUSTSPAN_MEM_STATE_MAPPED = 2,		// REE 当前已经 mmap 这段物理页
	 * // REE 已经 munmap 驱动认为这段页正在交给 Secure world 
	 * // 此时拒绝新的 mmap 和 FREE
	 * TRUSTSPAN_MEM_STATE_PREPARED = 3,
	 * // 	TA 已成功通过 TF-A 保护并使用这段页 REE 不可 mmap 不可 FREE
	 * TRUSTSPAN_MEM_STATE_PROTECTED = 4,    
    */

    bool mapRegion();
    bool unmapRegion();
    // cancelSecureRegion 专门用来处理 
    // REE munmap
    //     -> 驱动 PREPARE_SECURE
    //     -> 驱动状态变为 PREPARED
    //     -> TA prepare 失败
    bool cancelSecureRegion();

    TrustSpanModelRegion _model_region;
};

std::unique_ptr<ExecBridge_TEE> createCompiledTEEBridge();

} // namespace bridges
} // namespace Kernel

#endif
