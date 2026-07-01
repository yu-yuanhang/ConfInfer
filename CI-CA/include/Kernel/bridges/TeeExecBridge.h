#ifndef __TEE_EXEC_BRIDGE_H_CA__
#define __TEE_EXEC_BRIDGE_H_CA__

#include <core/ProtoExecBridge.h>
#include <core/Network.h>

#include <confinfer_host.h>

namespace Kernel {
namespace bridges {

// 这是具体的 TEE bridge 实现
// TeeExecBridge 持有用来进行通信的资源 confinfer_teec_client_t
// 和一个 ProtoExecBridge
// ProtoExecBridge 再通过 runner 使用 _client
class TeeExecBridge {
public:
    TeeExecBridge();
    ~TeeExecBridge();

    bool open(uint32_t *err_origin = nullptr);
    // 注册对应的 ProtoExecBridge
    bool install(Kernel::core::Executor *exec = EXECUTOR, uint32_t *err_origin = nullptr);
    void uninstall(Kernel::core::Executor *exec = EXECUTOR);
    void close(Kernel::core::Executor *exec = EXECUTOR);

    Kernel::core::ProtoExecBridge& bridge() { return _bridge; }
    const Kernel::core::ProtoExecBridge& bridge() const { return _bridge; }

private:
    static bool register_model(const confinfer_model_desc_t *desc,
                               confinfer_model_rsp_t *rsp,
                               void *user_ctx);
    static bool load_params(const confinfer_load_params_req_t *req,
                            const confinfer_param_desc_t *param_descs,
                            UINT param_count,
                            const void *param_blob,
                            UINT param_blob_size,
                            confinfer_load_params_rsp_t *rsp,
                            void *user_ctx);
    static bool register_partition(const confinfer_partition_req_t *req,
                                   const confinfer_layer_desc_t *layers,
                                   UINT layer_count,
                                   const void *layer_attr_blob,
                                   UINT layer_attr_blob_size,
                                   const confinfer_partition_data_req_t *data_req,
                                   const confinfer_value_desc_t *inputs,
                                   UINT input_count,
                                   const confinfer_value_desc_t *outputs,
                                   UINT output_count,
                                   const confinfer_value_desc_t *internals,
                                   UINT internal_count,
                                   const confinfer_layer_io_desc_t *layer_ios,
                                   UINT layer_io_count,
                                   const confinfer_layer_value_ref_t *input_refs,
                                   UINT input_ref_count,
                                   const confinfer_layer_value_ref_t *output_refs,
                                   UINT output_ref_count,
                                   const confinfer_layer_param_ref_t *param_refs,
                                   UINT param_ref_count,
                                   confinfer_partition_rsp_t *rsp,
                                   void *user_ctx);
    static bool unload_model(const confinfer_unload_model_req_t *req,
                             confinfer_unload_model_rsp_t *rsp,
                             void *user_ctx);
    // 留给 _bridge.setRunner(...) 用的静态回调
    // 所以这里是 static
    static bool runner(const confinfer_partition_req_t *req,
                       const confinfer_layer_desc_t *layers,
                       UINT layer_count,
                       const void *layer_attr_blob,
                       UINT layer_attr_blob_size,
                       const confinfer_partition_data_req_t *data_req,
                       const confinfer_value_desc_t *inputs,
                       UINT input_count,
                       const confinfer_value_desc_t *outputs,
                       UINT output_count,
                       const confinfer_value_desc_t *internals,
                       UINT internal_count,
                       const confinfer_layer_io_desc_t *layer_ios,
                       UINT layer_io_count,
                       const confinfer_layer_value_ref_t *input_refs,
                       UINT input_ref_count,
                       const confinfer_layer_value_ref_t *output_refs,
                       UINT output_ref_count,
                       const confinfer_layer_param_ref_t *param_refs,
                       UINT param_ref_count,
                       const void *input_blob,
                       UINT input_blob_size,
                       void *output_blob,
                       UINT output_blob_size,
                       confinfer_partition_rsp_t *rsp,
                       void *user_ctx);

private:
    confinfer_teec_client_t _client;
    bool _opened;   // 当前 client 是否已经 open
    Kernel::core::ProtoExecBridge _bridge;
};

} // namespace bridges
} // namespace Kernel

#endif
