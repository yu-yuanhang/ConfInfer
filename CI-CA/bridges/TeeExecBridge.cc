#include <bridges/TeeExecBridge.h>

namespace Kernel {
namespace bridges {

TeeExecBridge::TeeExecBridge()
    : _client(),
      _opened(false),
      _bridge(Kernel::core::ExecutionDomain::ED_CPU_TEE) {}

TeeExecBridge::~TeeExecBridge() {
    close();
}

bool TeeExecBridge::register_model(const confinfer_model_desc_t *desc,
                                   confinfer_model_rsp_t *rsp,
                                   void *user_ctx) {
    confinfer_teec_client_t *client = static_cast<confinfer_teec_client_t *>(user_ctx);
    uint32_t err_origin = 0;
    TEEC_Result res;

    if (nullptr == client || nullptr == desc) {
        return false;
    }
    res = confinfer_teec_register_model(client, desc, rsp, &err_origin);
    return res == TEEC_SUCCESS;
}

bool TeeExecBridge::load_params(const confinfer_load_params_req_t *req,
                                const confinfer_param_desc_t *param_descs,
                                UINT param_count,
                                const void *param_blob,
                                UINT param_blob_size,
                                confinfer_load_params_rsp_t *rsp,
                                void *user_ctx) {
    confinfer_teec_client_t *client = static_cast<confinfer_teec_client_t *>(user_ctx);
    uint32_t err_origin = 0;
    TEEC_Result res;

    if (nullptr == client || nullptr == req) {
        return false;
    }
    res = confinfer_teec_load_params(client,
                                     req,
                                     param_descs,
                                     static_cast<size_t>(param_count),
                                     param_blob,
                                     static_cast<size_t>(param_blob_size),
                                     rsp,
                                     &err_origin);
    return res == TEEC_SUCCESS;
}

bool TeeExecBridge::register_partition(const confinfer_partition_req_t *req,
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
                                       void *user_ctx) {
    confinfer_teec_client_t *client = static_cast<confinfer_teec_client_t *>(user_ctx);
    uint32_t err_origin = 0;
    TEEC_Result res;

    if (nullptr == client || nullptr == req || nullptr == data_req) {
        return false;
    }
    res = confinfer_teec_register_partition(client,
                                            req,
                                            layers,
                                            static_cast<size_t>(layer_count),
                                            layer_attr_blob,
                                            static_cast<size_t>(layer_attr_blob_size),
                                            data_req,
                                            inputs,
                                            static_cast<size_t>(input_count),
                                            outputs,
                                            static_cast<size_t>(output_count),
                                            internals,
                                            static_cast<size_t>(internal_count),
                                            layer_ios,
                                            static_cast<size_t>(layer_io_count),
                                            input_refs,
                                            static_cast<size_t>(input_ref_count),
                                            output_refs,
                                            static_cast<size_t>(output_ref_count),
                                            param_refs,
                                            static_cast<size_t>(param_ref_count),
                                            rsp,
                                            &err_origin);
    return res == TEEC_SUCCESS;
}

bool TeeExecBridge::unload_model(const confinfer_unload_model_req_t *req,
                                 confinfer_unload_model_rsp_t *rsp,
                                 void *user_ctx) {
    confinfer_teec_client_t *client = static_cast<confinfer_teec_client_t *>(user_ctx);
    uint32_t err_origin = 0;
    TEEC_Result res;

    if (nullptr == client || nullptr == req) {
        return false;
    }
    res = confinfer_teec_unload_model(client, req, rsp, &err_origin);
    return res == TEEC_SUCCESS;
}

bool TeeExecBridge::runner(const confinfer_partition_req_t *req,
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
                           void *user_ctx) {
    confinfer_teec_client_t *client = static_cast<confinfer_teec_client_t *>(user_ctx);
    uint32_t err_origin = 0;
    TEEC_Result res;

    if (nullptr == client) {
        return false;
    }

    res = confinfer_teec_exec_partition(client, req, layers,
                                        static_cast<size_t>(layer_count),
                                        layer_attr_blob,
                                        static_cast<size_t>(layer_attr_blob_size),
                                        data_req,
                                        inputs, static_cast<size_t>(input_count),
                                        outputs, static_cast<size_t>(output_count),
                                        internals, static_cast<size_t>(internal_count),
                                        layer_ios, static_cast<size_t>(layer_io_count),
                                        input_refs, static_cast<size_t>(input_ref_count),
                                        output_refs, static_cast<size_t>(output_ref_count),
                                        param_refs, static_cast<size_t>(param_ref_count),
                                        input_blob, static_cast<size_t>(input_blob_size),
                                        output_blob, static_cast<size_t>(output_blob_size),
                                        rsp, &err_origin);
    if (res != TEEC_SUCCESS) {
        return false;
    }
    return true;
}

bool TeeExecBridge::open(uint32_t *err_origin) {
    uint32_t origin = 0;
    TEEC_Result res;

    if (_opened) {
        if (nullptr != err_origin) {
            *err_origin = 0;
        }
        return true;
    }

    // 打开 OP-TEE 客户端 context + session
    res = confinfer_teec_open(&_client, &origin);
    if (res != TEEC_SUCCESS) {
        if (nullptr != err_origin) {
            *err_origin = origin;
        }
        return false;
    }

    // 这里分别注册了 runner 和 model 生命周期的回调函数
    _bridge.setRunner(runner, &_client);
    _bridge.setModelLifecycle(register_model, load_params, register_partition,
                              unload_model, &_client);
    _opened = true;
    if (nullptr != err_origin) {
        *err_origin = origin;
    }
    return true;
}

bool TeeExecBridge::install(Kernel::core::Executor *exec, uint32_t *err_origin) {
    EXIT_ERROR_CHECK_EQ(nullptr, exec, "Executor is nullptr");
    // open 中是具体的初始化 
    // 包括 TEE Session 并 填充 ExecDomainBridge 中的 runner client
    if (!open(err_origin)) {
        return false;
    }
    // 初始化完后 注册执行区域
    exec->setExecBridge(Kernel::core::ExecutionDomain::ED_CPU_TEE, &_bridge);
    return true;
}

void TeeExecBridge::uninstall(Kernel::core::Executor *exec) {
    if (nullptr == exec) {
        return;
    }
    if (exec->execBridge(Kernel::core::ExecutionDomain::ED_CPU_TEE) == &_bridge) {
        exec->clearExecBridge(Kernel::core::ExecutionDomain::ED_CPU_TEE);
    }
}

void TeeExecBridge::close(Kernel::core::Executor *exec) {
    uninstall(exec);
    _bridge.clearCallbacks();
    if (_opened) {
        confinfer_teec_close(&_client);
        _opened = false;
    }
}

} // namespace bridges
} // namespace Kernel
