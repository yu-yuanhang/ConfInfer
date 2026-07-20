#ifndef __EXEC_BRIDGE_PROTO_H_CA__
#define __EXEC_BRIDGE_PROTO_H_CA__

#include <core/ExecUnitProto.h>
#include <core/Network.h>

namespace Kernel {
namespace core {

// 下面定义了一个执行桥回调函数的类型
typedef bool (*ExecBridgeProtoRunnerFn)(const confinfer_partition_req_t *req,
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

typedef bool (*ExecBridgeProtoRegisterModelFn)(const confinfer_model_desc_t *desc,
                                               confinfer_model_rsp_t *rsp,
                                               void *user_ctx);

typedef bool (*ExecBridgeProtoLoadParamsFn)(const confinfer_load_params_req_t *req,
                                            const confinfer_param_desc_t *param_descs,
                                            UINT param_count,
                                            const void *param_blob,
                                            UINT param_blob_size,
                                            confinfer_load_params_rsp_t *rsp,
                                            void *user_ctx);

typedef bool (*ExecBridgeProtoRegisterPartitionFn)(const confinfer_partition_req_t *req,
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

typedef bool (*ExecBridgeProtoUnloadModelFn)(const confinfer_unload_model_req_t *req,
                                             confinfer_unload_model_rsp_t *rsp,
                                             void *user_ctx);

// 这一层核心是将 ExecUnit 编码成
// 控制面: confinfer_partition_req_t + layer_desc[]
// 数据面描述: confinfer_partition_data_req_t + value_desc[]
class ExecBridgeProto : public ExecDomainBridge {
public:
    ExecutionDomain domain() const override { return _domain; }
    bool execute(const ExecUnit& unit, Executor *exec, ThreadCtx_t *ctx) override;

    // 这里的默认值我考虑下来还是给了 ED_CPU_REE 虽然在这里只可能是 ED_CPU_TEE
    ExecBridgeProto(ExecutionDomain domain = ExecutionDomain::ED_CPU_REE)
        : _domain(domain),
          _runner(nullptr),
          _register_model(nullptr),
          _load_params(nullptr),
          _register_partition(nullptr),
          _unload_model(nullptr),
          _user_ctx(nullptr),
          _active_model_id(CONFINFER_INVALID_MODEL_ID),
          _last_proto() {}

    void setRunner(ExecBridgeProtoRunnerFn runner, void *user_ctx = nullptr) {
        _runner = runner;
        _user_ctx = user_ctx;
    }

    void setModelLifecycle(ExecBridgeProtoRegisterModelFn register_model,
                           ExecBridgeProtoLoadParamsFn load_params,
                           ExecBridgeProtoRegisterPartitionFn register_partition,
                           ExecBridgeProtoUnloadModelFn unload_model,
                           void *user_ctx = nullptr) {
        _register_model = register_model;
        _load_params = load_params;
        _register_partition = register_partition;
        _unload_model = unload_model;
        _user_ctx = user_ctx;
    }

    void clearRunner() {
        _runner = nullptr;
    }

    void clearModelLifecycle() {
        _register_model = nullptr;
        _load_params = nullptr;
        _register_partition = nullptr;
        _unload_model = nullptr;
    }

    void clearCallbacks() {
        clearRunner();
        clearModelLifecycle();
        _user_ctx = nullptr;
        _active_model_id = CONFINFER_INVALID_MODEL_ID;
    }

    bool ready() const { return nullptr != _runner; }
    bool lifecycleReady() const {
        return nullptr != _register_model &&
               nullptr != _load_params &&
               nullptr != _register_partition &&
               nullptr != _unload_model;
    }

    bool registerModel(const confinfer_model_desc_t& desc, confinfer_model_rsp_t *rsp = nullptr);
    bool loadParams(const confinfer_load_params_req_t& req,
                    const confinfer_param_desc_t *param_descs,
                    UINT param_count,
                    const void *param_blob,
                    UINT param_blob_size,
                    confinfer_load_params_rsp_t *rsp = nullptr);
    bool registerPartition(const ExecUnit& unit,
                           confinfer_model_id_t model_id,
                           confinfer_partition_rsp_t *rsp = nullptr);
    bool unloadModel(const confinfer_unload_model_req_t& req,
                     confinfer_unload_model_rsp_t *rsp = nullptr);

    const ExecUnitProto& lastProto() const { return _last_proto; }

private:
    ExecutionDomain _domain;
    // 分装五个基本的回调 
    ExecBridgeProtoRunnerFn _runner;
    ExecBridgeProtoRegisterModelFn _register_model;
    ExecBridgeProtoLoadParamsFn _load_params;
    ExecBridgeProtoRegisterPartitionFn _register_partition;
    ExecBridgeProtoUnloadModelFn _unload_model;
    void *_user_ctx;    // 给 runner 用的上下文 在当前项目中就是用来表示 confinfer_teec_client_t
    confinfer_model_id_t _active_model_id;
    // 分区描述信息 包括其中每个 Layer 的描述信息
    // 保存上一次执行时生成的协议对象
    ExecUnitProto _last_proto;
};

} // namespace core
} // namespace Kernel

#endif
