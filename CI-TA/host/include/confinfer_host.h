#ifndef CONFINFER_HOST_H
#define CONFINFER_HOST_H

#include <stddef.h>
#include <stdint.h>

#include <tee_client_api.h>

#include <conf_infer_ta.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    TEEC_Context ctx;
    TEEC_Session sess;
    uint32_t is_open;
} confinfer_teec_client_t;

typedef struct {
    void *buffer;
    size_t size;
} confinfer_teec_memref_t;

TEEC_Result confinfer_teec_open(confinfer_teec_client_t *client,
                                uint32_t *err_origin);

// 作为一个统一的 调用入口
// 但是这里 host 被设计为仅仅是 提供调用接口 
// 具体的协议的分装等等都是 CI-CA 完成 
TEEC_Result confinfer_teec_invoke_command(confinfer_teec_client_t *client,
                                          uint32_t cmd_id,
                                          uint32_t param_types,
                                          confinfer_teec_memref_t *mem0,
                                          confinfer_teec_memref_t *mem1,
                                          confinfer_teec_memref_t *mem2,
                                          confinfer_teec_memref_t *mem3,
                                          uint32_t *err_origin);

TEEC_Result confinfer_teec_invoke_value(confinfer_teec_client_t *client,
                                        uint32_t cmd_id,
                                        uint32_t *value,
                                        uint32_t *err_origin);

TEEC_Result confinfer_teec_register_model(confinfer_teec_client_t *client,
                                          const confinfer_model_desc_t *desc,
                                          confinfer_model_rsp_t *rsp,
                                          uint32_t *err_origin);

TEEC_Result confinfer_teec_load_params(confinfer_teec_client_t *client,
                                       const confinfer_load_params_req_t *req,
                                       const confinfer_param_desc_t *param_descs,
                                       size_t param_count,
                                       const void *param_blob,
                                       size_t param_blob_size,
                                       confinfer_load_params_rsp_t *rsp,
                                       uint32_t *err_origin);

TEEC_Result confinfer_teec_register_partition(confinfer_teec_client_t *client,
                                              const confinfer_partition_req_t *req,
                                              const confinfer_layer_desc_t *layers,
                                              size_t layer_count,
                                              const void *layer_attr_blob,
                                              size_t layer_attr_blob_size,
                                              const confinfer_partition_data_req_t *data_req,
                                              const confinfer_value_desc_t *inputs,
                                              size_t input_count,
                                              const confinfer_value_desc_t *outputs,
                                              size_t output_count,
                                              const confinfer_value_desc_t *internals,
                                              size_t internal_count,
                                              const confinfer_layer_io_desc_t *layer_ios,
                                              size_t layer_io_count,
                                              const confinfer_layer_value_ref_t *input_refs,
                                              size_t input_ref_count,
                                              const confinfer_layer_value_ref_t *output_refs,
                                              size_t output_ref_count,
                                              const confinfer_layer_param_ref_t *param_refs,
                                              size_t param_ref_count,
                                              confinfer_partition_rsp_t *rsp,
                                              uint32_t *err_origin);

TEEC_Result confinfer_teec_unload_model(confinfer_teec_client_t *client,
                                        const confinfer_unload_model_req_t *req,
                                        confinfer_unload_model_rsp_t *rsp,
                                        uint32_t *err_origin);

TEEC_Result confinfer_teec_exec_partition(confinfer_teec_client_t *client,
                                          const confinfer_partition_req_t *req,
                                          const confinfer_layer_desc_t *layers,
                                          size_t layer_count,
                                          const void *layer_attr_blob,
                                          size_t layer_attr_blob_size,
                                          const confinfer_partition_data_req_t *data_req,
                                          const confinfer_value_desc_t *inputs,
                                          size_t input_count,
                                          const confinfer_value_desc_t *outputs,
                                          size_t output_count,
                                          const confinfer_value_desc_t *internals,
                                          size_t internal_count,
                                          const confinfer_layer_io_desc_t *layer_ios,
                                          size_t layer_io_count,
                                          const confinfer_layer_value_ref_t *input_refs,
                                          size_t input_ref_count,
                                          const confinfer_layer_value_ref_t *output_refs,
                                          size_t output_ref_count,
                                          const confinfer_layer_param_ref_t *param_refs,
                                          size_t param_ref_count,
                                          const void *input_blob,
                                          size_t input_blob_size,
                                          void *output_blob,
                                          size_t output_blob_size,
                                          confinfer_partition_rsp_t *rsp,
                                          uint32_t *err_origin);

void confinfer_teec_close(confinfer_teec_client_t *client);

#ifdef __cplusplus
}
#endif

#endif
