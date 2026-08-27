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

// session 生命周期接口
// 只负责 TEEC 会话本身 不承载模型语义
// CI-CA 侧 backend 通过 bridge 持有一个 confinfer_teec_client_t 但是不直接操作
TEEC_Result confinfer_teec_open(confinfer_teec_client_t *client,
                                uint32_t *err_origin);
void confinfer_teec_close(confinfer_teec_client_t *client);

// 这组是底层通用封装 给 bridge 内部复用
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

// 面向默认 bridge 的三条模型执行接口          
TEEC_Result confinfer_teec_prepare_model_image(confinfer_teec_client_t *client,
                                               const confinfer_prepare_model_image_req_t *req,
                                               const void *image_data,
                                               size_t image_size,
                                               confinfer_prepare_model_image_rsp_t *rsp,
                                               uint32_t *err_origin);
TEEC_Result confinfer_teec_exec_partition(confinfer_teec_client_t *client,
                                          const confinfer_exec_partition_req_t *req,
                                          const void *input_blob,
                                          size_t input_blob_size,
                                          void *output_blob,
                                          size_t output_blob_size,
                                          confinfer_exec_partition_rsp_t *rsp,
                                          uint32_t *err_origin);
TEEC_Result confinfer_teec_unload_model(confinfer_teec_client_t *client,
                                        const confinfer_unload_model_req_t *req,
                                        confinfer_unload_model_rsp_t *rsp,
                                        uint32_t *err_origin);

#ifdef __cplusplus
}
#endif

#endif
