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

/*
 * host 层有意限制在较小职责范围内
 * 它只拥有 TEEC session 与原始命令调用细节
 * 它不理解 Layer ExecPartition 或模型运行时语义
 * 这样 bridge 保持在 model image 层
 * 不会让此文件演变为另一个执行框架
 */
TEEC_Result confinfer_teec_open(confinfer_teec_client_t *client,
                                uint32_t *err_origin);
void confinfer_teec_close(confinfer_teec_client_t *client);

/*
 * 这里暴露这些通用调用函数
 * 上层 bridge 可以复用传输原语 同时选择不同命令流程
 */
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

/*
 * 默认 bridge 在这里仅保留三个语义入口
 * 加载 model image 执行一个 partition 卸载模型
 * chunk 仍是本层以下的内部传输细节
 */
TEEC_Result confinfer_teec_prepare_model_image(confinfer_teec_client_t *client,
                                               const confinfer_prepare_model_image_req_t *req,
                                               const void *image_data,
                                               size_t image_size,
                                               confinfer_prepare_model_image_rsp_t *rsp,
                                               uint32_t *err_origin);
TEEC_Result confinfer_teec_prepare_model_image_trustspan(
    confinfer_teec_client_t *client,
    const confinfer_prepare_model_image_trustspan_req_t *req,
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
