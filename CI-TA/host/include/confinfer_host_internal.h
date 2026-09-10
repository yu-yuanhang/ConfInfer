#ifndef CONFINFER_HOST_INTERNAL_H
#define CONFINFER_HOST_INTERNAL_H

#include "confinfer_host.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CONFINFER_TEEC_CHUNK_BYTES (256u * 1024u)

/*
 * 对外 host 接口只应描述语义动作
 * begin chunk end 流程是默认 bridge 的实现选择
 * 因此将其隐藏在这一内部边界之后
 */
TEEC_Result invoke_prepare_model_image_begin(
    confinfer_teec_client_t *client,
    const confinfer_prepare_model_image_begin_req_t *req,
    confinfer_prepare_model_image_begin_rsp_t *rsp,
    uint32_t *err_origin);

TEEC_Result invoke_prepare_model_image_chunk(
    confinfer_teec_client_t *client,
    const confinfer_prepare_model_image_chunk_req_t *req,
    const void *chunk_data,
    size_t chunk_size,
    confinfer_prepare_model_image_chunk_rsp_t *rsp,
    uint32_t *err_origin);

TEEC_Result invoke_prepare_model_image_end(
    confinfer_teec_client_t *client,
    const confinfer_prepare_model_image_end_req_t *req,
    confinfer_prepare_model_image_rsp_t *rsp,
    uint32_t *err_origin);

TEEC_Result invoke_prepare_model_image_trustspan(
    confinfer_teec_client_t *client,
    const confinfer_prepare_model_image_trustspan_req_t *req,
    confinfer_prepare_model_image_rsp_t *rsp,
    uint32_t *err_origin);

TEEC_Result invoke_exec_partition_begin(
    confinfer_teec_client_t *client,
    const confinfer_exec_partition_begin_req_t *req,
    confinfer_exec_partition_begin_rsp_t *rsp,
    uint32_t *err_origin);

TEEC_Result invoke_exec_partition_input_chunk(
    confinfer_teec_client_t *client,
    const confinfer_exec_partition_input_chunk_req_t *req,
    const void *chunk_data,
    size_t chunk_size,
    confinfer_exec_partition_input_chunk_rsp_t *rsp,
    uint32_t *err_origin);

TEEC_Result invoke_exec_partition_run(
    confinfer_teec_client_t *client,
    const confinfer_exec_partition_run_req_t *req,
    confinfer_exec_partition_rsp_t *rsp,
    uint32_t *err_origin);

TEEC_Result invoke_exec_partition_output_chunk(
    confinfer_teec_client_t *client,
    const confinfer_exec_partition_output_chunk_req_t *req,
    void *chunk_data,
    size_t chunk_size,
    confinfer_exec_partition_output_chunk_rsp_t *rsp,
    uint32_t *err_origin);

TEEC_Result invoke_exec_partition_end(
    confinfer_teec_client_t *client,
    const confinfer_exec_partition_end_req_t *req,
    uint32_t *err_origin);

#ifdef __cplusplus
}
#endif

#endif
