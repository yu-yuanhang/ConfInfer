#include "confinfer_host_internal.h"

#include <string.h>

/*
 * tmpref 填充函数限制在本文件内 因为它只是 TEEC 参数封装细节
 * 上层无需关注 op 参数的具体排布
 */
static void fill_tmpref(TEEC_Parameter *param, confinfer_teec_memref_t *mem)
{
    if (!param || !mem) {
        return;
    }
    param->tmpref.buffer = mem->buffer;
    param->tmpref.size = mem->size;
}

TEEC_Result confinfer_teec_open(confinfer_teec_client_t *client,
                                uint32_t *err_origin)
{
    TEEC_Result res;
    TEEC_UUID uuid = TA_CONFINFER_UUID;

    /*
     * 此处重新初始化完整 client 对象
     * 这样 session 状态只有一个明确的所有权起点
     * 不会在失败的 open 尝试后保留不完整状态
     */
    if (!client || !err_origin) {
        return TEEC_ERROR_BAD_PARAMETERS;
    }

    memset(client, 0, sizeof(*client));

    res = TEEC_InitializeContext(NULL, &client->ctx);
    if (res != TEEC_SUCCESS) {
        return res;
    }

    res = TEEC_OpenSession(&client->ctx, &client->sess, &uuid,
                           TEEC_LOGIN_PUBLIC, NULL, NULL, err_origin);
    if (res != TEEC_SUCCESS) {
        TEEC_FinalizeContext(&client->ctx);
        memset(client, 0, sizeof(*client));
        return res;
    }

    client->is_open = 1;
    return TEEC_SUCCESS;
}

TEEC_Result confinfer_teec_invoke_value(confinfer_teec_client_t *client,
                                        uint32_t cmd_id,
                                        uint32_t *value,
                                        uint32_t *err_origin)
{
    TEEC_Operation op;
    TEEC_Result res;

    if (!client || !client->is_open || !value || !err_origin) {
        return TEEC_ERROR_BAD_PARAMETERS;
    }

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_VALUE_INOUT, TEEC_NONE,
                                     TEEC_NONE, TEEC_NONE);
    op.params[0].value.a = *value;

    res = TEEC_InvokeCommand(&client->sess, cmd_id, &op, err_origin);
    if (res != TEEC_SUCCESS) {
        return res;
    }

    *value = op.params[0].value.a;
    return TEEC_SUCCESS;
}

TEEC_Result confinfer_teec_invoke_command(confinfer_teec_client_t *client,
                                          uint32_t cmd_id,
                                          uint32_t param_types,
                                          confinfer_teec_memref_t *mem0,
                                          confinfer_teec_memref_t *mem1,
                                          confinfer_teec_memref_t *mem2,
                                          confinfer_teec_memref_t *mem3,
                                          uint32_t *err_origin)
{
    TEEC_Operation op;
    TEEC_Result res;

    /*
     * 原始命令提交统一收敛到此函数
     * 避免上层语义接口重复 TEEC_Operation 的封装逻辑
     */
    if (!client || !client->is_open || !err_origin) {
        return TEEC_ERROR_BAD_PARAMETERS;
    }

    memset(&op, 0, sizeof(op));
    op.paramTypes = param_types;
    fill_tmpref(&op.params[0], mem0);
    fill_tmpref(&op.params[1], mem1);
    fill_tmpref(&op.params[2], mem2);
    fill_tmpref(&op.params[3], mem3);

    res = TEEC_InvokeCommand(&client->sess, cmd_id, &op, err_origin);
    if (mem0) {
        mem0->size = op.params[0].tmpref.size;
    }
    if (mem1) {
        mem1->size = op.params[1].tmpref.size;
    }
    if (mem2) {
        mem2->size = op.params[2].tmpref.size;
    }
    if (mem3) {
        mem3->size = op.params[3].tmpref.size;
    }

    return res;
}

void confinfer_teec_close(confinfer_teec_client_t *client)
{
    /*
     * close 后完整清零 client
     * bridge 将它视为一次 session 能力而不是可部分有效的长期句柄
     */
    if (!client || !client->is_open) {
        return;
    }

    TEEC_CloseSession(&client->sess);
    TEEC_FinalizeContext(&client->ctx);
    memset(client, 0, sizeof(*client));
}

TEEC_Result invoke_prepare_model_image_begin(
    confinfer_teec_client_t *client,
    const confinfer_prepare_model_image_begin_req_t *req,
    confinfer_prepare_model_image_begin_rsp_t *rsp,
    uint32_t *err_origin)
{
    confinfer_teec_memref_t mem0;
    confinfer_teec_memref_t mem1;

    mem0.buffer = (void *)req;
    mem0.size = sizeof(*req);
    mem1.buffer = rsp;
    mem1.size = sizeof(*rsp);

    return confinfer_teec_invoke_command(client,
                                         TA_CONFINFER_CMD_PREPARE_MODEL_IMAGE_BEGIN,
                                         TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,
                                                          TEEC_MEMREF_TEMP_OUTPUT,
                                                          TEEC_NONE,
                                                          TEEC_NONE),
                                         &mem0, &mem1, NULL, NULL, err_origin);
}

TEEC_Result invoke_prepare_model_image_chunk(
    confinfer_teec_client_t *client,
    const confinfer_prepare_model_image_chunk_req_t *req,
    const void *chunk_data,
    size_t chunk_size,
    confinfer_prepare_model_image_chunk_rsp_t *rsp,
    uint32_t *err_origin)
{
    confinfer_teec_memref_t mem0;
    confinfer_teec_memref_t mem1;
    confinfer_teec_memref_t mem2;

    mem0.buffer = (void *)req;
    mem0.size = sizeof(*req);
    mem1.buffer = (void *)chunk_data;
    mem1.size = chunk_size;
    mem2.buffer = rsp;
    mem2.size = sizeof(*rsp);

    return confinfer_teec_invoke_command(client,
                                         TA_CONFINFER_CMD_PREPARE_MODEL_IMAGE_CHUNK,
                                         TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,
                                                          TEEC_MEMREF_TEMP_INPUT,
                                                          TEEC_MEMREF_TEMP_OUTPUT,
                                                          TEEC_NONE),
                                         &mem0, &mem1, &mem2, NULL, err_origin);
}

TEEC_Result invoke_prepare_model_image_end(
    confinfer_teec_client_t *client,
    const confinfer_prepare_model_image_end_req_t *req,
    confinfer_prepare_model_image_rsp_t *rsp,
    uint32_t *err_origin)
{
    confinfer_teec_memref_t mem0;
    confinfer_teec_memref_t mem1;

    mem0.buffer = (void *)req;
    mem0.size = sizeof(*req);
    mem1.buffer = rsp;
    mem1.size = sizeof(*rsp);

    return confinfer_teec_invoke_command(client,
                                         TA_CONFINFER_CMD_PREPARE_MODEL_IMAGE_END,
                                         TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,
                                                          TEEC_MEMREF_TEMP_OUTPUT,
                                                          TEEC_NONE,
                                                          TEEC_NONE),
                                         &mem0, &mem1, NULL, NULL, err_origin);
}

TEEC_Result invoke_prepare_model_image_trustspan(
    confinfer_teec_client_t *client,
    const confinfer_prepare_model_image_trustspan_req_t *req,
    confinfer_prepare_model_image_rsp_t *rsp,
    uint32_t *err_origin)
{
    confinfer_teec_memref_t mem0;
    confinfer_teec_memref_t mem1;

    mem0.buffer = (void *)req;
    mem0.size = sizeof(*req);
    mem1.buffer = rsp;
    mem1.size = sizeof(*rsp);

    return confinfer_teec_invoke_command(
        client, TA_CONFINFER_CMD_PREPARE_MODEL_IMAGE_TRUSTSPAN,
        TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,
                         TEEC_MEMREF_TEMP_OUTPUT,
                         TEEC_NONE,
                         TEEC_NONE),
        &mem0, &mem1, NULL, NULL, err_origin);
}

TEEC_Result invoke_exec_partition_begin(
    confinfer_teec_client_t *client,
    const confinfer_exec_partition_begin_req_t *req,
    confinfer_exec_partition_begin_rsp_t *rsp,
    uint32_t *err_origin)
{
    confinfer_teec_memref_t mem0;
    confinfer_teec_memref_t mem1;

    mem0.buffer = (void *)req;
    mem0.size = sizeof(*req);
    mem1.buffer = rsp;
    mem1.size = sizeof(*rsp);
    return confinfer_teec_invoke_command(client,
                                         TA_CONFINFER_CMD_EXEC_PARTITION_BEGIN,
                                         TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,
                                                          TEEC_MEMREF_TEMP_OUTPUT,
                                                          TEEC_NONE,
                                                          TEEC_NONE),
                                         &mem0, &mem1, NULL, NULL, err_origin);
}

TEEC_Result invoke_exec_partition_input_chunk(
    confinfer_teec_client_t *client,
    const confinfer_exec_partition_input_chunk_req_t *req,
    const void *chunk_data,
    size_t chunk_size,
    confinfer_exec_partition_input_chunk_rsp_t *rsp,
    uint32_t *err_origin)
{
    confinfer_teec_memref_t mem0;
    confinfer_teec_memref_t mem1;
    confinfer_teec_memref_t mem2;

    mem0.buffer = (void *)req;
    mem0.size = sizeof(*req);
    mem1.buffer = (void *)chunk_data;
    mem1.size = chunk_size;
    mem2.buffer = rsp;
    mem2.size = sizeof(*rsp);
    return confinfer_teec_invoke_command(client,
                                         TA_CONFINFER_CMD_EXEC_PARTITION_INPUT_CHUNK,
                                         TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,
                                                          TEEC_MEMREF_TEMP_INPUT,
                                                          TEEC_MEMREF_TEMP_OUTPUT,
                                                          TEEC_NONE),
                                         &mem0, &mem1, &mem2, NULL, err_origin);
}

TEEC_Result invoke_exec_partition_run(
    confinfer_teec_client_t *client,
    const confinfer_exec_partition_run_req_t *req,
    confinfer_exec_partition_rsp_t *rsp,
    uint32_t *err_origin)
{
    confinfer_teec_memref_t mem0;
    confinfer_teec_memref_t mem1;

    mem0.buffer = (void *)req;
    mem0.size = sizeof(*req);
    mem1.buffer = rsp;
    mem1.size = sizeof(*rsp);
    return confinfer_teec_invoke_command(client,
                                         TA_CONFINFER_CMD_EXEC_PARTITION_RUN,
                                         TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,
                                                          TEEC_MEMREF_TEMP_OUTPUT,
                                                          TEEC_NONE,
                                                          TEEC_NONE),
                                         &mem0, &mem1, NULL, NULL, err_origin);
}

TEEC_Result invoke_exec_partition_output_chunk(
    confinfer_teec_client_t *client,
    const confinfer_exec_partition_output_chunk_req_t *req,
    void *chunk_data,
    size_t chunk_size,
    confinfer_exec_partition_output_chunk_rsp_t *rsp,
    uint32_t *err_origin)
{
    confinfer_teec_memref_t mem0;
    confinfer_teec_memref_t mem1;
    confinfer_teec_memref_t mem2;

    mem0.buffer = (void *)req;
    mem0.size = sizeof(*req);
    mem1.buffer = chunk_data;
    mem1.size = chunk_size;
    mem2.buffer = rsp;
    mem2.size = sizeof(*rsp);
    return confinfer_teec_invoke_command(client,
                                         TA_CONFINFER_CMD_EXEC_PARTITION_OUTPUT_CHUNK,
                                         TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,
                                                          TEEC_MEMREF_TEMP_INOUT,
                                                          TEEC_MEMREF_TEMP_OUTPUT,
                                                          TEEC_NONE),
                                         &mem0, &mem1, &mem2, NULL, err_origin);
}

TEEC_Result invoke_exec_partition_end(
    confinfer_teec_client_t *client,
    const confinfer_exec_partition_end_req_t *req,
    uint32_t *err_origin)
{
    confinfer_teec_memref_t mem0;

    mem0.buffer = (void *)req;
    mem0.size = sizeof(*req);
    return confinfer_teec_invoke_command(client,
                                         TA_CONFINFER_CMD_EXEC_PARTITION_END,
                                         TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,
                                                          TEEC_NONE,
                                                          TEEC_NONE,
                                                          TEEC_NONE),
                                         &mem0, NULL, NULL, NULL, err_origin);
}
