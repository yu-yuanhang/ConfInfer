#include "confinfer_host_internal.h"

#include <string.h>

TEEC_Result confinfer_teec_prepare_model_image_trustspan(
    confinfer_teec_client_t *client,
    const confinfer_prepare_model_image_trustspan_req_t *req,
    confinfer_prepare_model_image_rsp_t *rsp,
    uint32_t *err_origin)
{
    if (!client || !client->is_open || !req || !rsp || !err_origin) {
        return TEEC_ERROR_BAD_PARAMETERS;
    }
    if (req->version != CONFINFER_PROTOCOL_VERSION ||
        req->image_size == 0 || req->phys_addr == 0 ||
        req->region_size < req->image_size) {
        return TEEC_ERROR_BAD_PARAMETERS;
    }

    return invoke_prepare_model_image_trustspan(client, req, rsp, err_origin);
}

/*
 * 负责判断一个语义请求能否由一次 TEEC 调用完成
 * 或者是否需要展开为 begin chunk end 三个步骤
 * 不过这里的函数 上层调用者不应该感知这一传输细节
 */
TEEC_Result confinfer_teec_prepare_model_image(confinfer_teec_client_t *client,
                                               const confinfer_prepare_model_image_req_t *req,
                                               const void *image_data,
                                               size_t image_size,
                                               confinfer_prepare_model_image_rsp_t *rsp,
                                               uint32_t *err_origin)
{
    confinfer_prepare_model_image_begin_req_t begin_req;
    confinfer_prepare_model_image_begin_rsp_t begin_rsp;
    confinfer_prepare_model_image_chunk_req_t chunk_req;
    confinfer_prepare_model_image_chunk_rsp_t chunk_rsp;
    confinfer_prepare_model_image_end_req_t end_req;
    const uint8_t *cursor = (const uint8_t *)image_data;
    uint32_t offset = 0;
    TEEC_Result res = TEEC_SUCCESS;
    confinfer_teec_memref_t mem0;
    confinfer_teec_memref_t mem1;
    confinfer_teec_memref_t mem2;

    if (!client || !client->is_open || !req || !rsp || !err_origin) {
        return TEEC_ERROR_BAD_PARAMETERS;
    }
    if (req->version != CONFINFER_PROTOCOL_VERSION || req->image_size != image_size) {
        return TEEC_ERROR_BAD_PARAMETERS;
    }
    /*
     * 只有当 image 超过单次传输容量时 才切换到 chunk 模式
     * 对上层而言 它仍然是一次 prepare model image 请求
     */
    if (image_size > CONFINFER_TEEC_CHUNK_BYTES) {
        memset(&begin_req, 0, sizeof(begin_req));
        memset(&begin_rsp, 0, sizeof(begin_rsp));
        begin_req.version = CONFINFER_PROTOCOL_VERSION;
        begin_req.model_id = req->model_id;
        begin_req.total_image_size = (uint32_t)image_size;
        begin_req.flags = req->flags;

        res = invoke_prepare_model_image_begin(client, &begin_req, &begin_rsp, err_origin);
        if (res != TEEC_SUCCESS) {
            return res;
        }
        if (begin_rsp.version != CONFINFER_PROTOCOL_VERSION ||
            begin_rsp.model_id != req->model_id ||
            begin_rsp.status != CONFINFER_STATUS_OK) {
            return TEEC_ERROR_GENERIC;
        }

        while (offset < image_size) {
            const uint32_t chunk_size =
                (uint32_t)(((image_size - offset) > CONFINFER_TEEC_CHUNK_BYTES) ?
                           CONFINFER_TEEC_CHUNK_BYTES : (image_size - offset));
            memset(&chunk_req, 0, sizeof(chunk_req));
            memset(&chunk_rsp, 0, sizeof(chunk_rsp));
            chunk_req.version = CONFINFER_PROTOCOL_VERSION;
            chunk_req.model_id = req->model_id;
            chunk_req.chunk_offset = offset;
            chunk_req.chunk_size = chunk_size;
            chunk_req.total_image_size = (uint32_t)image_size;
            chunk_req.flags = req->flags;

            res = invoke_prepare_model_image_chunk(client, &chunk_req,
                                                   cursor + offset, chunk_size,
                                                   &chunk_rsp, err_origin);
            if (res != TEEC_SUCCESS) {
                return res;
            }
            if (chunk_rsp.version != CONFINFER_PROTOCOL_VERSION ||
                chunk_rsp.model_id != req->model_id ||
                chunk_rsp.status != CONFINFER_STATUS_OK ||
                chunk_rsp.accepted_bytes != chunk_size ||
                chunk_rsp.next_offset != offset + chunk_size) {
                return TEEC_ERROR_GENERIC;
            }
            offset += chunk_size;
        }

        memset(&end_req, 0, sizeof(end_req));
        end_req.version = CONFINFER_PROTOCOL_VERSION;
        end_req.model_id = req->model_id;
        end_req.total_image_size = (uint32_t)image_size;
        end_req.flags = req->flags;
        memset(rsp, 0, sizeof(*rsp));
        return invoke_prepare_model_image_end(client, &end_req, rsp, err_origin);
    }

    mem0.buffer = (void *)req;
    mem0.size = sizeof(*req);
    mem1.buffer = (void *)image_data;
    mem1.size = image_size;
    mem2.buffer = rsp;
    mem2.size = sizeof(*rsp);

    return confinfer_teec_invoke_command(client,
                                         TA_CONFINFER_CMD_PREPARE_MODEL_IMAGE,
                                         TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,
                                                          TEEC_MEMREF_TEMP_INPUT,
                                                          TEEC_MEMREF_TEMP_OUTPUT,
                                                          TEEC_NONE),
                                         &mem0, &mem1, &mem2, NULL, err_origin);
}

TEEC_Result confinfer_teec_exec_partition(confinfer_teec_client_t *client,
                                          const confinfer_exec_partition_req_t *req,
                                          const void *input_blob,
                                          size_t input_blob_size,
                                          void *output_blob,
                                          size_t output_blob_size,
                                          confinfer_exec_partition_rsp_t *rsp,
                                          uint32_t *err_origin)
{
    confinfer_exec_partition_begin_req_t begin_req;
    confinfer_exec_partition_begin_rsp_t begin_rsp;
    confinfer_exec_partition_input_chunk_req_t input_chunk_req;
    confinfer_exec_partition_input_chunk_rsp_t input_chunk_rsp;
    confinfer_exec_partition_run_req_t run_req;
    confinfer_exec_partition_output_chunk_req_t output_chunk_req;
    confinfer_exec_partition_output_chunk_rsp_t output_chunk_rsp;
    confinfer_exec_partition_end_req_t end_req;
    const uint8_t *input_cursor = (const uint8_t *)input_blob;
    uint8_t *output_cursor = (uint8_t *)output_blob;
    uint32_t err_origin_local = 0;
    uint32_t offset = 0;
    TEEC_Result res = TEEC_SUCCESS;
    confinfer_teec_memref_t mem0;
    confinfer_teec_memref_t mem1;
    confinfer_teec_memref_t mem2;
    confinfer_teec_memref_t mem3;

    if (!client || !client->is_open || !req || !rsp || !err_origin) {
        return TEEC_ERROR_BAD_PARAMETERS;
    }
    if (req->version != CONFINFER_PROTOCOL_VERSION ||
        req->input_bytes != input_blob_size ||
        req->output_bytes != output_blob_size) {
        return TEEC_ERROR_BAD_PARAMETERS;
    }
    /*
     * 任一方向超过单个 chunk 容量时 执行传输拆分
     * 但仍保持原有语义顺序
     * 先写入全部输入 再执行一次 最后读取全部输出
     */
    if (input_blob_size > CONFINFER_TEEC_CHUNK_BYTES ||
        output_blob_size > CONFINFER_TEEC_CHUNK_BYTES) {
        memset(&begin_req, 0, sizeof(begin_req));
        memset(&begin_rsp, 0, sizeof(begin_rsp));
        begin_req.version = CONFINFER_PROTOCOL_VERSION;
        begin_req.model_id = req->model_id;
        begin_req.partition_id = req->partition_id;
        begin_req.input_count = req->input_count;
        begin_req.output_count = req->output_count;
        begin_req.total_input_bytes = (uint32_t)input_blob_size;
        begin_req.total_output_bytes = (uint32_t)output_blob_size;
        begin_req.flags = req->flags;
        res = invoke_exec_partition_begin(client, &begin_req, &begin_rsp, err_origin);
        if (res != TEEC_SUCCESS) {
            return res;
        }
        if (begin_rsp.version != CONFINFER_PROTOCOL_VERSION ||
            begin_rsp.model_id != req->model_id ||
            begin_rsp.partition_id != req->partition_id ||
            begin_rsp.status != CONFINFER_STATUS_OK) {
            return TEEC_ERROR_GENERIC;
        }

        offset = 0;
        while (offset < input_blob_size) {
            const uint32_t chunk_size =
                (uint32_t)(((input_blob_size - offset) > CONFINFER_TEEC_CHUNK_BYTES) ?
                           CONFINFER_TEEC_CHUNK_BYTES : (input_blob_size - offset));
            memset(&input_chunk_req, 0, sizeof(input_chunk_req));
            memset(&input_chunk_rsp, 0, sizeof(input_chunk_rsp));
            input_chunk_req.version = CONFINFER_PROTOCOL_VERSION;
            input_chunk_req.model_id = req->model_id;
            input_chunk_req.partition_id = req->partition_id;
            input_chunk_req.chunk_offset = offset;
            input_chunk_req.chunk_size = chunk_size;
            input_chunk_req.total_input_bytes = (uint32_t)input_blob_size;
            input_chunk_req.flags = req->flags;
            res = invoke_exec_partition_input_chunk(client, &input_chunk_req,
                                                    input_cursor + offset, chunk_size,
                                                    &input_chunk_rsp, err_origin);
            if (res != TEEC_SUCCESS) {
                goto exec_chunk_fail;
            }
            if (input_chunk_rsp.version != CONFINFER_PROTOCOL_VERSION ||
                input_chunk_rsp.model_id != req->model_id ||
                input_chunk_rsp.partition_id != req->partition_id ||
                input_chunk_rsp.status != CONFINFER_STATUS_OK ||
                input_chunk_rsp.accepted_bytes != chunk_size ||
                input_chunk_rsp.next_offset != offset + chunk_size) {
                res = TEEC_ERROR_GENERIC;
                goto exec_chunk_fail;
            }
            offset += chunk_size;
        }

        memset(&run_req, 0, sizeof(run_req));
        memset(rsp, 0, sizeof(*rsp));
        run_req.version = CONFINFER_PROTOCOL_VERSION;
        run_req.model_id = req->model_id;
        run_req.partition_id = req->partition_id;
        run_req.input_count = req->input_count;
        run_req.output_count = req->output_count;
        run_req.total_input_bytes = (uint32_t)input_blob_size;
        run_req.total_output_bytes = (uint32_t)output_blob_size;
        run_req.flags = req->flags;
        res = invoke_exec_partition_run(client, &run_req, rsp, err_origin);
        if (res != TEEC_SUCCESS) {
            goto exec_chunk_fail;
        }

        offset = 0;
        while (offset < output_blob_size) {
            const uint32_t chunk_size =
                (uint32_t)(((output_blob_size - offset) > CONFINFER_TEEC_CHUNK_BYTES) ?
                           CONFINFER_TEEC_CHUNK_BYTES : (output_blob_size - offset));
            memset(&output_chunk_req, 0, sizeof(output_chunk_req));
            memset(&output_chunk_rsp, 0, sizeof(output_chunk_rsp));
            output_chunk_req.version = CONFINFER_PROTOCOL_VERSION;
            output_chunk_req.model_id = req->model_id;
            output_chunk_req.partition_id = req->partition_id;
            output_chunk_req.chunk_offset = offset;
            output_chunk_req.chunk_size = chunk_size;
            output_chunk_req.total_output_bytes = (uint32_t)output_blob_size;
            output_chunk_req.flags = req->flags;
            res = invoke_exec_partition_output_chunk(client, &output_chunk_req,
                                                     output_cursor + offset, chunk_size,
                                                     &output_chunk_rsp, err_origin);
            if (res != TEEC_SUCCESS) {
                goto exec_chunk_fail;
            }
            if (output_chunk_rsp.version != CONFINFER_PROTOCOL_VERSION ||
                output_chunk_rsp.model_id != req->model_id ||
                output_chunk_rsp.partition_id != req->partition_id ||
                output_chunk_rsp.status != CONFINFER_STATUS_OK ||
                output_chunk_rsp.copied_bytes != chunk_size ||
                output_chunk_rsp.next_offset != offset + chunk_size) {
                res = TEEC_ERROR_GENERIC;
                goto exec_chunk_fail;
            }
            offset += chunk_size;
        }

        memset(&end_req, 0, sizeof(end_req));
        end_req.version = CONFINFER_PROTOCOL_VERSION;
        end_req.model_id = req->model_id;
        end_req.partition_id = req->partition_id;
        end_req.flags = req->flags;
        err_origin_local = 0;
        invoke_exec_partition_end(client, &end_req, &err_origin_local);
        return TEEC_SUCCESS;
exec_chunk_fail:
        memset(&end_req, 0, sizeof(end_req));
        end_req.version = CONFINFER_PROTOCOL_VERSION;
        end_req.model_id = req->model_id;
        end_req.partition_id = req->partition_id;
        end_req.flags = req->flags;
        err_origin_local = 0;
        invoke_exec_partition_end(client, &end_req, &err_origin_local);
        return res;
    }

    mem0.buffer = (void *)req;
    mem0.size = sizeof(*req);
    mem1.buffer = (void *)input_blob;
    mem1.size = input_blob_size;
    mem2.buffer = output_blob;
    mem2.size = output_blob_size;
    mem3.buffer = rsp;
    mem3.size = sizeof(*rsp);

    return confinfer_teec_invoke_command(client,
                                         TA_CONFINFER_CMD_EXEC_PARTITION,
                                         TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,
                                                          TEEC_MEMREF_TEMP_INPUT,
                                                          TEEC_MEMREF_TEMP_INOUT,
                                                          TEEC_MEMREF_TEMP_OUTPUT),
                                         &mem0, &mem1, &mem2, &mem3, err_origin);
}

TEEC_Result confinfer_teec_unload_model(confinfer_teec_client_t *client,
                                        const confinfer_unload_model_req_t *req,
                                        confinfer_unload_model_rsp_t *rsp,
                                        uint32_t *err_origin)
{
    confinfer_teec_memref_t mem0;
    confinfer_teec_memref_t mem1;

    /*
     * 当前 unload 只携带控制元数据 因此保持为一次调用
     * 它不应继承 image 输入输出的分块传输复杂度
     */
    if (!client || !client->is_open || !req || !rsp || !err_origin) {
        return TEEC_ERROR_BAD_PARAMETERS;
    }
    if (req->version != CONFINFER_PROTOCOL_VERSION) {
        return TEEC_ERROR_BAD_PARAMETERS;
    }

    mem0.buffer = (void *)req;
    mem0.size = sizeof(*req);
    mem1.buffer = rsp;
    mem1.size = sizeof(*rsp);

    return confinfer_teec_invoke_command(client,
                                         TA_CONFINFER_CMD_UNLOAD_MODEL,
                                         TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,
                                                          TEEC_MEMREF_TEMP_OUTPUT,
                                                          TEEC_NONE,
                                                          TEEC_NONE),
                                         &mem0, &mem1, NULL, NULL, err_origin);
}
