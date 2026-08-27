#include "confinfer_host.h"

#include <string.h>

#define CONFINFER_TEEC_CHUNK_BYTES (256u * 1024u)

static void fill_tmpref(TEEC_Parameter *param, confinfer_teec_memref_t *mem)
{
    if (!param || !mem) {
        return;
    }
    param->tmpref.buffer = mem->buffer;
    param->tmpref.size = mem->size;
}

static TEEC_Result invoke_prepare_model_image_begin(
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

static TEEC_Result invoke_prepare_model_image_chunk(
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

static TEEC_Result invoke_prepare_model_image_end(
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

static TEEC_Result invoke_exec_partition_begin(
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

static TEEC_Result invoke_exec_partition_input_chunk(
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

static TEEC_Result invoke_exec_partition_run(
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

static TEEC_Result invoke_exec_partition_output_chunk(
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

static TEEC_Result invoke_exec_partition_end(
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

TEEC_Result confinfer_teec_open(confinfer_teec_client_t *client,
                                uint32_t *err_origin)
{
    TEEC_Result res;
    TEEC_UUID uuid = TA_CONFINFER_UUID;

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

void confinfer_teec_close(confinfer_teec_client_t *client)
{
    if (!client || !client->is_open) {
        return;
    }

    TEEC_CloseSession(&client->sess);
    TEEC_FinalizeContext(&client->ctx);
    memset(client, 0, sizeof(*client));
}
