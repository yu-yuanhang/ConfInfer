#include "confinfer_host.h"

#include <stdlib.h>
#include <string.h>

static void fill_tmpref(TEEC_Parameter *param, confinfer_teec_memref_t *mem)
{
    if (!param || !mem) {
        return;
    }
    param->tmpref.buffer = mem->buffer;
    param->tmpref.size = mem->size;
}

// 计算 执行的数据面描述信息 size
// 一个 confinfer_partition_data_req_t
// 后面依次紧跟着:
// inputs / outputs / internals / layer_io_desc / input_refs / output_refs / param_refs
// 对于 EXEC_PARTITION，后面还会继续拼接 input_blob / output_blob
static size_t partition_data_blob_size(size_t input_count,
                                       size_t output_count,
                                       size_t internal_count,
                                       size_t layer_io_count,
                                       size_t input_ref_count,
                                       size_t output_ref_count,
                                       size_t param_ref_count,
                                       size_t input_blob_size,
                                       size_t output_blob_size)
{
    return sizeof(confinfer_partition_data_req_t) +
           (input_count + output_count + internal_count) * sizeof(confinfer_value_desc_t) +
           layer_io_count * sizeof(confinfer_layer_io_desc_t) +
           (input_ref_count + output_ref_count) * sizeof(confinfer_layer_value_ref_t) +
           param_ref_count * sizeof(confinfer_layer_param_ref_t) +
           input_blob_size + output_blob_size;
}

static int pack_partition_data_blob(uint8_t *blob,
                                    size_t blob_size,
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
                                    size_t output_blob_size)
{
    size_t expected_size = partition_data_blob_size(input_count, output_count,
                                                    internal_count, layer_io_count,
                                                    input_ref_count, output_ref_count,
                                                    param_ref_count,
                                                    input_blob_size, output_blob_size);
    uint8_t *cursor = blob;

    if (!blob || !data_req || blob_size != expected_size) {
        return -1;
    }
    if (data_req->input_count != input_count ||
        data_req->output_count != output_count ||
        data_req->internal_count != internal_count ||
        data_req->layer_io_count != layer_io_count ||
        data_req->input_ref_count != input_ref_count ||
        data_req->output_ref_count != output_ref_count ||
        data_req->param_ref_count != param_ref_count) {
        return -1;
    }
    if ((input_count > 0 && !inputs) ||
        (output_count > 0 && !outputs) ||
        (internal_count > 0 && !internals) ||
        (layer_io_count > 0 && !layer_ios) ||
        (input_ref_count > 0 && !input_refs) ||
        (output_ref_count > 0 && !output_refs) ||
        (param_ref_count > 0 && !param_refs) ||
        (input_blob_size > 0 && !input_blob) ||
        (output_blob_size > 0 && !output_blob)) {
        return -1;
    }

    memcpy(cursor, data_req, sizeof(*data_req));
    cursor += sizeof(*data_req);
    if (input_count > 0) {
        memcpy(cursor, inputs, input_count * sizeof(*inputs));
        cursor += input_count * sizeof(*inputs);
    }
    if (output_count > 0) {
        memcpy(cursor, outputs, output_count * sizeof(*outputs));
        cursor += output_count * sizeof(*outputs);
    }
    if (internal_count > 0) {
        memcpy(cursor, internals, internal_count * sizeof(*internals));
        cursor += internal_count * sizeof(*internals);
    }
    if (layer_io_count > 0) {
        memcpy(cursor, layer_ios, layer_io_count * sizeof(*layer_ios));
        cursor += layer_io_count * sizeof(*layer_ios);
    }
    if (input_ref_count > 0) {
        memcpy(cursor, input_refs, input_ref_count * sizeof(*input_refs));
        cursor += input_ref_count * sizeof(*input_refs);
    }
    if (output_ref_count > 0) {
        memcpy(cursor, output_refs, output_ref_count * sizeof(*output_refs));
        cursor += output_ref_count * sizeof(*output_refs);
    }
    if (param_ref_count > 0) {
        memcpy(cursor, param_refs, param_ref_count * sizeof(*param_refs));
        cursor += param_ref_count * sizeof(*param_refs);
    }
    if (input_blob_size > 0) {
        memcpy(cursor, input_blob, input_blob_size);
        cursor += input_blob_size;
    }
    if (output_blob_size > 0) {
        memcpy(cursor, output_blob, output_blob_size);
    }
    return 0;
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

TEEC_Result confinfer_teec_register_model(confinfer_teec_client_t *client,
                                          const confinfer_model_desc_t *desc,
                                          confinfer_model_rsp_t *rsp,
                                          uint32_t *err_origin)
{
    confinfer_teec_memref_t mem0;
    confinfer_teec_memref_t mem1;

    if (!client || !client->is_open || !desc || !rsp || !err_origin) {
        return TEEC_ERROR_BAD_PARAMETERS;
    }
    if (desc->version != CONFINFER_PROTOCOL_VERSION) {
        return TEEC_ERROR_BAD_PARAMETERS;
    }

    mem0.buffer = (void *)desc;
    mem0.size = sizeof(*desc);
    mem1.buffer = rsp;
    mem1.size = sizeof(*rsp);

    return confinfer_teec_invoke_command(client,
                                         TA_CONFINFER_CMD_REGISTER_MODEL,
                                         TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,
                                                          TEEC_MEMREF_TEMP_OUTPUT,
                                                          TEEC_NONE,
                                                          TEEC_NONE),
                                         &mem0, &mem1, NULL, NULL, err_origin);
}

TEEC_Result confinfer_teec_load_params(confinfer_teec_client_t *client,
                                       const confinfer_load_params_req_t *req,
                                       const confinfer_param_desc_t *param_descs,
                                       size_t param_count,
                                       const void *param_blob,
                                       size_t param_blob_size,
                                       confinfer_load_params_rsp_t *rsp,
                                       uint32_t *err_origin)
{
    confinfer_teec_memref_t mem0;
    confinfer_teec_memref_t mem1;
    confinfer_teec_memref_t mem2;
    confinfer_teec_memref_t mem3;

    if (!client || !client->is_open || !req || !rsp || !err_origin) {
        return TEEC_ERROR_BAD_PARAMETERS;
    }
    if (req->version != CONFINFER_PROTOCOL_VERSION ||
        req->param_count != param_count ||
        req->total_param_bytes != param_blob_size) {
        return TEEC_ERROR_BAD_PARAMETERS;
    }
    if ((param_count > 0 && !param_descs) ||
        (param_blob_size > 0 && !param_blob)) {
        return TEEC_ERROR_BAD_PARAMETERS;
    }

    mem0.buffer = (void *)req;
    mem0.size = sizeof(*req);
    mem1.buffer = (void *)param_descs;
    mem1.size = param_count * sizeof(*param_descs);
    mem2.buffer = (void *)param_blob;
    mem2.size = param_blob_size;
    mem3.buffer = rsp;
    mem3.size = sizeof(*rsp);

    return confinfer_teec_invoke_command(client,
                                         TA_CONFINFER_CMD_LOAD_PARAMS,
                                         TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,
                                                          TEEC_MEMREF_TEMP_INPUT,
                                                          TEEC_MEMREF_TEMP_INPUT,
                                                          TEEC_MEMREF_TEMP_OUTPUT),
                                         &mem0, &mem1, &mem2, &mem3, err_origin);
}

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
                                              uint32_t *err_origin)
{
    TEEC_Result res;
    size_t layers_size = 0;
    size_t data_blob_size = 0;
    confinfer_teec_memref_t mem0;
    confinfer_teec_memref_t mem1;
    confinfer_teec_memref_t mem2;
    confinfer_teec_memref_t mem3;
    uint8_t *data_blob = NULL;

    if (!client || !client->is_open || !req || !data_req || !rsp || !err_origin) {
        return TEEC_ERROR_BAD_PARAMETERS;
    }
    if (req->version != CONFINFER_PROTOCOL_VERSION ||
        req->layer_count != (uint32_t)layer_count ||
        data_req->version != CONFINFER_PROTOCOL_VERSION ||
        data_req->input_count != input_count ||
        data_req->output_count != output_count ||
        data_req->internal_count != internal_count ||
        data_req->layer_io_count != layer_io_count ||
        data_req->input_ref_count != input_ref_count ||
        data_req->output_ref_count != output_ref_count ||
        data_req->param_ref_count != param_ref_count) {
        return TEEC_ERROR_BAD_PARAMETERS;
    }
    if (layer_count > 0 && !layers) {
        return TEEC_ERROR_BAD_PARAMETERS;
    }
    if (layer_attr_blob_size > 0 && !layer_attr_blob) {
        return TEEC_ERROR_BAD_PARAMETERS;
    }

    layers_size = layer_count * sizeof(*layers) + layer_attr_blob_size;
    data_blob_size = partition_data_blob_size(input_count, output_count,
                                              internal_count, layer_io_count,
                                              input_ref_count, output_ref_count,
                                              param_ref_count, 0, 0);
    data_blob = (uint8_t *)malloc(data_blob_size);
    if (!data_blob) {
        return TEEC_ERROR_OUT_OF_MEMORY;
    }
    if (0 != pack_partition_data_blob(data_blob, data_blob_size,
                                      data_req, inputs, input_count,
                                      outputs, output_count,
                                      internals, internal_count,
                                      layer_ios, layer_io_count,
                                      input_refs, input_ref_count,
                                      output_refs, output_ref_count,
                                      param_refs, param_ref_count,
                                      NULL, 0, NULL, 0)) {
        free(data_blob);
        return TEEC_ERROR_BAD_PARAMETERS;
    }

    mem0.buffer = (void *)req;
    mem0.size = sizeof(*req);
    mem1.buffer = malloc(layers_size);
    if (!mem1.buffer) {
        free(data_blob);
        return TEEC_ERROR_OUT_OF_MEMORY;
    }
    memcpy(mem1.buffer, layers, layer_count * sizeof(*layers));
    if (layer_attr_blob_size > 0 && layer_attr_blob) {
        memcpy((uint8_t *)mem1.buffer + layer_count * sizeof(*layers),
               layer_attr_blob, layer_attr_blob_size);
    }
    mem1.size = layers_size;
    mem2.buffer = data_blob;
    mem2.size = data_blob_size;
    mem3.buffer = rsp;
    mem3.size = sizeof(*rsp);

    res = confinfer_teec_invoke_command(client,
                                        TA_CONFINFER_CMD_REGISTER_PARTITION,
                                        TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,
                                                         TEEC_MEMREF_TEMP_INPUT,
                                                         TEEC_MEMREF_TEMP_INPUT,
                                                         TEEC_MEMREF_TEMP_OUTPUT),
                                        &mem0, &mem1, &mem2, &mem3, err_origin);
    free(mem1.buffer);
    free(data_blob);
    return res;
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
    if (req->version != CONFINFER_PROTOCOL_VERSION ||
        req->model_id == CONFINFER_INVALID_MODEL_ID) {
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
                                          uint32_t *err_origin)
{
    TEEC_Result res;
    size_t layers_size = 0;
    size_t data_blob_size = 0;
    size_t meta_blob_size = 0;
    confinfer_teec_memref_t mem0;
    confinfer_teec_memref_t mem1;
    confinfer_teec_memref_t mem2;
    confinfer_teec_memref_t mem3;
    uint8_t *data_blob = NULL;

    if (!client || !client->is_open || !req || !data_req || !rsp || !err_origin) {
        return TEEC_ERROR_BAD_PARAMETERS;
    }
    if (req->version != CONFINFER_PROTOCOL_VERSION) {
        return TEEC_ERROR_BAD_PARAMETERS;
    }
    if (req->layer_count != (uint32_t)layer_count) {
        return TEEC_ERROR_BAD_PARAMETERS;
    }
    if (layer_count > 0 && !layers) {
        return TEEC_ERROR_BAD_PARAMETERS;
    }
    if (layer_attr_blob_size > 0 && !layer_attr_blob) {
        return TEEC_ERROR_BAD_PARAMETERS;
    }
    if (data_req->version != CONFINFER_PROTOCOL_VERSION) {
        return TEEC_ERROR_BAD_PARAMETERS;
    }
    if (data_req->input_count != input_count ||
        data_req->output_count != output_count ||
        data_req->internal_count != internal_count ||
        data_req->layer_io_count != layer_io_count ||
        data_req->input_ref_count != input_ref_count ||
        data_req->output_ref_count != output_ref_count ||
        data_req->param_ref_count != param_ref_count ||
        data_req->total_input_bytes != input_blob_size ||
        data_req->total_output_bytes != output_blob_size) {
        return TEEC_ERROR_BAD_PARAMETERS;
    }
    if ((input_blob_size > 0 && !input_blob) ||
        (output_blob_size > 0 && !output_blob)) {
        return TEEC_ERROR_BAD_PARAMETERS;
    }

    layers_size = layer_count * sizeof(*layers) + layer_attr_blob_size;
    meta_blob_size = partition_data_blob_size(input_count, output_count,
                                              internal_count, layer_io_count,
                                              input_ref_count, output_ref_count,
                                              param_ref_count, 0, 0);
    data_blob_size = partition_data_blob_size(input_count, output_count,
                                              internal_count, layer_io_count,
                                              input_ref_count, output_ref_count,
                                              param_ref_count,
                                              input_blob_size, output_blob_size);
    data_blob = (uint8_t *)malloc(data_blob_size);
    if (!data_blob) {
        return TEEC_ERROR_OUT_OF_MEMORY;
    }
    if (0 != pack_partition_data_blob(data_blob, data_blob_size,
                                      data_req, inputs, input_count,
                                      outputs, output_count,
                                      internals, internal_count,
                                      layer_ios, layer_io_count,
                                      input_refs, input_ref_count,
                                      output_refs, output_ref_count,
                                      param_refs, param_ref_count,
                                      input_blob, input_blob_size,
                                      output_blob, output_blob_size)) {
        free(data_blob);
        return TEEC_ERROR_BAD_PARAMETERS;
    }

    mem0.buffer = (void *)req;
    mem0.size = sizeof(*req);
    mem1.buffer = malloc(layers_size);
    if (!mem1.buffer) {
        free(data_blob);
        return TEEC_ERROR_OUT_OF_MEMORY;
    }
    memcpy(mem1.buffer, layers, layer_count * sizeof(*layers));
    if (layer_attr_blob_size > 0 && layer_attr_blob) {
        memcpy((uint8_t *)mem1.buffer + layer_count * sizeof(*layers),
               layer_attr_blob, layer_attr_blob_size);
    }
    mem1.size = layers_size;
    mem2.buffer = data_blob;
    mem2.size = data_blob_size;
    mem3.buffer = rsp;
    mem3.size = sizeof(*rsp);

    res = confinfer_teec_invoke_command(client,
                                        CONFINFER_CMD_EXEC_PARTITION,
                                        TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,
                                                         TEEC_MEMREF_TEMP_INPUT,
                                                         TEEC_MEMREF_TEMP_INOUT,
                                                         TEEC_MEMREF_TEMP_OUTPUT),
                                        &mem0,
                                        &mem1,
                                        &mem2,
                                        &mem3,
                                        err_origin);
    // mem2 在 EXEC_PARTITION 路径下是 INOUT:
    // 前半段是描述区，后半段携带本次执行的输入和输出字节流。
    if (res == TEEC_SUCCESS && output_blob_size > 0) {
        memcpy(output_blob, data_blob + meta_blob_size + input_blob_size, output_blob_size);
    }
    free(mem1.buffer);
    free(data_blob);
    if (res != TEEC_SUCCESS) {
        return res;
    }
    if (mem3.size < sizeof(*rsp)) {
        return TEEC_ERROR_SHORT_BUFFER;
    }

    return TEEC_SUCCESS;
}

void confinfer_teec_close(confinfer_teec_client_t *client)
{
    if (!client) {
        return;
    }

    if (client->is_open) {
        TEEC_CloseSession(&client->sess);
        TEEC_FinalizeContext(&client->ctx);
    }

    memset(client, 0, sizeof(*client));
}
