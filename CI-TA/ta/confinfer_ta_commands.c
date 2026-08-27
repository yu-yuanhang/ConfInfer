#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>

#include <conf_infer_ta.h>
#include <confinfer_ta_backend.h>
#include <confinfer_ta_commands.h>
#include <confinfer_ta_runtime.h>

#define CONFINFER_TA_MAX_STAGED_IMAGE_BYTES (2u * 1024u * 1024u)
#define CONFINFER_TA_MAX_STAGED_EXEC_BYTES  (2u * 1024u * 1024u)

static confinfer_ta_session_t *as_session(void *sess_ctx)
{
    return (confinfer_ta_session_t *)sess_ctx;
}

static ta_model_t *find_ready_model(confinfer_model_id_t model_id)
{
    ta_model_t *model = ta_model_find(model_id);

    if (!model || !model->is_registered) {
        return NULL;
    }
    return model;
}

static void reset_prepare_image_upload(confinfer_ta_session_t *session)
{
    if (!session) {
        return;
    }
    if (session->prepare_image_upload.buffer) {
        TEE_Free(session->prepare_image_upload.buffer);
    }
    TEE_MemFill(&session->prepare_image_upload, 0, sizeof(session->prepare_image_upload));
    session->prepare_image_upload.model_id = CONFINFER_INVALID_MODEL_ID;
}

static void reset_exec_partition_upload(confinfer_ta_session_t *session)
{
    if (!session) {
        return;
    }
    if (session->exec_partition_upload.input_buffer) {
        TEE_Free(session->exec_partition_upload.input_buffer);
    }
    if (session->exec_partition_upload.output_buffer) {
        TEE_Free(session->exec_partition_upload.output_buffer);
    }
    TEE_MemFill(&session->exec_partition_upload, 0, sizeof(session->exec_partition_upload));
    session->exec_partition_upload.model_id = CONFINFER_INVALID_MODEL_ID;
    session->exec_partition_upload.partition_id = CONFINFER_INVALID_PARTITION_ID;
}

static TEE_Result load_partition_inputs(ta_partition_t *part,
                                        const uint8_t *input_blob,
                                        uint32_t input_blob_size)
{
    uint32_t i = 0;
    const uint8_t *cursor = input_blob;
    uint32_t seen = 0;

    if (!part || (input_blob_size > 0 && !input_blob)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    for (i = 0; i < part->value_count; ++i) {
        ta_value_t *value = &part->values[i];
        uint32_t byte_size = 0;

        if (!(value->role_flags & TA_VALUE_ROLE_INPUT)) {
            continue;
        }
        if (!value->data.ptr) {
            return TEE_ERROR_BAD_STATE;
        }
        byte_size = value->data.byte_size;
        if (byte_size > input_blob_size) {
            return TEE_ERROR_BAD_PARAMETERS;
        }
        if (byte_size > 0) {
            TEE_MemMove(value->data.ptr, cursor, byte_size);
            cursor += byte_size;
            input_blob_size -= byte_size;
        }
        seen += 1;
    }
    return (input_blob_size == 0 && seen == part->input_count) ?
        TEE_SUCCESS : TEE_ERROR_BAD_PARAMETERS;
}

static TEE_Result store_partition_outputs(const ta_partition_t *part,
                                          uint8_t *output_blob,
                                          uint32_t output_blob_size)
{
    uint32_t i = 0;
    uint8_t *cursor = output_blob;
    uint32_t seen = 0;

    if (!part || (output_blob_size > 0 && !output_blob)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    for (i = 0; i < part->value_count; ++i) {
        ta_value_t *value = &part->values[i];
        uint32_t byte_size = 0;

        if (!(value->role_flags & TA_VALUE_ROLE_OUTPUT)) {
            continue;
        }
        if (!value->data.ptr) {
            return TEE_ERROR_BAD_STATE;
        }
        byte_size = value->data.byte_size;
        if (byte_size > output_blob_size) {
            return TEE_ERROR_BAD_PARAMETERS;
        }
        if (byte_size > 0) {
            TEE_MemMove(cursor, value->data.ptr, byte_size);
            cursor += byte_size;
            output_blob_size -= byte_size;
        }
        seen += 1;
    }
    return (output_blob_size == 0 && seen == part->output_count) ?
        TEE_SUCCESS : TEE_ERROR_BAD_PARAMETERS;
}

static TEE_Result execute_partition_once(confinfer_model_id_t model_id,
                                         confinfer_partition_id_t partition_id,
                                         uint32_t input_count,
                                         uint32_t output_count,
                                         const uint8_t *input_blob,
                                         uint32_t input_bytes,
                                         uint8_t *output_blob,
                                         uint32_t output_bytes)
{
    ta_model_t *model = NULL;
    ta_partition_t *part = NULL;
    TEE_Result res = TEE_SUCCESS;

    model = find_ready_model(model_id);
    if (!model) {
        return TEE_ERROR_ITEM_NOT_FOUND;
    }

    part = ta_model_find_partition(model, partition_id);
    if (!part) {
        return TEE_ERROR_ITEM_NOT_FOUND;
    }
    if (part->input_count != input_count || part->output_count != output_count) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    res = load_partition_inputs(part, input_blob, input_bytes);
    if (res == TEE_SUCCESS) {
        res = ta_backend_execute_partition(ta_backend_default(), model, part);
    }
    if (res == TEE_SUCCESS) {
        res = store_partition_outputs(part, output_blob, output_bytes);
    }
    return res;
}

TEE_Result confinfer_ta_prepare_model_image(void *sess_ctx,
                                            uint32_t param_types,
                                            TEE_Param params[4])
{
    const uint32_t exp_param_types =
        TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                        TEE_PARAM_TYPE_MEMREF_INPUT,
                        TEE_PARAM_TYPE_MEMREF_OUTPUT,
                        TEE_PARAM_TYPE_NONE);
    const confinfer_prepare_model_image_req_t *req = NULL;
    const void *image_data = NULL;
    confinfer_prepare_model_image_rsp_t *rsp = NULL;
    ta_model_t *model = NULL;
    TEE_Result res = TEE_SUCCESS;

    (void)sess_ctx;

    if (param_types != exp_param_types ||
        params[0].memref.size != sizeof(*req) ||
        params[2].memref.size < sizeof(*rsp)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    req = (const confinfer_prepare_model_image_req_t *)params[0].memref.buffer;
    image_data = params[1].memref.buffer;
    rsp = (confinfer_prepare_model_image_rsp_t *)params[2].memref.buffer;
    if (!req || !rsp ||
        req->version != CONFINFER_PROTOCOL_VERSION ||
        req->model_id == CONFINFER_INVALID_MODEL_ID ||
        req->image_size != params[1].memref.size ||
        (req->image_size > 0 && !image_data)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    TEE_MemFill(rsp, 0, sizeof(*rsp));
    rsp->version = CONFINFER_PROTOCOL_VERSION;
    rsp->model_id = req->model_id;

    res = ta_model_ensure(req->model_id, &model);
    if (res == TEE_SUCCESS) {
        res = ta_model_load_image(model, image_data, req->image_size);
    }
    rsp->status = (res == TEE_SUCCESS) ? CONFINFER_STATUS_OK : CONFINFER_STATUS_BAD_REQUEST;
    rsp->loaded_image_size = (res == TEE_SUCCESS) ? req->image_size : 0;
    params[2].memref.size = sizeof(*rsp);
    return res;
}

TEE_Result confinfer_ta_prepare_model_image_begin(void *sess_ctx,
                                                  uint32_t param_types,
                                                  TEE_Param params[4])
{
    confinfer_ta_session_t *session = as_session(sess_ctx);
    const uint32_t exp_param_types =
        TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                        TEE_PARAM_TYPE_MEMREF_OUTPUT,
                        TEE_PARAM_TYPE_NONE,
                        TEE_PARAM_TYPE_NONE);
    const confinfer_prepare_model_image_begin_req_t *req = NULL;
    confinfer_prepare_model_image_begin_rsp_t *rsp = NULL;

    if (!session || param_types != exp_param_types ||
        params[0].memref.size != sizeof(*req) ||
        params[1].memref.size < sizeof(*rsp)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    req = (const confinfer_prepare_model_image_begin_req_t *)params[0].memref.buffer;
    rsp = (confinfer_prepare_model_image_begin_rsp_t *)params[1].memref.buffer;
    if (!req || !rsp ||
        req->version != CONFINFER_PROTOCOL_VERSION ||
        req->model_id == CONFINFER_INVALID_MODEL_ID ||
        req->total_image_size > CONFINFER_TA_MAX_STAGED_IMAGE_BYTES) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    reset_prepare_image_upload(session);
    if (req->total_image_size > 0) {
        session->prepare_image_upload.buffer =
            TEE_Malloc(req->total_image_size, TEE_MALLOC_FILL_ZERO);
        if (!session->prepare_image_upload.buffer) {
            return TEE_ERROR_OUT_OF_MEMORY;
        }
    }
    session->prepare_image_upload.model_id = req->model_id;
    session->prepare_image_upload.total_size = req->total_image_size;

    TEE_MemFill(rsp, 0, sizeof(*rsp));
    rsp->version = CONFINFER_PROTOCOL_VERSION;
    rsp->status = CONFINFER_STATUS_OK;
    rsp->model_id = req->model_id;
    params[1].memref.size = sizeof(*rsp);
    return TEE_SUCCESS;
}

TEE_Result confinfer_ta_prepare_model_image_chunk(void *sess_ctx,
                                                  uint32_t param_types,
                                                  TEE_Param params[4])
{
    confinfer_ta_session_t *session = as_session(sess_ctx);
    confinfer_prepare_image_upload_t *upload = NULL;
    const uint32_t exp_param_types =
        TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                        TEE_PARAM_TYPE_MEMREF_INPUT,
                        TEE_PARAM_TYPE_MEMREF_OUTPUT,
                        TEE_PARAM_TYPE_NONE);
    const confinfer_prepare_model_image_chunk_req_t *req = NULL;
    const uint8_t *chunk_data = NULL;
    confinfer_prepare_model_image_chunk_rsp_t *rsp = NULL;

    if (!session) {
        return TEE_ERROR_BAD_STATE;
    }
    upload = &session->prepare_image_upload;
    if (param_types != exp_param_types ||
        params[0].memref.size != sizeof(*req) ||
        params[2].memref.size < sizeof(*rsp)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    req = (const confinfer_prepare_model_image_chunk_req_t *)params[0].memref.buffer;
    chunk_data = (const uint8_t *)params[1].memref.buffer;
    rsp = (confinfer_prepare_model_image_chunk_rsp_t *)params[2].memref.buffer;
    if (!req || !rsp ||
        req->version != CONFINFER_PROTOCOL_VERSION ||
        req->model_id != upload->model_id ||
        req->total_image_size != upload->total_size ||
        req->chunk_offset != upload->received_size ||
        req->chunk_size != params[1].memref.size ||
        req->chunk_offset > upload->total_size ||
        req->chunk_size > upload->total_size - req->chunk_offset ||
        (req->chunk_size > 0 && !chunk_data)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    if (req->chunk_size > 0) {
        TEE_MemMove(upload->buffer + req->chunk_offset, chunk_data, req->chunk_size);
    }
    upload->received_size += req->chunk_size;

    TEE_MemFill(rsp, 0, sizeof(*rsp));
    rsp->version = CONFINFER_PROTOCOL_VERSION;
    rsp->status = CONFINFER_STATUS_OK;
    rsp->model_id = req->model_id;
    rsp->next_offset = upload->received_size;
    rsp->accepted_bytes = req->chunk_size;
    params[2].memref.size = sizeof(*rsp);
    return TEE_SUCCESS;
}

TEE_Result confinfer_ta_prepare_model_image_end(void *sess_ctx,
                                                uint32_t param_types,
                                                TEE_Param params[4])
{
    confinfer_ta_session_t *session = as_session(sess_ctx);
    confinfer_prepare_image_upload_t *upload = NULL;
    const uint32_t exp_param_types =
        TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                        TEE_PARAM_TYPE_MEMREF_OUTPUT,
                        TEE_PARAM_TYPE_NONE,
                        TEE_PARAM_TYPE_NONE);
    const confinfer_prepare_model_image_end_req_t *req = NULL;
    confinfer_prepare_model_image_rsp_t *rsp = NULL;
    ta_model_t *model = NULL;
    TEE_Result res = TEE_SUCCESS;

    if (!session) {
        return TEE_ERROR_BAD_STATE;
    }
    upload = &session->prepare_image_upload;
    if (param_types != exp_param_types ||
        params[0].memref.size != sizeof(*req) ||
        params[1].memref.size < sizeof(*rsp)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    req = (const confinfer_prepare_model_image_end_req_t *)params[0].memref.buffer;
    rsp = (confinfer_prepare_model_image_rsp_t *)params[1].memref.buffer;
    if (!req || !rsp ||
        req->version != CONFINFER_PROTOCOL_VERSION ||
        req->model_id != upload->model_id ||
        req->total_image_size != upload->total_size ||
        upload->received_size != upload->total_size) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    TEE_MemFill(rsp, 0, sizeof(*rsp));
    rsp->version = CONFINFER_PROTOCOL_VERSION;
    rsp->model_id = req->model_id;

    res = ta_model_ensure(req->model_id, &model);
    if (res == TEE_SUCCESS) {
        res = ta_model_load_image(model, upload->buffer, upload->total_size);
    }
    rsp->status = (res == TEE_SUCCESS) ? CONFINFER_STATUS_OK : CONFINFER_STATUS_BAD_REQUEST;
    rsp->loaded_image_size = (res == TEE_SUCCESS) ? req->total_image_size : 0;
    params[1].memref.size = sizeof(*rsp);
    reset_prepare_image_upload(session);
    return res;
}

TEE_Result confinfer_ta_exec_partition(void *sess_ctx,
                                       uint32_t param_types,
                                       TEE_Param params[4])
{
    const uint32_t exp_param_types =
        TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                        TEE_PARAM_TYPE_MEMREF_INPUT,
                        TEE_PARAM_TYPE_MEMREF_INOUT,
                        TEE_PARAM_TYPE_MEMREF_OUTPUT);
    const confinfer_exec_partition_req_t *req = NULL;
    const uint8_t *input_blob = NULL;
    uint8_t *output_blob = NULL;
    confinfer_exec_partition_rsp_t *rsp = NULL;
    TEE_Result res = TEE_SUCCESS;

    (void)sess_ctx;

    if (param_types != exp_param_types ||
        params[0].memref.size != sizeof(*req) ||
        params[3].memref.size < sizeof(*rsp)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    req = (const confinfer_exec_partition_req_t *)params[0].memref.buffer;
    input_blob = (const uint8_t *)params[1].memref.buffer;
    output_blob = (uint8_t *)params[2].memref.buffer;
    rsp = (confinfer_exec_partition_rsp_t *)params[3].memref.buffer;
    if (!req || !rsp ||
        req->version != CONFINFER_PROTOCOL_VERSION ||
        req->model_id == CONFINFER_INVALID_MODEL_ID ||
        req->partition_id == CONFINFER_INVALID_PARTITION_ID ||
        req->input_bytes != params[1].memref.size ||
        req->output_bytes > params[2].memref.size ||
        (req->input_bytes > 0 && !input_blob) ||
        (req->output_bytes > 0 && !output_blob)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    res = execute_partition_once(req->model_id,
                                 req->partition_id,
                                 req->input_count,
                                 req->output_count,
                                 input_blob,
                                 req->input_bytes,
                                 output_blob,
                                 req->output_bytes);

    TEE_MemFill(rsp, 0, sizeof(*rsp));
    rsp->version = CONFINFER_PROTOCOL_VERSION;
    rsp->status = (res == TEE_SUCCESS) ? CONFINFER_STATUS_OK :
                  (res == TEE_ERROR_ITEM_NOT_FOUND ? CONFINFER_STATUS_NOT_FOUND :
                                                    CONFINFER_STATUS_BAD_REQUEST);
    rsp->model_id = req->model_id;
    rsp->partition_id = req->partition_id;
    rsp->consumed_inputs = (res == TEE_SUCCESS) ? req->input_count : 0;
    rsp->produced_outputs = (res == TEE_SUCCESS) ? req->output_count : 0;
    rsp->output_bytes = (res == TEE_SUCCESS) ? req->output_bytes : 0;
    params[3].memref.size = sizeof(*rsp);
    return res;
}

TEE_Result confinfer_ta_exec_partition_begin(void *sess_ctx,
                                             uint32_t param_types,
                                             TEE_Param params[4])
{
    confinfer_ta_session_t *session = as_session(sess_ctx);
    confinfer_exec_partition_upload_t *upload = NULL;
    const uint32_t exp_param_types =
        TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                        TEE_PARAM_TYPE_MEMREF_OUTPUT,
                        TEE_PARAM_TYPE_NONE,
                        TEE_PARAM_TYPE_NONE);
    const confinfer_exec_partition_begin_req_t *req = NULL;
    confinfer_exec_partition_begin_rsp_t *rsp = NULL;

    if (!session) {
        return TEE_ERROR_BAD_STATE;
    }
    upload = &session->exec_partition_upload;
    if (param_types != exp_param_types ||
        params[0].memref.size != sizeof(*req) ||
        params[1].memref.size < sizeof(*rsp)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    req = (const confinfer_exec_partition_begin_req_t *)params[0].memref.buffer;
    rsp = (confinfer_exec_partition_begin_rsp_t *)params[1].memref.buffer;
    if (!req || !rsp ||
        req->version != CONFINFER_PROTOCOL_VERSION ||
        req->model_id == CONFINFER_INVALID_MODEL_ID ||
        req->partition_id == CONFINFER_INVALID_PARTITION_ID ||
        req->total_input_bytes > CONFINFER_TA_MAX_STAGED_EXEC_BYTES ||
        req->total_output_bytes > CONFINFER_TA_MAX_STAGED_EXEC_BYTES) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    reset_exec_partition_upload(session);
    if (req->total_input_bytes > 0) {
        upload->input_buffer = TEE_Malloc(req->total_input_bytes, TEE_MALLOC_FILL_ZERO);
        if (!upload->input_buffer) {
            return TEE_ERROR_OUT_OF_MEMORY;
        }
    }
    if (req->total_output_bytes > 0) {
        upload->output_buffer = TEE_Malloc(req->total_output_bytes, TEE_MALLOC_FILL_ZERO);
        if (!upload->output_buffer) {
            reset_exec_partition_upload(session);
            return TEE_ERROR_OUT_OF_MEMORY;
        }
    }

    upload->model_id = req->model_id;
    upload->partition_id = req->partition_id;
    upload->input_count = req->input_count;
    upload->output_count = req->output_count;
    upload->total_input_bytes = req->total_input_bytes;
    upload->total_output_bytes = req->total_output_bytes;

    TEE_MemFill(rsp, 0, sizeof(*rsp));
    rsp->version = CONFINFER_PROTOCOL_VERSION;
    rsp->status = CONFINFER_STATUS_OK;
    rsp->model_id = req->model_id;
    rsp->partition_id = req->partition_id;
    params[1].memref.size = sizeof(*rsp);
    return TEE_SUCCESS;
}

TEE_Result confinfer_ta_exec_partition_input_chunk(void *sess_ctx,
                                                   uint32_t param_types,
                                                   TEE_Param params[4])
{
    confinfer_ta_session_t *session = as_session(sess_ctx);
    confinfer_exec_partition_upload_t *upload = NULL;
    const uint32_t exp_param_types =
        TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                        TEE_PARAM_TYPE_MEMREF_INPUT,
                        TEE_PARAM_TYPE_MEMREF_OUTPUT,
                        TEE_PARAM_TYPE_NONE);
    const confinfer_exec_partition_input_chunk_req_t *req = NULL;
    const uint8_t *chunk_data = NULL;
    confinfer_exec_partition_input_chunk_rsp_t *rsp = NULL;

    if (!session) {
        return TEE_ERROR_BAD_STATE;
    }
    upload = &session->exec_partition_upload;
    if (param_types != exp_param_types ||
        params[0].memref.size != sizeof(*req) ||
        params[2].memref.size < sizeof(*rsp)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    req = (const confinfer_exec_partition_input_chunk_req_t *)params[0].memref.buffer;
    chunk_data = (const uint8_t *)params[1].memref.buffer;
    rsp = (confinfer_exec_partition_input_chunk_rsp_t *)params[2].memref.buffer;
    if (!req || !rsp ||
        req->version != CONFINFER_PROTOCOL_VERSION ||
        req->model_id != upload->model_id ||
        req->partition_id != upload->partition_id ||
        req->total_input_bytes != upload->total_input_bytes ||
        req->chunk_offset != upload->received_input_bytes ||
        req->chunk_size != params[1].memref.size ||
        req->chunk_offset > upload->total_input_bytes ||
        req->chunk_size > upload->total_input_bytes - req->chunk_offset ||
        (req->chunk_size > 0 && !chunk_data)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    if (req->chunk_size > 0) {
        TEE_MemMove(upload->input_buffer + req->chunk_offset, chunk_data, req->chunk_size);
    }
    upload->received_input_bytes += req->chunk_size;

    TEE_MemFill(rsp, 0, sizeof(*rsp));
    rsp->version = CONFINFER_PROTOCOL_VERSION;
    rsp->status = CONFINFER_STATUS_OK;
    rsp->model_id = req->model_id;
    rsp->partition_id = req->partition_id;
    rsp->next_offset = upload->received_input_bytes;
    rsp->accepted_bytes = req->chunk_size;
    params[2].memref.size = sizeof(*rsp);
    return TEE_SUCCESS;
}

TEE_Result confinfer_ta_exec_partition_run(void *sess_ctx,
                                           uint32_t param_types,
                                           TEE_Param params[4])
{
    confinfer_ta_session_t *session = as_session(sess_ctx);
    confinfer_exec_partition_upload_t *upload = NULL;
    const uint32_t exp_param_types =
        TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                        TEE_PARAM_TYPE_MEMREF_OUTPUT,
                        TEE_PARAM_TYPE_NONE,
                        TEE_PARAM_TYPE_NONE);
    const confinfer_exec_partition_run_req_t *req = NULL;
    confinfer_exec_partition_rsp_t *rsp = NULL;
    TEE_Result res = TEE_SUCCESS;

    if (!session) {
        return TEE_ERROR_BAD_STATE;
    }
    upload = &session->exec_partition_upload;
    if (param_types != exp_param_types ||
        params[0].memref.size != sizeof(*req) ||
        params[1].memref.size < sizeof(*rsp)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    req = (const confinfer_exec_partition_run_req_t *)params[0].memref.buffer;
    rsp = (confinfer_exec_partition_rsp_t *)params[1].memref.buffer;
    if (!req || !rsp ||
        req->version != CONFINFER_PROTOCOL_VERSION ||
        req->model_id != upload->model_id ||
        req->partition_id != upload->partition_id ||
        req->input_count != upload->input_count ||
        req->output_count != upload->output_count ||
        req->total_input_bytes != upload->total_input_bytes ||
        req->total_output_bytes != upload->total_output_bytes ||
        upload->received_input_bytes != upload->total_input_bytes) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    res = execute_partition_once(req->model_id,
                                 req->partition_id,
                                 req->input_count,
                                 req->output_count,
                                 upload->input_buffer,
                                 upload->total_input_bytes,
                                 upload->output_buffer,
                                 upload->total_output_bytes);

    upload->produced_output_bytes = (res == TEE_SUCCESS) ? upload->total_output_bytes : 0;
    upload->run_completed = (res == TEE_SUCCESS) ? 1u : 0u;

    TEE_MemFill(rsp, 0, sizeof(*rsp));
    rsp->version = CONFINFER_PROTOCOL_VERSION;
    rsp->status = (res == TEE_SUCCESS) ? CONFINFER_STATUS_OK :
                  (res == TEE_ERROR_ITEM_NOT_FOUND ? CONFINFER_STATUS_NOT_FOUND :
                                                    CONFINFER_STATUS_BAD_REQUEST);
    rsp->model_id = req->model_id;
    rsp->partition_id = req->partition_id;
    rsp->consumed_inputs = (res == TEE_SUCCESS) ? req->input_count : 0;
    rsp->produced_outputs = (res == TEE_SUCCESS) ? req->output_count : 0;
    rsp->output_bytes = (res == TEE_SUCCESS) ? upload->total_output_bytes : 0;
    params[1].memref.size = sizeof(*rsp);
    return res;
}

TEE_Result confinfer_ta_exec_partition_output_chunk(void *sess_ctx,
                                                    uint32_t param_types,
                                                    TEE_Param params[4])
{
    confinfer_ta_session_t *session = as_session(sess_ctx);
    confinfer_exec_partition_upload_t *upload = NULL;
    const uint32_t exp_param_types =
        TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                        TEE_PARAM_TYPE_MEMREF_INOUT,
                        TEE_PARAM_TYPE_MEMREF_OUTPUT,
                        TEE_PARAM_TYPE_NONE);
    const confinfer_exec_partition_output_chunk_req_t *req = NULL;
    uint8_t *chunk_data = NULL;
    confinfer_exec_partition_output_chunk_rsp_t *rsp = NULL;

    if (!session) {
        return TEE_ERROR_BAD_STATE;
    }
    upload = &session->exec_partition_upload;
    if (param_types != exp_param_types ||
        params[0].memref.size != sizeof(*req) ||
        params[2].memref.size < sizeof(*rsp)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    req = (const confinfer_exec_partition_output_chunk_req_t *)params[0].memref.buffer;
    chunk_data = (uint8_t *)params[1].memref.buffer;
    rsp = (confinfer_exec_partition_output_chunk_rsp_t *)params[2].memref.buffer;
    if (!req || !rsp ||
        req->version != CONFINFER_PROTOCOL_VERSION ||
        req->model_id != upload->model_id ||
        req->partition_id != upload->partition_id ||
        !upload->run_completed ||
        req->total_output_bytes != upload->produced_output_bytes ||
        req->chunk_offset > upload->produced_output_bytes ||
        req->chunk_size > upload->produced_output_bytes - req->chunk_offset ||
        req->chunk_size > params[1].memref.size ||
        (req->chunk_size > 0 && !chunk_data)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    if (req->chunk_size > 0) {
        TEE_MemMove(chunk_data, upload->output_buffer + req->chunk_offset, req->chunk_size);
    }

    TEE_MemFill(rsp, 0, sizeof(*rsp));
    rsp->version = CONFINFER_PROTOCOL_VERSION;
    rsp->status = CONFINFER_STATUS_OK;
    rsp->model_id = req->model_id;
    rsp->partition_id = req->partition_id;
    rsp->next_offset = req->chunk_offset + req->chunk_size;
    rsp->copied_bytes = req->chunk_size;
    params[1].memref.size = req->chunk_size;
    params[2].memref.size = sizeof(*rsp);
    return TEE_SUCCESS;
}

TEE_Result confinfer_ta_exec_partition_end(void *sess_ctx,
                                           uint32_t param_types,
                                           TEE_Param params[4])
{
    confinfer_ta_session_t *session = as_session(sess_ctx);
    const uint32_t exp_param_types =
        TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                        TEE_PARAM_TYPE_NONE,
                        TEE_PARAM_TYPE_NONE,
                        TEE_PARAM_TYPE_NONE);
    const confinfer_exec_partition_end_req_t *req = NULL;

    if (!session || param_types != exp_param_types ||
        params[0].memref.size != sizeof(*req)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    req = (const confinfer_exec_partition_end_req_t *)params[0].memref.buffer;
    if (!req ||
        req->version != CONFINFER_PROTOCOL_VERSION ||
        req->model_id != session->exec_partition_upload.model_id ||
        req->partition_id != session->exec_partition_upload.partition_id) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    reset_exec_partition_upload(session);
    return TEE_SUCCESS;
}

TEE_Result confinfer_ta_unload_model(void *sess_ctx,
                                     uint32_t param_types,
                                     TEE_Param params[4])
{
    const uint32_t exp_param_types =
        TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                        TEE_PARAM_TYPE_MEMREF_OUTPUT,
                        TEE_PARAM_TYPE_NONE,
                        TEE_PARAM_TYPE_NONE);
    const confinfer_unload_model_req_t *req = NULL;
    confinfer_unload_model_rsp_t *rsp = NULL;
    ta_model_t *model = NULL;

    (void)sess_ctx;

    if (param_types != exp_param_types ||
        params[0].memref.size != sizeof(*req) ||
        params[1].memref.size < sizeof(*rsp)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    req = (const confinfer_unload_model_req_t *)params[0].memref.buffer;
    rsp = (confinfer_unload_model_rsp_t *)params[1].memref.buffer;
    if (!req || !rsp ||
        req->version != CONFINFER_PROTOCOL_VERSION ||
        req->model_id == CONFINFER_INVALID_MODEL_ID) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    TEE_MemFill(rsp, 0, sizeof(*rsp));
    rsp->version = CONFINFER_PROTOCOL_VERSION;
    rsp->model_id = req->model_id;

    model = ta_model_find(req->model_id);
    if (!model) {
        rsp->status = CONFINFER_STATUS_NOT_FOUND;
        params[1].memref.size = sizeof(*rsp);
        return TEE_SUCCESS;
    }

    ta_model_release(model);
    rsp->status = CONFINFER_STATUS_OK;
    params[1].memref.size = sizeof(*rsp);
    return TEE_SUCCESS;
}

TEE_Result confinfer_ta_inc_value(void *sess_ctx, uint32_t param_types, TEE_Param params[4])
{
    (void)sess_ctx;

    if (param_types != TEE_PARAM_TYPES(TEE_PARAM_TYPE_VALUE_INOUT,
                                       TEE_PARAM_TYPE_NONE,
                                       TEE_PARAM_TYPE_NONE,
                                       TEE_PARAM_TYPE_NONE)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    params[0].value.a += 1;
    return TEE_SUCCESS;
}

TEE_Result confinfer_ta_dec_value(void *sess_ctx, uint32_t param_types, TEE_Param params[4])
{
    (void)sess_ctx;

    if (param_types != TEE_PARAM_TYPES(TEE_PARAM_TYPE_VALUE_INOUT,
                                       TEE_PARAM_TYPE_NONE,
                                       TEE_PARAM_TYPE_NONE,
                                       TEE_PARAM_TYPE_NONE)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    params[0].value.a -= 1;
    return TEE_SUCCESS;
}
