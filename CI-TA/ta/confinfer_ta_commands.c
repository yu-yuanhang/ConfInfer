#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>

#include <conf_infer_ta.h>
#include <confinfer_ta_backend.h>
#include <confinfer_ta_commands.h>
#include <confinfer_ta_runtime.h>
#include <pta_span.h>

#define CONFINFER_TA_MAX_STAGED_IMAGE_BYTES (2u * 1024u * 1024u)
#define CONFINFER_TA_MAX_STAGED_EXEC_BYTES  (2u * 1024u * 1024u)

static confinfer_ta_session_t *as_session(void *sess_ctx)
{
    return (confinfer_ta_session_t *)sess_ctx;
}

/*
 * session 暂存状态与模型 runtime 状态分开
 * 分块上传可能中途失败 不应破坏已准备完成的模型
 */
static ta_model_t *find_ready_model(confinfer_model_id_t model_id)
{
    ta_model_t *model = ta_model_find(model_id);

    if (!model || !model->is_registered) {
        return NULL;
    }
    return model;
}

// 两个本地辅助函数 span_map_image / span_release_image
// 输入 REE 连续区的 phys_addr 与 region_size
// 打开 Span PTA session
// 调用 PTA_SPAN_CMD_PROTECT
// PTA 先通过 TF-A 保护物理区，再映射到当前 ConfInfer TA
// 返回 ta_vaddr 和实际 mapped_size
static TEE_Result span_map_image(uint64_t phys_addr, uint64_t region_size,
                                 uint64_t *ta_vaddr, uint64_t *mapped_size,
                                 bool *region_released)
{
    TEE_TASessionHandle session = TEE_HANDLE_NULL;
    TEE_UUID uuid = PTA_SPAN_UUID;
    TEE_Param params[TEE_NUM_PARAMS] = { };
    uint32_t origin = 0;
    TEE_Result res = TEE_SUCCESS;
    const uint32_t param_types =
        TEE_PARAM_TYPES(TEE_PARAM_TYPE_VALUE_INPUT,
                        TEE_PARAM_TYPE_VALUE_INPUT,
                        TEE_PARAM_TYPE_VALUE_OUTPUT,
                        TEE_PARAM_TYPE_VALUE_OUTPUT);

    if (!phys_addr || !region_size || !ta_vaddr || !mapped_size ||
        !region_released) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    *region_released = false;

    res = TEE_OpenTASession(&uuid, TEE_TIMEOUT_INFINITE, 0, NULL,
                            &session, &origin);
    if (res != TEE_SUCCESS) {
        *region_released = true;
        return res;
    }

    params[0].value.a = (uint32_t)(phys_addr >> 32);
    params[0].value.b = (uint32_t)phys_addr;
    params[1].value.a = (uint32_t)(region_size >> 32);
    params[1].value.b = (uint32_t)region_size;
    res = TEE_InvokeTACommand(session, TEE_TIMEOUT_INFINITE,
                              PTA_SPAN_CMD_PROTECT, param_types,
                              params, &origin);
    if (res == TEE_SUCCESS &&
        params[2].value.a == PTA_SPAN_PROTECT_REGION_RELEASED &&
        params[2].value.b == 0 && params[3].value.a == 0 &&
        params[3].value.b == 0) {
        *region_released = true;
        res = TEE_ERROR_GENERIC;
    } else if (res == TEE_SUCCESS) {
        *ta_vaddr = ((uint64_t)params[2].value.a << 32) | params[2].value.b;
        *mapped_size = ((uint64_t)params[3].value.a << 32) | params[3].value.b;
    }

    TEE_CloseTASession(session);
    return res;
}

// 输入此前得到的 ta_vaddr 与 mapped_size
// 调用 PTA_SPAN_CMD_RELEASE
// PTA 先解除当前 ConfInfer TA 的虚拟映射
// 再通过 TF-A 撤销该物理区的 TZC 保护
// 此处不处理 REE 或 Linux 侧的物理页释放
static TEE_Result span_release_image(uint64_t ta_vaddr, uint64_t mapped_size)
{
    TEE_TASessionHandle session = TEE_HANDLE_NULL;
    TEE_UUID uuid = PTA_SPAN_UUID;
    TEE_Param params[TEE_NUM_PARAMS] = { };
    uint32_t origin = 0;
    TEE_Result res = TEE_SUCCESS;
    const uint32_t param_types =
        TEE_PARAM_TYPES(TEE_PARAM_TYPE_VALUE_INPUT,
                        TEE_PARAM_TYPE_VALUE_INPUT,
                        TEE_PARAM_TYPE_NONE,
                        TEE_PARAM_TYPE_NONE);

    if (!ta_vaddr || !mapped_size) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    res = TEE_OpenTASession(&uuid, TEE_TIMEOUT_INFINITE, 0, NULL,
                            &session, &origin);
    if (res != TEE_SUCCESS) {
        return res;
    }

    params[0].value.a = (uint32_t)(ta_vaddr >> 32);
    params[0].value.b = (uint32_t)ta_vaddr;
    params[1].value.a = (uint32_t)(mapped_size >> 32);
    params[1].value.b = (uint32_t)mapped_size;
    res = TEE_InvokeTACommand(session, TEE_TIMEOUT_INFINITE,
                              PTA_SPAN_CMD_RELEASE, param_types,
                              params, &origin);
    TEE_CloseTASession(session);
    return res;
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

    /*
     * 执行保持为三个步骤
     * 导入输入 执行一次 导出输出
     * 在引入更复杂的 buffer 共享策略前
     * 这能让 backend 契约保持简单
     */
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

// 检查协议请求
// -> 确认模型尚未加载
// -> 调用 span_map_image
// -> 检查 PTA 返回的地址和映射范围
// -> ta_model_attach_image
// -> 返回 prepare response
// 其中 ta_model_attach_image 只解析并建立 runtime 视图
// image_data 直接指向 ta_vaddr 不会执行 TEE_Malloc 或 TEE_MemMove 复制整块模型 Image
TEE_Result confinfer_ta_prepare_model_image_trustspan(void *sess_ctx,
                                                      uint32_t param_types,
                                                      TEE_Param params[4])
{
    const uint32_t exp_param_types =
        TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                        TEE_PARAM_TYPE_MEMREF_OUTPUT,
                        TEE_PARAM_TYPE_NONE,
                        TEE_PARAM_TYPE_NONE);
    const confinfer_prepare_model_image_trustspan_req_t *req = NULL;
    confinfer_prepare_model_image_rsp_t *rsp = NULL;
    ta_model_t *model = NULL;
    uint64_t ta_vaddr = 0;
    uint64_t mapped_size = 0;
    bool region_released = true;
    TEE_Result res = TEE_SUCCESS;

    (void)sess_ctx;

    if (param_types != exp_param_types ||
        params[0].memref.size != sizeof(*req) ||
        params[1].memref.size < sizeof(*rsp)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    req = (const confinfer_prepare_model_image_trustspan_req_t *)
        params[0].memref.buffer;
    rsp = (confinfer_prepare_model_image_rsp_t *)params[1].memref.buffer;
    if (!req || !rsp || req->version != CONFINFER_PROTOCOL_VERSION ||
        req->model_id == CONFINFER_INVALID_MODEL_ID || !req->image_size ||
        !req->phys_addr || req->region_size < req->image_size ||
        (uint64_t)(size_t)req->region_size != req->region_size) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    TEE_MemFill(rsp, 0, sizeof(*rsp));
    rsp->version = CONFINFER_PROTOCOL_VERSION;
    rsp->model_id = req->model_id;

    res = ta_model_ensure(req->model_id, &model);
    if (res == TEE_SUCCESS && model->image_data) {
        res = TEE_ERROR_BAD_STATE;
        region_released = false;
    }
    if (res == TEE_SUCCESS) {
        region_released = false;
        res = span_map_image(req->phys_addr, req->region_size,
                             &ta_vaddr, &mapped_size, &region_released);
    }
    if (res == TEE_SUCCESS &&
        (!ta_vaddr || mapped_size < req->image_size ||
         (uint64_t)(size_t)mapped_size != mapped_size)) {
        res = TEE_ERROR_BAD_FORMAT;
    }
    if (res == TEE_SUCCESS) {
        res = ta_model_attach_image(model, (void *)(uintptr_t)ta_vaddr,
                                    req->image_size, (size_t)mapped_size);
    }
    // 若 PTA 映射已成功 但 Image 校验或 runtime 展开失败
    // 仍必须执行完整 release 撤销映射和 TZC 保护
    if (res != TEE_SUCCESS && ta_vaddr) {
        if (span_release_image(ta_vaddr, mapped_size) == TEE_SUCCESS) {
            region_released = true;
        }
    }

    rsp->status = (res == TEE_SUCCESS) ? CONFINFER_STATUS_OK :
                                         CONFINFER_STATUS_BAD_REQUEST;
    rsp->loaded_image_size = (res == TEE_SUCCESS) ? req->image_size : 0;
    if (res != TEE_SUCCESS && region_released) {
        rsp->flags |= CONFINFER_PREPARE_MODEL_IMAGE_RSP_FLAG_REGION_RELEASED;
    }
    params[1].memref.size = sizeof(*rsp);
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

    /*
     * 先处理可能存在的旧暂存状态
     * 再预留完整暂存缓冲区
     * 因为默认 bridge 选择先接收完整 image 再解析模型
     * 而不在 chunk 到达时逐步解析
     */
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

    /*
     * chunk 必须严格按单调 offset 到达
     * 这样可以将暂存缓冲区视为线性字节流
     * 接收端无需实现随机 offset 重组逻辑
     */
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

    /*
     * 在完整 image 到达前 不改变 runtime 所有权
     * begin 与 chunk 只构造完整字节缓冲区
     * end 是传输状态转换为模型状态的唯一边界
     */
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

    // confinfer_ta_unload_model 现在区分两类 Image：
    // - image_owned == 1：默认 bridge 上传的 Image 由 runtime 内部 TEE_Free
    // - image_owned == 0：TrustSpan Image 先完成 span_release_image 再执行 ta_model_release
    if (!model->image_owned && model->image_data) {
        TEE_Result res = span_release_image((uint64_t)(uintptr_t)model->image_data,
                                            model->image_mapping_size);

        if (res != TEE_SUCCESS) {
            rsp->status = CONFINFER_STATUS_BAD_REQUEST;
            params[1].memref.size = sizeof(*rsp);
            return res;
        }
    }

    // 两种 Image 最后统一销毁 runtime 视图和 ta_model_t
    // 只有 image_owned 为 1 时 runtime 才会 TEE_Free Image 字节
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
