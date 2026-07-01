#include <stdbool.h>
#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>

#include <confinfer_ta_backend.h>
#include <conf_infer_ta.h>
#include <confinfer_ta_commands.h>
#include <confinfer_ta_runtime.h>

static uint32_t value_desc_total_bytes(const confinfer_value_desc_t *values, uint32_t count)
{
    uint32_t total = 0;
    uint32_t i = 0;

    for (i = 0; i < count; ++i) {
        total += values[i].byte_size;
    }
    return total;
}

static ta_model_t *find_registered_model(confinfer_model_id_t model_id)
{
    ta_model_t *model = NULL;

    if (model_id == CONFINFER_INVALID_MODEL_ID) {
        return NULL;
    }
    model = ta_model_find(model_id);
    if (!model || !model->is_registered) {
        return NULL;
    }
    return model;
}

typedef struct {
    const confinfer_partition_req_t *req;
    const confinfer_layer_desc_t *layers;
    const uint8_t *layer_attr_blob;
    uint32_t layer_attr_blob_size;
    const confinfer_partition_data_req_t *data_req;
    const confinfer_value_desc_t *input_values;
    const confinfer_value_desc_t *output_values;
    const confinfer_value_desc_t *internal_values;
    const confinfer_layer_io_desc_t *layer_ios;
    const confinfer_layer_value_ref_t *input_refs;
    const confinfer_layer_value_ref_t *output_refs;
    const confinfer_layer_param_ref_t *param_refs;
    const uint8_t *input_blob;
    uint8_t *output_blob;
} ta_partition_proto_view_t;

static TEE_Result parse_partition_proto(uint32_t param_types,
                                        TEE_Param params[4],
                                        bool with_runtime_data,
                                        ta_partition_proto_view_t *view,
                                        confinfer_partition_rsp_t **rsp)
{
    const uint32_t exp_param_types =
        with_runtime_data ?
        TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                        TEE_PARAM_TYPE_MEMREF_INPUT,
                        TEE_PARAM_TYPE_MEMREF_INOUT,
                        TEE_PARAM_TYPE_MEMREF_OUTPUT) :
        TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                        TEE_PARAM_TYPE_MEMREF_INPUT,
                        TEE_PARAM_TYPE_MEMREF_INPUT,
                        TEE_PARAM_TYPE_MEMREF_OUTPUT);
    uint32_t expected_layers_size = 0;
    uint32_t expected_data_size = 0;
    uint32_t input_descs_size = 0;
    uint32_t output_descs_size = 0;
    uint32_t internal_descs_size = 0;
    uint32_t layer_ios_size = 0;
    uint32_t input_refs_size = 0;
    uint32_t output_refs_size = 0;
    uint32_t param_refs_size = 0;
    uint32_t layer_attr_total_size = 0;
    uint32_t i = 0;

    if (!view || !rsp) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    TEE_MemFill(view, 0, sizeof(*view));

    if (param_types != exp_param_types) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    if (params[0].memref.size != sizeof(*view->req) ||
        params[2].memref.size < sizeof(confinfer_partition_data_req_t) ||
        params[3].memref.size < sizeof(**rsp)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    view->req = (const confinfer_partition_req_t *)params[0].memref.buffer;
    view->layers = (const confinfer_layer_desc_t *)params[1].memref.buffer;
    view->data_req = (const confinfer_partition_data_req_t *)params[2].memref.buffer;
    *rsp = (confinfer_partition_rsp_t *)params[3].memref.buffer;
    if (!view->req || !view->data_req || !*rsp) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    if (view->req->version != CONFINFER_PROTOCOL_VERSION ||
        view->data_req->version != CONFINFER_PROTOCOL_VERSION ||
        view->req->model_id == CONFINFER_INVALID_MODEL_ID ||
        view->req->partition_id == CONFINFER_INVALID_PARTITION_ID) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    TEE_MemFill(*rsp, 0, sizeof(**rsp));
    (*rsp)->version = CONFINFER_PROTOCOL_VERSION;
    (*rsp)->domain = view->req->domain;
    (*rsp)->model_id = view->req->model_id;
    (*rsp)->partition_id = view->req->partition_id;
    (*rsp)->consumed_inputs = view->req->input_count;
    (*rsp)->produced_outputs = view->req->output_count;

    expected_layers_size = view->req->layer_count * sizeof(*view->layers);
    if (params[1].memref.size < expected_layers_size) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    if (view->req->layer_count > 0 && !view->layers) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    for (i = 0; i < view->req->layer_count; ++i) {
        layer_attr_total_size += view->layers[i].attr_size;
    }
    if (params[1].memref.size != expected_layers_size + layer_attr_total_size) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    view->layer_attr_blob = ((const uint8_t *)view->layers) + expected_layers_size;
    view->layer_attr_blob_size = layer_attr_total_size;

    input_descs_size = view->data_req->input_count * sizeof(confinfer_value_desc_t);
    output_descs_size = view->data_req->output_count * sizeof(confinfer_value_desc_t);
    internal_descs_size = view->data_req->internal_count * sizeof(confinfer_value_desc_t);
    layer_ios_size = view->data_req->layer_io_count * sizeof(confinfer_layer_io_desc_t);
    input_refs_size = view->data_req->input_ref_count * sizeof(confinfer_layer_value_ref_t);
    output_refs_size = view->data_req->output_ref_count * sizeof(confinfer_layer_value_ref_t);
    param_refs_size = view->data_req->param_ref_count * sizeof(confinfer_layer_param_ref_t);
    expected_data_size = sizeof(*view->data_req) + input_descs_size + output_descs_size +
                         internal_descs_size + layer_ios_size + input_refs_size +
                         output_refs_size + param_refs_size;
    if (with_runtime_data) {
        expected_data_size += view->data_req->total_input_bytes + view->data_req->total_output_bytes;
    }
    if (expected_data_size != params[2].memref.size) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    if (view->req->input_count != view->data_req->input_count ||
        view->req->output_count != view->data_req->output_count ||
        view->req->layer_count != view->data_req->layer_io_count) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    view->input_values = (const confinfer_value_desc_t *)(view->data_req + 1);
    view->output_values = (const confinfer_value_desc_t *)((const uint8_t *)view->input_values + input_descs_size);
    view->internal_values = (const confinfer_value_desc_t *)((const uint8_t *)view->output_values + output_descs_size);
    view->layer_ios = (const confinfer_layer_io_desc_t *)((const uint8_t *)view->internal_values + internal_descs_size);
    view->input_refs = (const confinfer_layer_value_ref_t *)((const uint8_t *)view->layer_ios + layer_ios_size);
    view->output_refs = (const confinfer_layer_value_ref_t *)((const uint8_t *)view->input_refs + input_refs_size);
    view->param_refs = (const confinfer_layer_param_ref_t *)((const uint8_t *)view->output_refs + output_refs_size);
    // register_partition 只消费到 param_refs 为止；
    // exec_partition 会在其后继续拼接 input/output 的真实字节流。
    view->input_blob = ((const uint8_t *)view->param_refs) + param_refs_size;
    view->output_blob = with_runtime_data ?
        (uint8_t *)(view->input_blob + view->data_req->total_input_bytes) : NULL;

    if (view->req->domain != CONFINFER_DOMAIN_CPU_TEE) {
        (*rsp)->status = CONFINFER_PART_UNSUPPORTED_DOMAIN;
        params[3].memref.size = sizeof(**rsp);
        return TEE_SUCCESS;
    }
    if (view->req->unit_type != CONFINFER_UNIT_PARTITION &&
        view->req->unit_type != CONFINFER_UNIT_LAYER) {
        (*rsp)->status = CONFINFER_PART_UNSUPPORTED_UNIT;
        params[3].memref.size = sizeof(**rsp);
        return TEE_SUCCESS;
    }

    for (i = 0; i < view->req->layer_count; ++i) {
        const confinfer_layer_desc_t *desc = &view->layers[i];
        const confinfer_layer_io_desc_t *io = &view->layer_ios[i];
        if (desc->reserved != 0) {
            (*rsp)->status = CONFINFER_PART_BAD_LAYER_DESC;
            params[3].memref.size = sizeof(**rsp);
            return TEE_SUCCESS;
        }
        if (desc->layer_id == CONFINFER_INVALID_LAYER_ID ||
            desc->attr_offset > view->layer_attr_blob_size ||
            desc->attr_size > view->layer_attr_blob_size - desc->attr_offset ||
            io->layer_id != desc->layer_id ||
            io->input_ref_start > view->data_req->input_ref_count ||
            io->output_ref_start > view->data_req->output_ref_count ||
            io->param_ref_start > view->data_req->param_ref_count ||
            io->input_ref_count > view->data_req->input_ref_count - io->input_ref_start ||
            io->output_ref_count > view->data_req->output_ref_count - io->output_ref_start ||
            io->param_ref_count > view->data_req->param_ref_count - io->param_ref_start) {
            (*rsp)->status = CONFINFER_PART_BAD_LAYER_DESC;
            params[3].memref.size = sizeof(**rsp);
            return TEE_SUCCESS;
        }
    }
    if (value_desc_total_bytes(view->input_values, view->data_req->input_count) != view->data_req->total_input_bytes ||
        value_desc_total_bytes(view->output_values, view->data_req->output_count) != view->data_req->total_output_bytes) {
        (*rsp)->status = CONFINFER_PART_BAD_DATA_DESC;
        params[3].memref.size = sizeof(**rsp);
        return TEE_SUCCESS;
    }
    for (i = 0; i < view->data_req->input_count; ++i) {
        if (view->input_values[i].value_id == 0 ||
            view->input_values[i].ndim > CONFINFER_VALUE_MAX_DIMS) {
            (*rsp)->status = CONFINFER_PART_BAD_DATA_DESC;
            params[3].memref.size = sizeof(**rsp);
            return TEE_SUCCESS;
        }
    }
    for (i = 0; i < view->data_req->output_count; ++i) {
        if (view->output_values[i].value_id == 0 ||
            view->output_values[i].ndim > CONFINFER_VALUE_MAX_DIMS) {
            (*rsp)->status = CONFINFER_PART_BAD_DATA_DESC;
            params[3].memref.size = sizeof(**rsp);
            return TEE_SUCCESS;
        }
    }
    for (i = 0; i < view->data_req->internal_count; ++i) {
        if (view->internal_values[i].value_id == 0 ||
            view->internal_values[i].ndim > CONFINFER_VALUE_MAX_DIMS) {
            (*rsp)->status = CONFINFER_PART_BAD_DATA_DESC;
            params[3].memref.size = sizeof(**rsp);
            return TEE_SUCCESS;
        }
    }
    for (i = 0; i < view->data_req->input_ref_count; ++i) {
        if (view->input_refs[i].value_id == 0 || view->input_refs[i].reserved != 0) {
            (*rsp)->status = CONFINFER_PART_BAD_DATA_DESC;
            params[3].memref.size = sizeof(**rsp);
            return TEE_SUCCESS;
        }
    }
    for (i = 0; i < view->data_req->output_ref_count; ++i) {
        if (view->output_refs[i].value_id == 0 || view->output_refs[i].reserved != 0) {
            (*rsp)->status = CONFINFER_PART_BAD_DATA_DESC;
            params[3].memref.size = sizeof(**rsp);
            return TEE_SUCCESS;
        }
    }
    for (i = 0; i < view->data_req->param_ref_count; ++i) {
        if (view->param_refs[i].param_id == CONFINFER_INVALID_PARAM_ID) {
            (*rsp)->status = CONFINFER_PART_BAD_DATA_DESC;
            params[3].memref.size = sizeof(**rsp);
            return TEE_SUCCESS;
        }
    }

    return TEE_SUCCESS;
}

static TEE_Result load_partition_inputs(ta_partition_t *part,
                                        const confinfer_value_desc_t *input_descs,
                                        uint32_t input_count,
                                        const uint8_t *input_blob,
                                        uint32_t input_blob_size)
{
    uint32_t i = 0;
    const uint8_t *cursor = input_blob;

    if ((input_blob_size > 0 && !input_blob) || (!part && input_count > 0)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    for (i = 0; i < input_count; ++i) {
        ta_value_t *value = ta_partition_find_value_by_id(part, input_descs[i].value_id);
        if (!value || !value->data.ptr) {
            return TEE_ERROR_BAD_STATE;
        }
        if (value->data.byte_size != input_descs[i].byte_size ||
            input_descs[i].byte_size > input_blob_size) {
            return TEE_ERROR_BAD_PARAMETERS;
        }
        if (input_descs[i].byte_size > 0) {
            TEE_MemMove(value->data.ptr, cursor, input_descs[i].byte_size);
            cursor += input_descs[i].byte_size;
            input_blob_size -= input_descs[i].byte_size;
        }
    }
    if (input_blob_size != 0) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    return TEE_SUCCESS;
}

static TEE_Result store_partition_outputs(const ta_partition_t *part,
                                          const confinfer_value_desc_t *output_descs,
                                          uint32_t output_count,
                                          uint8_t *output_blob,
                                          uint32_t output_blob_size)
{
    uint32_t i = 0;
    uint8_t *cursor = output_blob;

    if ((output_blob_size > 0 && !output_blob) || (!part && output_count > 0)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    for (i = 0; i < output_count; ++i) {
        ta_value_t *value = ta_partition_find_value_by_id((ta_partition_t *)part, output_descs[i].value_id);
        if (!value || !value->data.ptr) {
            return TEE_ERROR_BAD_STATE;
        }
        if (value->data.byte_size != output_descs[i].byte_size ||
            output_descs[i].byte_size > output_blob_size) {
            return TEE_ERROR_BAD_PARAMETERS;
        }
        if (output_descs[i].byte_size > 0) {
            TEE_MemMove(cursor, value->data.ptr, output_descs[i].byte_size);
            cursor += output_descs[i].byte_size;
            output_blob_size -= output_descs[i].byte_size;
        }
    }
    if (output_blob_size != 0) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    return TEE_SUCCESS;
}

static void log_partition_runtime(const ta_partition_t *part)
{
    uint32_t i = 0;
    uint32_t j = 0;

    if (!part) {
        return;
    }

    IMSG("runtime partition: partition_id=%u layer_count=%u value_count=%u input_count=%u output_count=%u internal_count=%u",
         part->partition_id, part->layer_count, part->value_count,
         part->input_count, part->output_count, part->internal_count);
    for (i = 0; i < part->layer_count; ++i) {
        IMSG("runtime layer[%u]: id=%u type=%u flags=0x%x",
             i, part->layers[i].layer_id, part->layers[i].layer_type,
             part->layers[i].layer_flags);
        for (j = 0; j < part->layers[i].input_value_count; ++j) {
            IMSG("runtime layer[%u] input_ref[%u]: value_id=%u",
                 i, j, part->layers[i].input_value_ids[j]);
        }
        for (j = 0; j < part->layers[i].output_value_count; ++j) {
            IMSG("runtime layer[%u] output_ref[%u]: value_id=%u",
                 i, j, part->layers[i].output_value_ids[j]);
        }
    }
}

TEE_Result confinfer_ta_inc_value(uint32_t param_types, TEE_Param params[4])
{
    const uint32_t exp_param_types =
        TEE_PARAM_TYPES(TEE_PARAM_TYPE_VALUE_INOUT,
                        TEE_PARAM_TYPE_NONE,
                        TEE_PARAM_TYPE_NONE,
                        TEE_PARAM_TYPE_NONE);

    if (param_types != exp_param_types) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    params[0].value.a++;
    return TEE_SUCCESS;
}

TEE_Result confinfer_ta_dec_value(uint32_t param_types, TEE_Param params[4])
{
    const uint32_t exp_param_types =
        TEE_PARAM_TYPES(TEE_PARAM_TYPE_VALUE_INOUT,
                        TEE_PARAM_TYPE_NONE,
                        TEE_PARAM_TYPE_NONE,
                        TEE_PARAM_TYPE_NONE);

    if (param_types != exp_param_types) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    params[0].value.a--;
    return TEE_SUCCESS;
}
 
// 不构建 layer/partition/param 只是构建一个 TA 内模型上下文根对象 g_model_store
TEE_Result confinfer_ta_register_model(uint32_t param_types, TEE_Param params[4])
{
    const confinfer_model_desc_t *desc = NULL;
    confinfer_model_rsp_t *rsp = NULL;
    ta_model_t *model = NULL;
    TEE_Result res = TEE_SUCCESS;
    const uint32_t exp_param_types =
        TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                        TEE_PARAM_TYPE_MEMREF_OUTPUT,
                        TEE_PARAM_TYPE_NONE,
                        TEE_PARAM_TYPE_NONE);

    if (param_types != exp_param_types) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    if (params[0].memref.size != sizeof(*desc) ||
        params[1].memref.size < sizeof(*rsp)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    desc = (const confinfer_model_desc_t *)params[0].memref.buffer;
    rsp = (confinfer_model_rsp_t *)params[1].memref.buffer;
    if (!desc || !rsp) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    if (desc->version != CONFINFER_PROTOCOL_VERSION ||
        desc->model_id == CONFINFER_INVALID_MODEL_ID ||
        desc->reserved0 != 0 ||
        desc->reserved1 != 0) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    TEE_MemFill(rsp, 0, sizeof(*rsp));
    rsp->version = CONFINFER_PROTOCOL_VERSION;
    rsp->model_id = desc->model_id;
    rsp->flags = desc->flags;

    // 挂载到全局 ta_model_store_t g_model_store
    res = ta_model_register(desc, &model);
    if (res != TEE_SUCCESS) {
        return res;
    }

    rsp->status = CONFINFER_MODEL_OK;
    rsp->partition_count = model->partition_count;
    rsp->param_count = model->param_count;
    params[1].memref.size = sizeof(*rsp);

    IMSG("register_model ok: model_id=%u flags=0x%x expected_partition_count=%u expected_param_count=%u",
         model->model_id, model->flags,
         model->expected_partition_count, model->expected_param_count);
    return TEE_SUCCESS;
}

TEE_Result confinfer_ta_load_params(uint32_t param_types, TEE_Param params[4])
{
    const uint32_t exp_param_types =
        TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                        TEE_PARAM_TYPE_MEMREF_INPUT,
                        TEE_PARAM_TYPE_MEMREF_INPUT,
                        TEE_PARAM_TYPE_MEMREF_OUTPUT);
    const confinfer_load_params_req_t *req = NULL;
    const confinfer_param_desc_t *param_descs = NULL;
    const void *param_blob = NULL;
    confinfer_load_params_rsp_t *rsp = NULL;
    ta_model_t *model = NULL;
    TEE_Result res = TEE_SUCCESS;
    uint32_t expected_desc_size = 0;

    if (param_types != exp_param_types) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    if (params[0].memref.size != sizeof(*req) ||
        params[3].memref.size < sizeof(*rsp)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    req = (const confinfer_load_params_req_t *)params[0].memref.buffer;
    param_descs = (const confinfer_param_desc_t *)params[1].memref.buffer;
    param_blob = params[2].memref.buffer;
    rsp = (confinfer_load_params_rsp_t *)params[3].memref.buffer;
    if (!req || !rsp) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    if (req->version != CONFINFER_PROTOCOL_VERSION ||
        req->model_id == CONFINFER_INVALID_MODEL_ID ||
        req->reserved0 != 0 ||
        req->reserved1 != 0) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    expected_desc_size = req->param_count * sizeof(*param_descs);
    if (params[1].memref.size != expected_desc_size) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    if (params[2].memref.size != req->total_param_bytes) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    if ((req->param_count > 0 && !param_descs) ||
        (req->total_param_bytes > 0 && !param_blob)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    // 根据 model_id 找到已注册模型
    model = find_registered_model(req->model_id);
    if (!model) {
        return TEE_ERROR_ITEM_NOT_FOUND;
    }

    TEE_MemFill(rsp, 0, sizeof(*rsp));
    rsp->version = CONFINFER_PROTOCOL_VERSION;
    rsp->model_id = req->model_id;

    res = ta_model_load_params(model, req, param_descs, param_blob, params[2].memref.size);
    if (res != TEE_SUCCESS) {
        return res;
    }

    rsp->status = CONFINFER_PARAM_OK;
    rsp->loaded_param_count = model->param_count;
    rsp->total_param_bytes = req->total_param_bytes;
    params[3].memref.size = sizeof(*rsp);

    IMSG("load_params ok: model_id=%u param_count=%u total_param_bytes=%u",
         model->model_id, model->param_count, req->total_param_bytes);
    return TEE_SUCCESS;
}

TEE_Result confinfer_ta_register_partition(uint32_t param_types, TEE_Param params[4])
{
    ta_partition_proto_view_t view;
    ta_partition_t runtime_part;
    ta_model_t *runtime_model = NULL;
    confinfer_partition_rsp_t *rsp = NULL;
    TEE_Result res = TEE_SUCCESS;

    ta_partition_reset(&runtime_part);
    res = parse_partition_proto(param_types, params, false, &view, &rsp);
    if (res != TEE_SUCCESS) {
        return res;
    }

    if (rsp->status != 0) {
        params[3].memref.size = sizeof(*rsp);
        return TEE_SUCCESS;
    }

    runtime_model = find_registered_model(view.req->model_id);
    if (!runtime_model) {
        return TEE_ERROR_ITEM_NOT_FOUND;
    }

    res = ta_partition_init_from_proto(&runtime_part,
                                       view.req,
                                       view.layers,
                                       view.layer_attr_blob,
                                       view.layer_attr_blob_size,
                                       view.data_req,
                                       view.input_values,
                                       view.output_values,
                                       view.internal_values,
                                       view.layer_ios,
                                       view.input_refs,
                                       view.output_refs,
                                       view.param_refs);
    if (res != TEE_SUCCESS) {
        return res;
    }

    res = ta_model_upsert_partition(runtime_model, &runtime_part);
    ta_partition_release(&runtime_part);
    if (res != TEE_SUCCESS) {
        return res;
    }

    rsp->status = CONFINFER_PART_OK;
    rsp->executed_layers = view.req->layer_count;
    params[3].memref.size = sizeof(*rsp);
    IMSG("register_partition ok: model_id=%u partition_id=%u layer_count=%u",
         runtime_model->model_id, view.req->partition_id, view.req->layer_count);
    return TEE_SUCCESS;
}

TEE_Result confinfer_ta_exec_partition(uint32_t param_types, TEE_Param params[4])
{
    ta_partition_proto_view_t view;
    ta_model_t *runtime_model = NULL;
    confinfer_partition_rsp_t *rsp = NULL;
    TEE_Result res = TEE_SUCCESS;

    res = parse_partition_proto(param_types, params, true, &view, &rsp);
    if (res != TEE_SUCCESS) {
        return res;
    }

    if (rsp->status != 0) {
        params[3].memref.size = sizeof(*rsp);
        return TEE_SUCCESS;
    }

    if (view.req->model_id != CONFINFER_INVALID_MODEL_ID) {
        ta_partition_t *registered_part = NULL;

        runtime_model = find_registered_model(view.req->model_id);
        if (!runtime_model) {
            return TEE_ERROR_ITEM_NOT_FOUND;
        }
        registered_part = ta_model_find_partition(runtime_model, view.req->partition_id);
        if (!registered_part) {
            EMSG("exec_partition partition is not registered: model_id=%u partition_id=%u",
                 view.req->model_id, view.req->partition_id);
            return TEE_ERROR_ITEM_NOT_FOUND;
        }
        res = load_partition_inputs(registered_part,
                                    view.input_values,
                                    view.data_req->input_count,
                                    view.input_blob,
                                    view.data_req->total_input_bytes);
        if (res != TEE_SUCCESS) {
            return res;
        }
        res = ta_backend_execute_partition(ta_backend_default(),
                                           runtime_model,
                                           registered_part);
        if (res != TEE_SUCCESS) {
            return res;
        }
        res = store_partition_outputs(registered_part,
                                      view.output_values,
                                      view.data_req->output_count,
                                      view.output_blob,
                                      view.data_req->total_output_bytes);
        if (res != TEE_SUCCESS) {
            return res;
        }
    }

    rsp->status = CONFINFER_PART_OK;
    rsp->executed_layers = view.req->layer_count;
    rsp->produced_outputs = view.req->output_count;
    params[3].memref.size = sizeof(*rsp);
    IMSG("exec_partition ready: model_id=%u partition_id=%u layer_count=%u",
         view.req->model_id, view.req->partition_id, view.req->layer_count);
    return TEE_SUCCESS;
}

TEE_Result confinfer_ta_unload_model(uint32_t param_types, TEE_Param params[4])
{
    const confinfer_unload_model_req_t *req = NULL;
    confinfer_unload_model_rsp_t *rsp = NULL;
    ta_model_t *model = NULL;
    const uint32_t exp_param_types =
        TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                        TEE_PARAM_TYPE_MEMREF_OUTPUT,
                        TEE_PARAM_TYPE_NONE,
                        TEE_PARAM_TYPE_NONE);

    if (param_types != exp_param_types) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    if (params[0].memref.size != sizeof(*req) ||
        params[1].memref.size < sizeof(*rsp)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    req = (const confinfer_unload_model_req_t *)params[0].memref.buffer;
    rsp = (confinfer_unload_model_rsp_t *)params[1].memref.buffer;
    if (!req || !rsp) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    if (req->version != CONFINFER_PROTOCOL_VERSION ||
        req->model_id == CONFINFER_INVALID_MODEL_ID) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    TEE_MemFill(rsp, 0, sizeof(*rsp));
    rsp->version = CONFINFER_PROTOCOL_VERSION;
    rsp->model_id = req->model_id;

    model = ta_model_find(req->model_id);
    if (!model) {
        rsp->status = CONFINFER_UNLOAD_MODEL_NOT_FOUND;
        params[1].memref.size = sizeof(*rsp);
        return TEE_SUCCESS;
    }

    rsp->released_partition_count = model->partition_count;
    rsp->released_param_count = model->param_count;
    ta_model_release(model);
    rsp->status = CONFINFER_UNLOAD_MODEL_OK;
    params[1].memref.size = sizeof(*rsp);

    IMSG("unload_model ok: model_id=%u released_partition_count=%u released_param_count=%u",
         req->model_id, rsp->released_partition_count, rsp->released_param_count);
    return TEE_SUCCESS;
}
