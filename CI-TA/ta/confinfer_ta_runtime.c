#include <stdbool.h>

#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>

#include <confinfer_ta_runtime.h>

static ta_model_store_t g_model_store;

static void sort_value_index(ta_value_index_t *items, uint32_t count)
{
    uint32_t i = 0;

    if (!items) {
        return;
    }
    for (i = 1; i < count; ++i) {
        ta_value_index_t key = items[i];
        uint32_t j = i;
        while (j > 0 && items[j - 1].value_id > key.value_id) {
            items[j] = items[j - 1];
            --j;
        }
        items[j] = key;
    }
}

static void sort_param_index(ta_param_index_t *items, uint32_t count)
{
    uint32_t i = 0;

    if (!items) {
        return;
    }
    for (i = 1; i < count; ++i) {
        ta_param_index_t key = items[i];
        uint32_t j = i;
        while (j > 0 && items[j - 1].param_id > key.param_id) {
            items[j] = items[j - 1];
            --j;
        }
        items[j] = key;
    }
}

static void sort_partition_index(ta_partition_index_t *items, uint32_t count)
{
    uint32_t i = 0;

    if (!items) {
        return;
    }
    for (i = 1; i < count; ++i) {
        ta_partition_index_t key = items[i];
        uint32_t j = i;
        while (j > 0 && items[j - 1].partition_id > key.partition_id) {
            items[j] = items[j - 1];
            --j;
        }
        items[j] = key;
    }
}

static ta_value_index_t *find_value_index(ta_value_index_t *items,
                                          uint32_t count,
                                          confinfer_value_id_t value_id)
{
    uint32_t left = 0;
    uint32_t right = count;

    while (left < right) {
        uint32_t mid = left + (right - left) / 2;
        if (items[mid].value_id == value_id) {
            return &items[mid];
        }
        if (items[mid].value_id < value_id) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return NULL;
}

static ta_param_index_t *find_param_index(ta_param_index_t *items,
                                          uint32_t count,
                                          confinfer_param_id_t param_id)
{
    uint32_t left = 0;
    uint32_t right = count;

    while (left < right) {
        uint32_t mid = left + (right - left) / 2;
        if (items[mid].param_id == param_id) {
            return &items[mid];
        }
        if (items[mid].param_id < param_id) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return NULL;
}

static ta_partition_index_t *find_partition_index(ta_partition_index_t *items,
                                                  uint32_t count,
                                                  confinfer_partition_id_t partition_id)
{
    uint32_t left = 0;
    uint32_t right = count;

    while (left < right) {
        uint32_t mid = left + (right - left) / 2;
        if (items[mid].partition_id == partition_id) {
            return &items[mid];
        }
        if (items[mid].partition_id < partition_id) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return NULL;
}

static void ta_data_reset(ta_data_t *data)
{
    if (!data) {
        return;
    }
    TEE_MemFill(data, 0, sizeof(*data));
}

static void ta_data_release(ta_data_t *data)
{
    if (!data) {
        return;
    }
    if (data->ptr) {
        TEE_Free(data->ptr);
        data->ptr = NULL;
    }
    ta_data_reset(data);
}

static void ta_value_reset(ta_value_t *value)
{
    if (!value) {
        return;
    }
    ta_data_release(&value->data);
    TEE_MemFill(value, 0, sizeof(*value));
}

static void ta_layer_reset(ta_layer_t *layer)
{
    if (!layer) {
        return;
    }
    if (layer->attr.ptr) {
        TEE_Free(layer->attr.ptr);
        layer->attr.ptr = NULL;
    }
    if (layer->input_value_ids) {
        TEE_Free(layer->input_value_ids);
        layer->input_value_ids = NULL;
    }
    if (layer->output_value_ids) {
        TEE_Free(layer->output_value_ids);
        layer->output_value_ids = NULL;
    }
    if (layer->inputs) {
        TEE_Free(layer->inputs);
        layer->inputs = NULL;
    }
    if (layer->outputs) {
        TEE_Free(layer->outputs);
        layer->outputs = NULL;
    }
    if (layer->param_refs) {
        TEE_Free(layer->param_refs);
        layer->param_refs = NULL;
    }
    if (layer->params) {
        TEE_Free(layer->params);
        layer->params = NULL;
    }
    TEE_MemFill(layer, 0, sizeof(*layer));
}

static void ta_param_reset(ta_param_t *param)
{
    if (!param) {
        return;
    }
    ta_data_release(&param->data);
    TEE_MemFill(param, 0, sizeof(*param));
    param->param_id = CONFINFER_INVALID_PARAM_ID;
    param->owner_layer_id = CONFINFER_INVALID_LAYER_ID;
    param->owner_partition_id = CONFINFER_INVALID_PARTITION_ID;
}

static void ta_data_from_value_desc(ta_data_t *dst,
                                    const confinfer_value_desc_t *src)
{
    uint32_t i = 0;

    if (!dst || !src) {
        return;
    }

    ta_data_reset(dst);
    dst->shape.elem_count = src->elem_count;
    dst->shape.ndim = src->ndim;
    for (i = 0; i < src->ndim && i < CONFINFER_VALUE_MAX_DIMS; ++i) {
        dst->shape.dims[i] = src->dims[i];
    }
    dst->dtype = src->dtype;
    dst->location = src->location;
    dst->flags = src->flags;
    dst->byte_size = src->byte_size;
    dst->ptr = NULL;
}

static void ta_data_from_param_desc(ta_data_t *dst,
                                    const confinfer_param_desc_t *src)
{
    uint32_t i = 0;

    if (!dst || !src) {
        return;
    }

    ta_data_reset(dst);
    dst->shape.elem_count = src->elem_count;
    dst->shape.ndim = src->ndim;
    for (i = 0; i < src->ndim && i < CONFINFER_VALUE_MAX_DIMS; ++i) {
        dst->shape.dims[i] = src->dims[i];
    }
    dst->dtype = src->dtype;
    dst->location = src->location;
    dst->flags = src->flags;
    dst->byte_size = src->byte_size;
    dst->ptr = NULL;
}

static void ta_value_from_desc(ta_value_t *dst,
                               const confinfer_value_desc_t *src)
{
    if (!dst || !src) {
        return;
    }

    ta_value_reset(dst);
    dst->value_id = src->value_id;
    dst->producer_layer_id = src->producer_layer_id;
    dst->output_index = src->output_index;
    dst->kind = src->kind;
    dst->role_flags = TA_VALUE_ROLE_NONE;
    ta_data_from_value_desc(&dst->data, src);
    if (dst->data.byte_size > 0) {
        dst->data.ptr = TEE_Malloc(dst->data.byte_size, TEE_MALLOC_FILL_ZERO);
    }
}

static TEE_Result ta_layer_from_desc(ta_layer_t *dst,
                                     const confinfer_layer_desc_t *src,
                                     const uint8_t *attr_blob,
                                     size_t attr_blob_size)
{
    if (!dst || !src) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    ta_layer_reset(dst);
    dst->layer_id = src->layer_id;
    dst->layer_type = src->layer_type;
    dst->layer_flags = src->layer_flags;
    dst->attr_flags = 0;
    if (src->attr_size > 0) {
        if (!attr_blob ||
            src->attr_offset > attr_blob_size ||
            src->attr_size > attr_blob_size - src->attr_offset) {
            return TEE_ERROR_BAD_PARAMETERS;
        }
        dst->attr.ptr = TEE_Malloc(src->attr_size, TEE_MALLOC_FILL_ZERO);
        if (!dst->attr.ptr) {
            return TEE_ERROR_OUT_OF_MEMORY;
        }
        dst->attr.size = src->attr_size;
        TEE_MemMove(dst->attr.ptr, attr_blob + src->attr_offset, src->attr_size);
    }
    return TEE_SUCCESS;
}

static TEE_Result ta_layer_deep_copy(ta_layer_t *dst,
                                     const ta_layer_t *src)
{
    if (!dst || !src) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    ta_layer_reset(dst);
    dst->layer_id = src->layer_id;
    dst->layer_type = src->layer_type;
    dst->layer_flags = src->layer_flags;
    dst->attr_flags = src->attr_flags;
    dst->input_value_count = src->input_value_count;
    dst->output_value_count = src->output_value_count;
    dst->param_ref_count = src->param_ref_count;

    if (src->attr.size > 0) {
        if (!src->attr.ptr) {
            return TEE_ERROR_BAD_PARAMETERS;
        }
        dst->attr.ptr = TEE_Malloc(src->attr.size, TEE_MALLOC_FILL_ZERO);
        if (!dst->attr.ptr) {
            goto oom;
        }
        dst->attr.size = src->attr.size;
        TEE_MemMove(dst->attr.ptr, src->attr.ptr, src->attr.size);
    }

    if (src->input_value_count > 0) {
        dst->input_value_ids = TEE_Malloc(src->input_value_count * sizeof(*dst->input_value_ids),
                                          TEE_MALLOC_FILL_ZERO);
        if (!dst->input_value_ids) {
            goto oom;
        }
        TEE_MemMove(dst->input_value_ids, src->input_value_ids,
                    src->input_value_count * sizeof(*dst->input_value_ids));
    }
    if (src->output_value_count > 0) {
        dst->output_value_ids = TEE_Malloc(src->output_value_count * sizeof(*dst->output_value_ids),
                                           TEE_MALLOC_FILL_ZERO);
        if (!dst->output_value_ids) {
            goto oom;
        }
        TEE_MemMove(dst->output_value_ids, src->output_value_ids,
                    src->output_value_count * sizeof(*dst->output_value_ids));
    }

    if (src->param_ref_count > 0) {
        dst->param_refs = TEE_Malloc(src->param_ref_count * sizeof(*dst->param_refs),
                                     TEE_MALLOC_FILL_ZERO);
        if (!dst->param_refs) {
            goto oom;
        }
        TEE_MemMove(dst->param_refs, src->param_refs,
                    src->param_ref_count * sizeof(*dst->param_refs));
    }
    dst->inputs = NULL;
    dst->outputs = NULL;
    dst->params = NULL;
    return TEE_SUCCESS;

oom:
    ta_layer_reset(dst);
    return TEE_ERROR_OUT_OF_MEMORY;
}

static TEE_Result copy_layers(ta_layer_t **dst_layers,
                              uint32_t *dst_count,
                              const ta_layer_t *src_layers,
                              uint32_t src_count)
{
    ta_layer_t *new_layers = NULL;
    uint32_t i = 0;

    if (!dst_layers || !dst_count) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    if (src_count == 0) {
        *dst_layers = NULL;
        *dst_count = 0;
        return TEE_SUCCESS;
    }

    new_layers = TEE_Malloc(src_count * sizeof(*new_layers), TEE_MALLOC_FILL_ZERO);
    if (!new_layers) {
        return TEE_ERROR_OUT_OF_MEMORY;
    }
    for (i = 0; i < src_count; ++i) {
        if (TEE_SUCCESS != ta_layer_deep_copy(&new_layers[i], &src_layers[i])) {
            while (i > 0) {
                --i;
                ta_layer_reset(&new_layers[i]);
            }
            TEE_Free(new_layers);
            return TEE_ERROR_OUT_OF_MEMORY;
        }
    }
    *dst_layers = new_layers;
    *dst_count = src_count;
    return TEE_SUCCESS;
}

static TEE_Result copy_values(ta_value_t **dst_values,
                              uint32_t *dst_count,
                              const ta_value_t *src_values,
                              uint32_t src_count)
{
    ta_value_t *new_values = NULL;
    uint32_t i = 0;

    if (!dst_values || !dst_count) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    if (src_count == 0) {
        *dst_values = NULL;
        *dst_count = 0;
        return TEE_SUCCESS;
    }

    new_values = TEE_Malloc(src_count * sizeof(*new_values), TEE_MALLOC_FILL_ZERO);
    if (!new_values) {
        return TEE_ERROR_OUT_OF_MEMORY;
    }
    for (i = 0; i < src_count; ++i) {
        new_values[i] = src_values[i];
        new_values[i].data.ptr = NULL;
        if (src_values[i].data.byte_size > 0) {
            new_values[i].data.ptr = TEE_Malloc(src_values[i].data.byte_size, TEE_MALLOC_FILL_ZERO);
            if (!new_values[i].data.ptr) {
                goto oom;
            }
            if (src_values[i].data.ptr) {
                TEE_MemMove(new_values[i].data.ptr,
                            src_values[i].data.ptr,
                            src_values[i].data.byte_size);
            }
        }
    }
    *dst_values = new_values;
    *dst_count = src_count;
    return TEE_SUCCESS;

oom:
    while (i > 0) {
        --i;
        ta_value_reset(&new_values[i]);
    }
    TEE_Free(new_values);
    return TEE_ERROR_OUT_OF_MEMORY;
}

static TEE_Result build_partition_value_index(ta_partition_t *part)
{
    uint32_t i = 0;

    if (!part) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    if (part->value_index) {
        TEE_Free(part->value_index);
        part->value_index = NULL;
    }
    if (part->value_count == 0) {
        return TEE_SUCCESS;
    }

    part->value_index = TEE_Malloc(part->value_count * sizeof(*part->value_index),
                                   TEE_MALLOC_FILL_ZERO);
    if (!part->value_index) {
        return TEE_ERROR_OUT_OF_MEMORY;
    }
    for (i = 0; i < part->value_count; ++i) {
        part->value_index[i].value_id = part->values[i].value_id;
        part->value_index[i].value = &part->values[i];
    }
    sort_value_index(part->value_index, part->value_count);
    return TEE_SUCCESS;
}

static TEE_Result build_model_param_index(ta_model_t *model)
{
    uint32_t i = 0;

    if (!model) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    if (model->param_index) {
        TEE_Free(model->param_index);
        model->param_index = NULL;
    }
    if (model->param_count == 0) {
        return TEE_SUCCESS;
    }

    model->param_index = TEE_Malloc(model->param_count * sizeof(*model->param_index),
                                    TEE_MALLOC_FILL_ZERO);
    if (!model->param_index) {
        return TEE_ERROR_OUT_OF_MEMORY;
    }
    for (i = 0; i < model->param_count; ++i) {
        model->param_index[i].param_id = model->params[i].param_id;
        model->param_index[i].param = &model->params[i];
    }
    sort_param_index(model->param_index, model->param_count);
    return TEE_SUCCESS;
}

static TEE_Result build_model_partition_index(ta_model_t *model)
{
    uint32_t i = 0;

    if (!model) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    if (model->partition_index) {
        TEE_Free(model->partition_index);
        model->partition_index = NULL;
    }
    if (model->partition_count == 0) {
        return TEE_SUCCESS;
    }

    model->partition_index = TEE_Malloc(model->partition_count * sizeof(*model->partition_index),
                                        TEE_MALLOC_FILL_ZERO);
    if (!model->partition_index) {
        return TEE_ERROR_OUT_OF_MEMORY;
    }
    for (i = 0; i < model->partition_count; ++i) {
        model->partition_index[i].partition_id = model->partitions[i].partition_id;
        model->partition_index[i].partition = &model->partitions[i];
    }
    sort_partition_index(model->partition_index, model->partition_count);
    return TEE_SUCCESS;
}

static bool is_valid_param_role(uint32_t role)
{
    switch (role) {
    case CONFINFER_PARAM_ROLE_WEIGHT:
    case CONFINFER_PARAM_ROLE_BIAS:
    case CONFINFER_PARAM_ROLE_RUNNING_MEAN:
    case CONFINFER_PARAM_ROLE_RUNNING_VAR:
    case CONFINFER_PARAM_ROLE_UNKNOWN:
        return true;
    default:
        return false;
    }
}

static bool partition_has_value_id(const ta_partition_t *part,
                                   confinfer_value_id_t value_id)
{
    uint32_t i = 0;

    if (!part || !part->values || value_id == 0) {
        return false;
    }
    for (i = 0; i < part->value_count; ++i) {
        if (part->values[i].value_id == value_id) {
            return true;
        }
    }
    return false;
}

static TEE_Result validate_partition_value_descs(const ta_partition_t *part)
{
    uint32_t i = 0;
    uint32_t j = 0;

    if (!part) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    for (i = 0; i < part->value_count; ++i) {
        const ta_value_t *value = &part->values[i];
        if (value->value_id == 0 || value->data.shape.ndim > CONFINFER_VALUE_MAX_DIMS) {
            return TEE_ERROR_BAD_PARAMETERS;
        }
        for (j = i + 1; j < part->value_count; ++j) {
            if (value->value_id == part->values[j].value_id) {
                return TEE_ERROR_BAD_PARAMETERS;
            }
        }
    }
    return TEE_SUCCESS;
}

static TEE_Result validate_partition_layer_descs(const ta_partition_t *part)
{
    uint32_t i = 0;
    uint32_t j = 0;

    if (!part) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    for (i = 0; i < part->layer_count; ++i) {
        const ta_layer_t *layer = &part->layers[i];
        if (layer->layer_id == CONFINFER_INVALID_LAYER_ID) {
            return TEE_ERROR_BAD_PARAMETERS;
        }
        for (j = i + 1; j < part->layer_count; ++j) {
            if (layer->layer_id == part->layers[j].layer_id) {
                return TEE_ERROR_BAD_PARAMETERS;
            }
        }
    }
    return TEE_SUCCESS;
}

static TEE_Result validate_partition_value_refs(const ta_partition_t *part)
{
    uint32_t i = 0;
    uint32_t j = 0;

    if (!part) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    for (i = 0; i < part->layer_count; ++i) {
        const ta_layer_t *layer = &part->layers[i];
        for (j = 0; j < layer->input_value_count; ++j) {
            if (!partition_has_value_id(part, layer->input_value_ids[j])) {
                return TEE_ERROR_BAD_PARAMETERS;
            }
        }
        for (j = 0; j < layer->output_value_count; ++j) {
            if (!partition_has_value_id(part, layer->output_value_ids[j])) {
                return TEE_ERROR_BAD_PARAMETERS;
            }
        }
    }
    return TEE_SUCCESS;
}

static TEE_Result validate_partition_param_refs(ta_model_t *model,
                                                const ta_partition_t *part)
{
    uint32_t i = 0;
    uint32_t j = 0;

    if (!model || !part) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    for (i = 0; i < part->layer_count; ++i) {
        const ta_layer_t *layer = &part->layers[i];
        for (j = 0; j < layer->param_ref_count; ++j) {
            const ta_param_ref_t *ref = &layer->param_refs[j];
            ta_param_t *param = NULL;

            if (ref->param_id == CONFINFER_INVALID_PARAM_ID ||
                !is_valid_param_role(ref->role)) {
                return TEE_ERROR_BAD_PARAMETERS;
            }
            param = ta_model_find_param(model, ref->param_id);
            if (!param) {
                return TEE_ERROR_BAD_STATE;
            }
            // 这里的 owner_layer_id 只表示“参数槽位的规范归属来源”，
            // 不能据此禁止多个 layer 共享同一个 param_id。
            if (param->owner_partition_id != CONFINFER_INVALID_PARTITION_ID &&
                param->owner_partition_id != part->partition_id) {
                return TEE_ERROR_BAD_STATE;
            }
            if (param->role != CONFINFER_PARAM_ROLE_UNKNOWN &&
                param->role != ref->role) {
                return TEE_ERROR_BAD_STATE;
            }
        }
    }
    return TEE_SUCCESS;
}

static TEE_Result ta_partition_bind_value_graph(ta_partition_t *part)
{
    uint32_t i = 0;
    uint32_t j = 0;

    if (!part) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    if (part->layer_count > 0) {
        part->topo = TEE_Malloc(part->layer_count * sizeof(*part->topo),
                                TEE_MALLOC_FILL_ZERO);
        if (!part->topo) {
            return TEE_ERROR_OUT_OF_MEMORY;
        }
    }
    for (i = 0; i < part->layer_count; ++i) {
        ta_layer_t *layer = &part->layers[i];
        part->topo[i] = layer;

        if (layer->input_value_count > 0) {
            layer->inputs = TEE_Malloc(layer->input_value_count * sizeof(*layer->inputs),
                                       TEE_MALLOC_FILL_ZERO);
            if (!layer->inputs) {
                return TEE_ERROR_OUT_OF_MEMORY;
            }
            for (j = 0; j < layer->input_value_count; ++j) {
                ta_value_t *value = ta_partition_find_value_by_id(part, layer->input_value_ids[j]);
                if (!value) {
                    return TEE_ERROR_BAD_STATE;
                }
                layer->inputs[j] = value;
            }
        }

        if (layer->output_value_count > 0) {
            layer->outputs = TEE_Malloc(layer->output_value_count * sizeof(*layer->outputs),
                                        TEE_MALLOC_FILL_ZERO);
            if (!layer->outputs) {
                return TEE_ERROR_OUT_OF_MEMORY;
            }
            for (j = 0; j < layer->output_value_count; ++j) {
                ta_value_t *value = ta_partition_find_value_by_id(part, layer->output_value_ids[j]);
                if (!value) {
                    return TEE_ERROR_BAD_STATE;
                }
                layer->outputs[j] = value;
            }
        }
    }

    if (part->input_count > 0) {
        part->inputs = TEE_Malloc(part->input_count * sizeof(*part->inputs),
                                  TEE_MALLOC_FILL_ZERO);
        if (!part->inputs) {
            return TEE_ERROR_OUT_OF_MEMORY;
        }
        for (i = 0; i < part->input_count; ++i) {
            part->inputs[i] = &part->values[i];
        }
    }
    if (part->output_count > 0) {
        part->outputs = TEE_Malloc(part->output_count * sizeof(*part->outputs),
                                   TEE_MALLOC_FILL_ZERO);
        if (!part->outputs) {
            return TEE_ERROR_OUT_OF_MEMORY;
        }
        for (i = 0; i < part->output_count; ++i) {
            part->outputs[i] = &part->values[part->input_count + i];
        }
    }
    if (part->internal_count > 0) {
        const uint32_t base = part->input_count + part->output_count;
        part->internals = TEE_Malloc(part->internal_count * sizeof(*part->internals),
                                     TEE_MALLOC_FILL_ZERO);
        if (!part->internals) {
            return TEE_ERROR_OUT_OF_MEMORY;
        }
        for (i = 0; i < part->internal_count; ++i) {
            part->internals[i] = &part->values[base + i];
        }
    }

    return TEE_SUCCESS;
}

static TEE_Result ta_partition_bind_param_graph(ta_model_t *model, ta_partition_t *part)
{
    uint32_t i = 0;
    uint32_t j = 0;

    if (!model || !part) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    for (i = 0; i < part->layer_count; ++i) {
        ta_layer_t *layer = &part->layers[i];
        if (layer->param_ref_count == 0) {
            continue;
        }
        if (layer->params) {
            TEE_Free(layer->params);
            layer->params = NULL;
        }
        layer->params = TEE_Malloc(layer->param_ref_count * sizeof(*layer->params),
                                   TEE_MALLOC_FILL_ZERO);
        if (!layer->params) {
            return TEE_ERROR_OUT_OF_MEMORY;
        }
        for (j = 0; j < layer->param_ref_count; ++j) {
            ta_param_t *param = ta_model_find_param(model, layer->param_refs[j].param_id);
            if (!param) {
                return TEE_ERROR_BAD_STATE;
            }
            layer->params[j] = param;
        }
    }
    return TEE_SUCCESS;
}

static void free_partition_arrays(ta_partition_t *part)
{
    uint32_t i = 0;

    if (!part) {
        return;
    }
    if (part->layer_ids) {
        TEE_Free(part->layer_ids);
        part->layer_ids = NULL;
    }
    if (part->topo) {
        TEE_Free(part->topo);
        part->topo = NULL;
    }
    if (part->inputs) {
        TEE_Free(part->inputs);
        part->inputs = NULL;
    }
    if (part->outputs) {
        TEE_Free(part->outputs);
        part->outputs = NULL;
    }
    if (part->internals) {
        TEE_Free(part->internals);
        part->internals = NULL;
    }
    if (part->value_index) {
        TEE_Free(part->value_index);
        part->value_index = NULL;
    }
    if (part->layers) {
        for (i = 0; i < part->layer_count; ++i) {
            ta_layer_reset(&part->layers[i]);
        }
        TEE_Free(part->layers);
        part->layers = NULL;
    }
    if (part->values) {
        for (i = 0; i < part->value_count; ++i) {
            ta_value_reset(&part->values[i]);
        }
        TEE_Free(part->values);
        part->values = NULL;
    }
    part->layer_count = 0;
    part->value_count = 0;
    part->input_count = 0;
    part->output_count = 0;
    part->internal_count = 0;
}

static void free_model_arrays(ta_model_t *model)
{
    uint32_t i = 0;

    if (!model) {
        return;
    }

    if (model->layers) {
        for (i = 0; i < model->layer_count; ++i) {
            ta_layer_reset(&model->layers[i]);
        }
        TEE_Free(model->layers);
        model->layers = NULL;
    }
    if (model->partitions) {
        for (i = 0; i < model->partition_count; ++i) {
            free_partition_arrays(&model->partitions[i]);
        }
        TEE_Free(model->partitions);
        model->partitions = NULL;
    }
    if (model->params) {
        for (i = 0; i < model->param_count; ++i) {
            ta_param_reset(&model->params[i]);
        }
        TEE_Free(model->params);
        model->params = NULL;
    }
    if (model->partition_index) {
        TEE_Free(model->partition_index);
        model->partition_index = NULL;
    }
    if (model->param_index) {
        TEE_Free(model->param_index);
        model->param_index = NULL;
    }

    model->layer_count = 0;
    model->partition_count = 0;
    model->param_count = 0;
}

ta_partition_t *ta_model_find_partition(ta_model_t *model,
                                        confinfer_partition_id_t partition_id)
{
    ta_partition_index_t *found = NULL;

    if (!model || !model->partition_index ||
        partition_id == CONFINFER_INVALID_PARTITION_ID) {
        return NULL;
    }
    found = find_partition_index(model->partition_index,
                                 model->partition_count,
                                 partition_id);
    return found ? found->partition : NULL;
}

ta_value_t *ta_partition_find_value_by_id(ta_partition_t *part, confinfer_value_id_t value_id)
{
    ta_value_index_t *found = NULL;

    if (!part || !part->value_index || value_id == 0) {
        return NULL;
    }
    found = find_value_index(part->value_index,
                             part->value_count,
                             value_id);
    return found ? found->value : NULL;
}

ta_param_t *ta_model_find_param(ta_model_t *model, confinfer_param_id_t param_id)
{
    ta_param_index_t *found = NULL;

    if (!model || !model->param_index || param_id == CONFINFER_INVALID_PARAM_ID) {
        return NULL;
    }
    found = find_param_index(model->param_index,
                             model->param_count,
                             param_id);
    return found ? found->param : NULL;
}

static TEE_Result model_append_missing_layers(ta_model_t *model,
                                              const ta_partition_t *part)
{
    uint32_t i = 0;

    if (!model || !part) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    for (i = 0; i < part->layer_count; ++i) {
        const ta_layer_t *src = &part->layers[i];
        ta_layer_t *new_layers = NULL;
        ta_layer_t *old_layers = model->layers;
        bool exists = false;
        uint32_t j = 0;

        for (j = 0; j < model->layer_count; ++j) {
            if (model->layers[j].layer_id == src->layer_id) {
                exists = true;
                break;
            }
        }
        if (exists) {
            continue;
        }

        new_layers = TEE_Malloc((model->layer_count + 1) * sizeof(*new_layers),
                                TEE_MALLOC_FILL_ZERO);
        if (!new_layers) {
            return TEE_ERROR_OUT_OF_MEMORY;
        }
        if (model->layer_count > 0) {
            TEE_MemMove(new_layers, model->layers,
                        model->layer_count * sizeof(*new_layers));
        }
        if (TEE_SUCCESS != ta_layer_deep_copy(&new_layers[model->layer_count], src)) {
            TEE_Free(new_layers);
            model->layers = old_layers;
            return TEE_ERROR_OUT_OF_MEMORY;
        }
        TEE_Free(old_layers);
        model->layers = new_layers;
        model->layer_count += 1;
    }
    return TEE_SUCCESS;
}

void ta_runtime_init(void)
{
    static bool initialized = false;

    if (initialized) {
        return;
    }
    TEE_MemFill(&g_model_store, 0, sizeof(g_model_store));
    initialized = true;
}

void ta_runtime_deinit(void)
{
    ta_model_store_t *store = &g_model_store;

    while (store->model_count > 0) {
        ta_model_release(store->models[store->model_count - 1]);
    }
    TEE_MemFill(store, 0, sizeof(*store));
}

ta_model_store_t *ta_runtime_store(void)
{
    ta_runtime_init();
    return &g_model_store;
}

void ta_partition_reset(ta_partition_t *part)
{
    if (!part) {
        return;
    }
    TEE_MemFill(part, 0, sizeof(*part));
    part->partition_id = CONFINFER_INVALID_PARTITION_ID;
}

void ta_partition_release(ta_partition_t *part)
{
    if (!part) {
        return;
    }
    free_partition_arrays(part);
    ta_partition_reset(part);
}

TEE_Result ta_partition_init_from_proto(
    ta_partition_t *part,
    const confinfer_partition_req_t *req,
    const confinfer_layer_desc_t *layers,
    const void *layer_attr_blob,
    size_t layer_attr_blob_size,
    const confinfer_partition_data_req_t *data_req,
    const confinfer_value_desc_t *input_values,
    const confinfer_value_desc_t *output_values,
    const confinfer_value_desc_t *internal_values,
    const confinfer_layer_io_desc_t *layer_ios,
    const confinfer_layer_value_ref_t *input_refs,
    const confinfer_layer_value_ref_t *output_refs,
    const confinfer_layer_param_ref_t *param_refs)
{
    uint32_t i = 0;
    TEE_Result res = TEE_SUCCESS;
    const uint32_t input_base = 0;
    const uint32_t output_base = data_req->input_count;
    const uint32_t internal_base = data_req->input_count + data_req->output_count;
    const uint8_t *attr_blob = (const uint8_t *)layer_attr_blob;

    if (!part || !req || !data_req) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    ta_partition_reset(part);
    part->partition_id = req->partition_id;
    part->domain = req->domain;
    part->unit_type = req->unit_type;
    part->flags = req->flags;
    part->input_count = data_req->input_count;
    part->output_count = data_req->output_count;
    part->internal_count = data_req->internal_count;
    part->value_count = data_req->input_count + data_req->output_count + data_req->internal_count;

    if (req->layer_count > 0) {
        part->layers = TEE_Malloc(req->layer_count * sizeof(*part->layers),
                                  TEE_MALLOC_FILL_ZERO);
        if (!part->layers) {
            res = TEE_ERROR_OUT_OF_MEMORY;
            goto fail;
        }
        part->layer_count = req->layer_count;
        for (i = 0; i < req->layer_count; ++i) {
            res = ta_layer_from_desc(&part->layers[i], &layers[i],
                                     attr_blob, layer_attr_blob_size);
            if (res != TEE_SUCCESS) {
                goto fail;
            }
        }
    }

    if (part->value_count > 0) {
        part->values = TEE_Malloc(part->value_count * sizeof(*part->values),
                                  TEE_MALLOC_FILL_ZERO);
        if (!part->values) {
            res = TEE_ERROR_OUT_OF_MEMORY;
            goto fail;
        }
        for (i = 0; i < data_req->input_count; ++i) {
            ta_value_from_desc(&part->values[input_base + i], &input_values[i]);
            part->values[input_base + i].role_flags |= TA_VALUE_ROLE_INPUT;
        }
        for (i = 0; i < data_req->output_count; ++i) {
            ta_value_from_desc(&part->values[output_base + i], &output_values[i]);
            part->values[output_base + i].role_flags |= TA_VALUE_ROLE_OUTPUT;
        }
        for (i = 0; i < data_req->internal_count; ++i) {
            ta_value_from_desc(&part->values[internal_base + i], &internal_values[i]);
            part->values[internal_base + i].role_flags |= TA_VALUE_ROLE_INTERNAL;
        }
    }

    if (layer_ios) {
        for (i = 0; i < req->layer_count; ++i) {
            const confinfer_layer_io_desc_t *io = &layer_ios[i];
            ta_layer_t *layer = &part->layers[i];
            uint32_t j = 0;

            if (io->input_ref_count > 0) {
                layer->input_value_ids = TEE_Malloc(io->input_ref_count * sizeof(*layer->input_value_ids),
                                                    TEE_MALLOC_FILL_ZERO);
                if (!layer->input_value_ids) {
                    res = TEE_ERROR_OUT_OF_MEMORY;
                    goto fail;
                }
                layer->input_value_count = io->input_ref_count;
                for (j = 0; j < io->input_ref_count; ++j) {
                    const confinfer_layer_value_ref_t *ref = &input_refs[io->input_ref_start + j];
                    layer->input_value_ids[j] = ref->value_id;
                }
            }

            if (io->output_ref_count > 0) {
                layer->output_value_ids = TEE_Malloc(io->output_ref_count * sizeof(*layer->output_value_ids),
                                                     TEE_MALLOC_FILL_ZERO);
                if (!layer->output_value_ids) {
                    res = TEE_ERROR_OUT_OF_MEMORY;
                    goto fail;
                }
                layer->output_value_count = io->output_ref_count;
                for (j = 0; j < io->output_ref_count; ++j) {
                    const confinfer_layer_value_ref_t *ref = &output_refs[io->output_ref_start + j];
                    layer->output_value_ids[j] = ref->value_id;
                }
            }
            if (io->param_ref_count > 0) {
                layer->param_refs = TEE_Malloc(io->param_ref_count * sizeof(*layer->param_refs),
                                               TEE_MALLOC_FILL_ZERO);
                if (!layer->param_refs) {
                    res = TEE_ERROR_OUT_OF_MEMORY;
                    goto fail;
                }
                layer->param_ref_count = io->param_ref_count;
                for (j = 0; j < io->param_ref_count; ++j) {
                    const confinfer_layer_param_ref_t *ref =
                        &param_refs[io->param_ref_start + j];
                    layer->param_refs[j].param_id = ref->param_id;
                    layer->param_refs[j].role = ref->role;
                }
            }
        }
    }

    res = validate_partition_value_descs(part);
    if (res != TEE_SUCCESS) {
        goto fail;
    }
    res = build_partition_value_index(part);
    if (res != TEE_SUCCESS) {
        goto fail;
    }
    res = validate_partition_layer_descs(part);
    if (res != TEE_SUCCESS) {
        goto fail;
    }
    res = validate_partition_value_refs(part);
    if (res != TEE_SUCCESS) {
        goto fail;
    }

    res = ta_partition_bind_value_graph(part);
    if (res != TEE_SUCCESS) {
        goto fail;
    }

    return TEE_SUCCESS;

fail:
    ta_partition_release(part);
    return res;
}

ta_model_t *ta_model_find(confinfer_model_id_t model_id)
{
    ta_model_store_t *store = ta_runtime_store();
    uint32_t i = 0;

    for (i = 0; i < store->model_count; ++i) {
        if (store->models[i] && store->models[i]->model_id == model_id) {
            return store->models[i];
        }
    }
    return NULL;
}

TEE_Result ta_model_ensure(confinfer_model_id_t model_id,
                           ta_model_t **out_model)
{
    ta_model_store_t *store = ta_runtime_store();
    ta_model_t *model = NULL;

    if (!out_model) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    model = ta_model_find(model_id);
    if (model) {
        *out_model = model;
        return TEE_SUCCESS;
    }
    if (store->model_count >= CONFINFER_TA_MAX_MODELS) {
        return TEE_ERROR_OUT_OF_MEMORY;
    }

    model = TEE_Malloc(sizeof(*model), TEE_MALLOC_FILL_ZERO);
    if (!model) {
        return TEE_ERROR_OUT_OF_MEMORY;
    }
    model->model_id = model_id;
    store->models[store->model_count++] = model;
    *out_model = model;
    return TEE_SUCCESS;
}

// 专门用来 用 confinfer_model_desc_t 初始化 ta_model_t
TEE_Result ta_model_register(const confinfer_model_desc_t *desc,
                             ta_model_t **out_model)
{
    ta_model_t *model = NULL;
    TEE_Result res = TEE_SUCCESS;

    if (!desc || !out_model) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    if (desc->version != CONFINFER_PROTOCOL_VERSION ||
        desc->model_id == CONFINFER_INVALID_MODEL_ID) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    res = ta_model_ensure(desc->model_id, &model);
    if (res != TEE_SUCCESS) {
        return res;
    }

    // REGISTER_MODEL 代表重新初始化这个模型上下文，
    // 所以先把已有的 layer / partition / param 内容清空。
    free_model_arrays(model);
    model->model_id = desc->model_id;
    model->flags = desc->flags;
    model->expected_partition_count = desc->expected_partition_count;
    model->expected_param_count = desc->expected_param_count;
    model->is_registered = 1;
    build_model_partition_index(model);
    build_model_param_index(model);

    *out_model = model;
    return TEE_SUCCESS;
}

TEE_Result ta_model_upsert_partition(ta_model_t *model,
                                     const ta_partition_t *part)
{
    ta_partition_t *target = NULL;
    ta_partition_t *new_parts = NULL;
    uint32_t old_partition_count = 0;
    TEE_Result res = TEE_SUCCESS;
    bool is_new_partition = false;

    if (!model || !part) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    if (!model->is_registered) {
        return TEE_ERROR_BAD_STATE;
    }
    res = model_append_missing_layers(model, part);
    if (res != TEE_SUCCESS) {
        return res;
    }
    res = validate_partition_param_refs(model, part);
    if (res != TEE_SUCCESS) {
        return res;
    }

    target = ta_model_find_partition(model, part->partition_id);
    is_new_partition = (target == NULL);
    if (model->expected_partition_count > 0 &&
        is_new_partition &&
        model->partition_count >= model->expected_partition_count) {
        return TEE_ERROR_BAD_STATE;
    }
    if (!target) {
        old_partition_count = model->partition_count;
        new_parts = TEE_Malloc((model->partition_count + 1) * sizeof(*new_parts),
                               TEE_MALLOC_FILL_ZERO);
        if (!new_parts) {
            return TEE_ERROR_OUT_OF_MEMORY;
        }
        if (model->partition_count > 0) {
            TEE_MemMove(new_parts, model->partitions,
                        model->partition_count * sizeof(*new_parts));
            TEE_Free(model->partitions);
        }
        model->partitions = new_parts;
        target = &model->partitions[model->partition_count];
        ta_partition_reset(target);
        model->partition_count += 1;
    } else {
        free_partition_arrays(target);
    }

    target->partition_id = part->partition_id;
    target->domain = part->domain;
    target->unit_type = part->unit_type;
    target->flags = part->flags;
    target->input_count = part->input_count;
    target->output_count = part->output_count;
    target->internal_count = part->internal_count;

    if (part->layer_count > 0) {
        uint32_t i = 0;

        target->layer_ids = TEE_Malloc(part->layer_count * sizeof(*target->layer_ids),
                                       TEE_MALLOC_FILL_ZERO);
        if (!target->layer_ids) {
            return TEE_ERROR_OUT_OF_MEMORY;
        }
        for (i = 0; i < part->layer_count; ++i) {
            target->layer_ids[i] = part->layers[i].layer_id;
        }
    }

    res = copy_layers(&target->layers, &target->layer_count, part->layers, part->layer_count);
    if (res != TEE_SUCCESS) {
        goto fail;
    }
    res = copy_values(&target->values, &target->value_count, part->values, part->value_count);
    if (res != TEE_SUCCESS) {
        goto fail;
    }
    res = build_partition_value_index(target);
    if (res != TEE_SUCCESS) {
        goto fail;
    }
    res = ta_partition_bind_value_graph(target);
    if (res != TEE_SUCCESS) {
        goto fail;
    }
    res = ta_partition_bind_param_graph(model, target);
    if (res != TEE_SUCCESS) {
        goto fail;
    }
    res = build_model_partition_index(model);
    if (res != TEE_SUCCESS) {
        goto fail;
    }

    return TEE_SUCCESS;

fail:
    if (is_new_partition) {
        free_partition_arrays(target);
        model->partition_count = old_partition_count;
    } else {
        free_partition_arrays(target);
    }
    build_model_partition_index(model);
    return res;
}

TEE_Result ta_model_load_params(ta_model_t *model,
                                const confinfer_load_params_req_t *req,
                                const confinfer_param_desc_t *param_descs,
                                const void *param_blob,
                                size_t param_blob_size)
{
    ta_param_t *new_params = NULL;
    ta_param_t *old_params = NULL;
    uint32_t old_param_count = 0;
    TEE_Result res = TEE_SUCCESS;
    const uint8_t *blob = (const uint8_t *)param_blob;
    uint32_t i = 0;

    if (!model || !req) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    if (!model->is_registered) {
        return TEE_ERROR_BAD_STATE;
    }
    if (req->version != CONFINFER_PROTOCOL_VERSION ||
        req->model_id != model->model_id) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    if (req->param_count > 0 && (!param_descs || !blob)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    if (req->total_param_bytes != param_blob_size) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    if (model->expected_param_count > 0 &&
        req->param_count != model->expected_param_count) {
        return TEE_ERROR_BAD_STATE;
    }

    if (req->param_count == 0) {
        if (model->params) {
            for (i = 0; i < model->param_count; ++i) {
                ta_param_reset(&model->params[i]);
            }
            TEE_Free(model->params);
            model->params = NULL;
        }
        model->param_count = 0;
        if (model->param_index) {
            TEE_Free(model->param_index);
            model->param_index = NULL;
        }
        return TEE_SUCCESS;
    }

    new_params = TEE_Malloc(req->param_count * sizeof(*new_params), TEE_MALLOC_FILL_ZERO);
    if (!new_params) {
        return TEE_ERROR_OUT_OF_MEMORY;
    }

    for (i = 0; i < req->param_count; ++i) {
        const confinfer_param_desc_t *src = &param_descs[i];
        ta_param_t *dst = &new_params[i];
        uint32_t j = 0;

        if (src->param_id == CONFINFER_INVALID_PARAM_ID ||
            src->ndim > CONFINFER_VALUE_MAX_DIMS ||
            !is_valid_param_role(src->role)) {
            goto bad_desc;
        }
        if (src->data_offset > req->total_param_bytes ||
            src->byte_size > req->total_param_bytes - src->data_offset) {
            goto bad_desc;
        }
        for (j = 0; j < i; ++j) {
            if (new_params[j].param_id == src->param_id) {
                goto bad_desc;
            }
        }

        ta_param_reset(dst);
        dst->param_id = src->param_id;
        dst->owner_layer_id = src->owner_layer_id;
        dst->owner_partition_id = src->owner_partition_id;
        dst->flags = src->flags;
        dst->role = src->role;
        ta_data_from_param_desc(&dst->data, src);

        if (src->byte_size > 0) {
            dst->data.ptr = TEE_Malloc(src->byte_size, TEE_MALLOC_FILL_ZERO);
            if (!dst->data.ptr) {
                goto oom;
            }
            TEE_MemMove(dst->data.ptr, blob + src->data_offset, src->byte_size);
        }
    }

    old_params = model->params;
    old_param_count = model->param_count;
    model->params = new_params;
    model->param_count = req->param_count;
    res = build_model_param_index(model);
    if (res != TEE_SUCCESS) {
        goto restore_old;
    }
    for (i = 0; i < model->partition_count; ++i) {
        TEE_Result bind_res = ta_partition_bind_param_graph(model, &model->partitions[i]);
        if (bind_res != TEE_SUCCESS) {
            res = bind_res;
            goto restore_old;
        }
    }
    if (old_params) {
        for (i = 0; i < old_param_count; ++i) {
            ta_param_reset(&old_params[i]);
        }
        TEE_Free(old_params);
    }
    return TEE_SUCCESS;

restore_old:
    for (i = 0; i < req->param_count; ++i) {
        ta_param_reset(&new_params[i]);
    }
    TEE_Free(new_params);
    model->params = old_params;
    model->param_count = old_param_count;
    if (build_model_param_index(model) == TEE_SUCCESS) {
        for (i = 0; i < model->partition_count; ++i) {
            ta_partition_bind_param_graph(model, &model->partitions[i]);
        }
    }
    return res;

bad_desc:
    for (i = 0; i < req->param_count; ++i) {
        ta_param_reset(&new_params[i]);
    }
    TEE_Free(new_params);
    return TEE_ERROR_BAD_PARAMETERS;

oom:
    for (i = 0; i < req->param_count; ++i) {
        ta_param_reset(&new_params[i]);
    }
    TEE_Free(new_params);
    return TEE_ERROR_OUT_OF_MEMORY;
}

void ta_model_release(ta_model_t *model)
{
    ta_model_store_t *store = ta_runtime_store();
    uint32_t i = 0;

    if (!model) {
        return;
    }

    free_model_arrays(model);

    for (i = 0; i < store->model_count; ++i) {
        if (store->models[i] == model) {
            uint32_t j = i;
            for (; j + 1 < store->model_count; ++j) {
                store->models[j] = store->models[j + 1];
            }
            store->models[store->model_count - 1] = NULL;
            store->model_count -= 1;
            break;
        }
    }
    TEE_Free(model);
}
