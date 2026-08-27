#include <stdbool.h>

#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>

#include <confinfer_ta_runtime.h>

static ta_model_store_t g_model_store;

static void ta_data_init_from_value_desc(ta_data_t *dst,
                                         const confinfer_model_image_value_desc_t *src,
                                         void *buffer)
{
    uint32_t i = 0;

    TEE_MemFill(dst, 0, sizeof(*dst));
    dst->shape.elem_count = src->elem_count;
    dst->shape.ndim = src->ndim;
    for (i = 0; i < src->ndim && i < CONFINFER_VALUE_MAX_DIMS; ++i) {
        dst->shape.dims[i] = src->dims[i];
    }
    dst->dtype = src->dtype;
    dst->location = src->location;
    dst->flags = src->flags;
    dst->byte_size = src->byte_size;
    dst->ptr = buffer;
}

static void ta_data_init_from_param_desc(ta_data_t *dst,
                                         const confinfer_model_image_param_desc_t *src,
                                         const void *buffer)
{
    uint32_t i = 0;

    TEE_MemFill(dst, 0, sizeof(*dst));
    dst->shape.elem_count = src->elem_count;
    dst->shape.ndim = src->ndim;
    for (i = 0; i < src->ndim && i < CONFINFER_VALUE_MAX_DIMS; ++i) {
        dst->shape.dims[i] = src->dims[i];
    }
    dst->dtype = src->dtype;
    dst->location = src->location;
    dst->flags = src->flags;
    dst->byte_size = src->byte_size;
    dst->ptr = (void *)buffer;
}

static void ta_release_layer(ta_layer_t *layer)
{
    if (!layer) {
        return;
    }
    TEE_MemFill(layer, 0, sizeof(*layer));
}

static void ta_release_value(ta_value_t *value)
{
    if (!value) {
        return;
    }
    TEE_MemFill(value, 0, sizeof(*value));
}

static void ta_release_model_image(ta_model_t *model)
{
    uint32_t i = 0;

    if (!model) {
        return;
    }
    if (model->partitions) {
        for (i = 0; i < model->partition_count; ++i) {
            ta_partition_release(&model->partitions[i]);
        }
        TEE_Free(model->partitions);
    }
    model->partitions = NULL;
    if (model->image_data) {
        TEE_Free(model->image_data);
    }
    model->image_data = NULL;
    model->image_size = 0;
    model->partition_count = 0;
    model->param_count = 0;
    model->is_registered = 0;
}

static bool image_range_ok(size_t total_size, size_t offset, size_t span)
{
    return offset <= total_size && span <= total_size - offset;
}

static TEE_Result validate_model_image_header(const confinfer_model_image_header_t *hdr,
                                              size_t image_size)
{
    uint32_t i = 0;
    const confinfer_model_image_partition_entry_t *entries = NULL;
    uint32_t table_span = 0;
    uint32_t param_desc_span = 0;

    if (!hdr) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    if (hdr->magic != CONFINFER_MODEL_IMAGE_MAGIC ||
        hdr->version_major != CONFINFER_MODEL_IMAGE_VERSION_MAJOR ||
        hdr->version_minor != CONFINFER_MODEL_IMAGE_VERSION_MINOR ||
        hdr->header_size < sizeof(*hdr) ||
        hdr->total_size != image_size ||
        hdr->model_id == CONFINFER_INVALID_MODEL_ID) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    table_span = hdr->partition_count * sizeof(*entries);
    param_desc_span = hdr->param_desc_count * sizeof(confinfer_model_image_param_desc_t);
    if (hdr->partition_table_size != table_span) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    if (hdr->header_size != sizeof(*hdr)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    if (!image_range_ok(image_size, hdr->partition_table_off, table_span) ||
        !image_range_ok(image_size, hdr->param_desc_off, param_desc_span) ||
        !image_range_ok(image_size, hdr->param_data_off, hdr->param_data_size)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    entries = (const confinfer_model_image_partition_entry_t *)
        (((const uint8_t *)hdr) + hdr->partition_table_off);
    for (i = 0; i < hdr->partition_count; ++i) {
        const confinfer_partition_image_header_t *part_hdr = NULL;

        if (entries[i].partition_id == CONFINFER_INVALID_PARTITION_ID ||
            !image_range_ok(image_size, entries[i].image_off, entries[i].image_size) ||
            entries[i].image_size < sizeof(*part_hdr)) {
            return TEE_ERROR_BAD_PARAMETERS;
        }
        part_hdr = (const confinfer_partition_image_header_t *)
            (((const uint8_t *)hdr) + entries[i].image_off);
        if (part_hdr->magic != CONFINFER_PARTITION_IMAGE_MAGIC ||
            part_hdr->version_major != CONFINFER_MODEL_IMAGE_VERSION_MAJOR ||
            part_hdr->version_minor != CONFINFER_MODEL_IMAGE_VERSION_MINOR ||
            part_hdr->partition_id != entries[i].partition_id ||
            part_hdr->total_size != entries[i].image_size) {
            return TEE_ERROR_BAD_PARAMETERS;
        }
    }

    return TEE_SUCCESS;
}

ta_partition_t *ta_model_find_partition(ta_model_t *model,
                                        confinfer_partition_id_t partition_id)
{
    uint32_t i = 0;

    if (!model || !model->partitions) {
        return NULL;
    }
    for (i = 0; i < model->partition_count; ++i) {
        if (model->partitions[i].partition_id == partition_id) {
            return &model->partitions[i];
        }
    }
    return NULL;
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

TEE_Result ta_model_ensure(confinfer_model_id_t model_id, ta_model_t **out_model)
{
    ta_model_store_t *store = ta_runtime_store();
    ta_model_t *model = NULL;

    if (!out_model || model_id == CONFINFER_INVALID_MODEL_ID) {
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

TEE_Result ta_model_load_image(ta_model_t *model,
                               const void *image_data,
                               size_t image_size)
{
    void *image_copy = NULL;
    const confinfer_model_image_header_t *hdr =
        (const confinfer_model_image_header_t *)image_data;
    const confinfer_model_image_partition_entry_t *entries = NULL;
    TEE_Result res = TEE_SUCCESS;
    uint32_t i = 0;

    if (!model || !image_data || image_size < sizeof(*hdr)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    res = validate_model_image_header(hdr, image_size);
    if (res != TEE_SUCCESS) {
        return res;
    }

    image_copy = TEE_Malloc(image_size, TEE_MALLOC_FILL_ZERO);
    if (!image_copy) {
        return TEE_ERROR_OUT_OF_MEMORY;
    }
    TEE_MemMove(image_copy, image_data, image_size);

    ta_release_model_image(model);
    model->model_id = hdr->model_id;
    model->flags = hdr->flags;
    model->partition_count = hdr->partition_count;
    model->param_count = hdr->param_desc_count;
    model->image_data = image_copy;
    model->image_size = (uint32_t)image_size;
    model->is_registered = 1;
    if (model->partition_count == 0) {
        return TEE_SUCCESS;
    }

    model->partitions = TEE_Malloc(model->partition_count * sizeof(*model->partitions),
                                   TEE_MALLOC_FILL_ZERO);
    if (!model->partitions) {
        ta_release_model_image(model);
        return TEE_ERROR_OUT_OF_MEMORY;
    }

    entries = (const confinfer_model_image_partition_entry_t *)
        (((const uint8_t *)model->image_data) + hdr->partition_table_off);
    for (i = 0; i < model->partition_count; ++i) {
        res = ta_model_materialize_partition(model,
                                             entries[i].partition_id,
                                             &model->partitions[i]);
        if (res != TEE_SUCCESS) {
            ta_release_model_image(model);
            return res;
        }
    }
    return TEE_SUCCESS;
}

const confinfer_model_image_header_t *ta_model_image_header(const ta_model_t *model)
{
    if (!model || !model->image_data || model->image_size < sizeof(confinfer_model_image_header_t)) {
        return NULL;
    }
    return (const confinfer_model_image_header_t *)model->image_data;
}

const confinfer_model_image_partition_entry_t *ta_model_partition_entry(
    const ta_model_t *model,
    confinfer_partition_id_t partition_id)
{
    const confinfer_model_image_header_t *hdr = ta_model_image_header(model);
    const confinfer_model_image_partition_entry_t *entries = NULL;
    uint32_t i = 0;

    if (!hdr) {
        return NULL;
    }
    entries = (const confinfer_model_image_partition_entry_t *)
        (((const uint8_t *)hdr) + hdr->partition_table_off);
    for (i = 0; i < hdr->partition_count; ++i) {
        if (entries[i].partition_id == partition_id) {
            return &entries[i];
        }
    }
    return NULL;
}

const confinfer_model_image_param_desc_t *ta_model_param_descs(const ta_model_t *model)
{
    const confinfer_model_image_header_t *hdr = ta_model_image_header(model);

    if (!hdr || hdr->param_desc_count == 0) {
        return NULL;
    }
    return (const confinfer_model_image_param_desc_t *)
        (((const uint8_t *)hdr) + hdr->param_desc_off);
}

const uint8_t *ta_model_param_data(const ta_model_t *model)
{
    const confinfer_model_image_header_t *hdr = ta_model_image_header(model);

    if (!hdr || hdr->param_data_size == 0) {
        return NULL;
    }
    return ((const uint8_t *)hdr) + hdr->param_data_off;
}

const uint8_t *ta_model_partition_image_data(
    const ta_model_t *model,
    const confinfer_model_image_partition_entry_t *entry)
{
    const confinfer_model_image_header_t *hdr = ta_model_image_header(model);

    if (!hdr || !entry || entry->image_size == 0) {
        return NULL;
    }
    return ((const uint8_t *)hdr) + entry->image_off;
}

const confinfer_model_image_param_desc_t *ta_model_find_param_desc(
    const ta_model_t *model,
    confinfer_param_id_t param_id)
{
    const confinfer_model_image_header_t *hdr = ta_model_image_header(model);
    const confinfer_model_image_param_desc_t *descs = ta_model_param_descs(model);
    uint32_t i = 0;

    if (!hdr || !descs || param_id == CONFINFER_INVALID_PARAM_ID) {
        return NULL;
    }
    for (i = 0; i < hdr->param_desc_count; ++i) {
        if (descs[i].param_id == param_id) {
            return &descs[i];
        }
    }
    return NULL;
}

const void *ta_model_param_buffer(const ta_model_t *model,
                                  const confinfer_model_image_param_desc_t *desc)
{
    const confinfer_model_image_header_t *hdr = ta_model_image_header(model);
    const uint8_t *data = ta_model_param_data(model);

    if (!hdr || !desc) {
        return NULL;
    }
    if (desc->byte_size == 0) {
        return NULL;
    }
    if (!data ||
        desc->data_offset > hdr->param_data_size ||
        desc->byte_size > hdr->param_data_size - desc->data_offset) {
        return NULL;
    }
    return data + desc->data_offset;
}

TEE_Result ta_model_open_partition_image(const ta_model_t *model,
                                         confinfer_partition_id_t partition_id,
                                         ta_partition_image_view_t *view)
{
    const confinfer_model_image_partition_entry_t *entry = NULL;
    const uint8_t *base = NULL;
    const confinfer_partition_image_header_t *hdr = NULL;
    uint32_t layer_desc_span = 0;
    uint32_t value_desc_span = 0;
    uint32_t layer_io_span = 0;
    uint32_t input_ref_span = 0;
    uint32_t output_ref_span = 0;
    uint32_t param_ref_span = 0;

    if (!view) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    TEE_MemFill(view, 0, sizeof(*view));

    entry = ta_model_partition_entry(model, partition_id);
    if (!entry) {
        return TEE_ERROR_ITEM_NOT_FOUND;
    }

    base = ta_model_partition_image_data(model, entry);
    if (!base || entry->image_size < sizeof(*hdr)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    hdr = (const confinfer_partition_image_header_t *)base;
    if (hdr->magic != CONFINFER_PARTITION_IMAGE_MAGIC ||
        hdr->partition_id != partition_id ||
        hdr->total_size != entry->image_size) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    layer_desc_span = hdr->layer_count * sizeof(confinfer_model_image_layer_desc_t);
    value_desc_span = hdr->value_count * sizeof(confinfer_model_image_value_desc_t);
    layer_io_span = hdr->layer_count * sizeof(confinfer_model_image_layer_io_t);
    input_ref_span = hdr->input_ref_count * sizeof(confinfer_model_image_value_ref_t);
    output_ref_span = hdr->output_ref_count * sizeof(confinfer_model_image_value_ref_t);
    if (hdr->input_count + hdr->output_count + hdr->internal_count != hdr->value_count) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    if (hdr->param_ref_off > hdr->total_size ||
        hdr->attr_blob_off > hdr->total_size ||
        hdr->attr_blob_off < hdr->param_ref_off ||
        hdr->attr_blob_size > hdr->total_size - hdr->attr_blob_off ||
        hdr->runtime_data_off < hdr->attr_blob_off + hdr->attr_blob_size ||
        hdr->runtime_data_off > hdr->total_size ||
        hdr->runtime_data_size > hdr->total_size - hdr->runtime_data_off ||
        !image_range_ok(hdr->total_size, hdr->layer_desc_off, layer_desc_span) ||
        !image_range_ok(hdr->total_size, hdr->value_desc_off, value_desc_span) ||
        !image_range_ok(hdr->total_size, hdr->layer_io_off, layer_io_span) ||
        !image_range_ok(hdr->total_size, hdr->input_ref_off, input_ref_span) ||
        !image_range_ok(hdr->total_size, hdr->output_ref_off, output_ref_span)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    param_ref_span = hdr->param_ref_count * sizeof(confinfer_model_image_param_ref_t);
    if (!image_range_ok(hdr->total_size, hdr->param_ref_off, param_ref_span) ||
        hdr->param_ref_off + param_ref_span != hdr->attr_blob_off) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    view->header = hdr;
    view->layers = (const confinfer_model_image_layer_desc_t *)(base + hdr->layer_desc_off);
    view->values = (const confinfer_model_image_value_desc_t *)(base + hdr->value_desc_off);
    view->layer_ios = (const confinfer_model_image_layer_io_t *)(base + hdr->layer_io_off);
    view->input_refs = (const confinfer_model_image_value_ref_t *)(base + hdr->input_ref_off);
    view->output_refs = (const confinfer_model_image_value_ref_t *)(base + hdr->output_ref_off);
    view->param_refs = (const confinfer_model_image_param_ref_t *)(base + hdr->param_ref_off);
    view->attr_blob = base + hdr->attr_blob_off;
    return TEE_SUCCESS;
}

static ta_value_t *value_by_id(ta_partition_t *part, confinfer_value_id_t value_id)
{
    uint32_t i = 0;

    if (!part || value_id == 0) {
        return NULL;
    }
    for (i = 0; i < part->value_count; ++i) {
        if (part->values[i].value_id == value_id) {
            return &part->values[i];
        }
    }
    return NULL;
}

static TEE_Result build_partition_values(ta_partition_t *part,
                                         const ta_partition_image_view_t *view)
{
    uint32_t i = 0;
    uint8_t *runtime_data = NULL;

    if (view->header->value_count == 0) {
        return TEE_SUCCESS;
    }
    runtime_data = (uint8_t *)view->header + view->header->runtime_data_off;

    part->values = TEE_Malloc(view->header->value_count * sizeof(*part->values),
                              TEE_MALLOC_FILL_ZERO);
    if (!part->values) {
        return TEE_ERROR_OUT_OF_MEMORY;
    }

    for (i = 0; i < view->header->value_count; ++i) {
        const confinfer_model_image_value_desc_t *src = &view->values[i];
        ta_value_t *dst = &part->values[i];

        dst->value_id = src->value_id;
        dst->producer_layer_id = src->producer_layer_id;
        dst->output_index = src->output_index;
        dst->kind = src->kind;
        if (src->data_offset > view->header->runtime_data_size ||
            src->byte_size > view->header->runtime_data_size - src->data_offset) {
            return TEE_ERROR_BAD_PARAMETERS;
        }
        ta_data_init_from_value_desc(&dst->data, src, runtime_data + src->data_offset);

        switch (src->role) {
        case CONFINFER_MODEL_IMAGE_VALUE_INPUT:
            dst->role_flags = TA_VALUE_ROLE_INPUT;
            break;
        case CONFINFER_MODEL_IMAGE_VALUE_OUTPUT:
            dst->role_flags = TA_VALUE_ROLE_OUTPUT;
            break;
        case CONFINFER_MODEL_IMAGE_VALUE_INTERNAL:
            dst->role_flags = TA_VALUE_ROLE_INTERNAL;
            break;
        default:
            return TEE_ERROR_BAD_PARAMETERS;
        }
    }
    return TEE_SUCCESS;
}

static TEE_Result build_partition_layers(const ta_model_t *model,
                                         ta_partition_t *part,
                                         const ta_partition_image_view_t *view)
{
    uint32_t i = 0;
    uint32_t param_base = 0;

    if (part->layer_count == 0) {
        return TEE_SUCCESS;
    }

    part->layers = TEE_Malloc(part->layer_count * sizeof(*part->layers),
                              TEE_MALLOC_FILL_ZERO);
    if (!part->layers) {
        return TEE_ERROR_OUT_OF_MEMORY;
    }

    for (i = 0; i < part->layer_count; ++i) {
        part->param_count += view->layer_ios[i].param_ref_count;
    }
    if (part->param_count > 0) {
        part->params = TEE_Malloc(part->param_count * sizeof(*part->params),
                                  TEE_MALLOC_FILL_ZERO);
        if (!part->params) {
            return TEE_ERROR_OUT_OF_MEMORY;
        }
    }

    for (i = 0; i < part->layer_count; ++i) {
        const confinfer_model_image_layer_desc_t *src = &view->layers[i];
        const confinfer_model_image_layer_io_t *io = &view->layer_ios[i];
        ta_layer_t *dst = &part->layers[i];
        uint32_t j = 0;

        dst->layer_id = src->layer_id;
        dst->layer_type = src->layer_type;
        dst->layer_flags = src->flags;
        dst->input_value_count = io->input_ref_count;
        dst->output_value_count = io->output_ref_count;
        dst->param_ref_count = io->param_ref_count;
        dst->input_refs = view->input_refs + io->input_ref_begin;
        dst->output_refs = view->output_refs + io->output_ref_begin;
        dst->param_refs = view->param_refs + io->param_ref_begin;
        dst->param_begin = param_base;
        if (io->input_ref_begin > view->header->input_ref_count ||
            io->input_ref_count > view->header->input_ref_count - io->input_ref_begin ||
            io->output_ref_begin > view->header->output_ref_count ||
            io->output_ref_count > view->header->output_ref_count - io->output_ref_begin ||
            io->param_ref_begin > view->header->param_ref_count ||
            io->param_ref_count > view->header->param_ref_count - io->param_ref_begin) {
            return TEE_ERROR_BAD_PARAMETERS;
        }
        if (src->attr_size > 0) {
            if (src->attr_off > view->header->attr_blob_size ||
                src->attr_size > view->header->attr_blob_size - src->attr_off) {
                return TEE_ERROR_BAD_PARAMETERS;
            }
            dst->attr.ptr = view->attr_blob + src->attr_off;
            dst->attr.size = src->attr_size;
        }

        if (dst->param_ref_count > 0) {
            for (j = 0; j < dst->param_ref_count; ++j) {
                const confinfer_model_image_param_ref_t *ref = &dst->param_refs[j];
                const confinfer_model_image_param_desc_t *param_desc =
                    ta_model_find_param_desc(model, ref->param_id);
                ta_param_t *param = &part->params[param_base + j];
                const void *param_buffer = NULL;

                if (!param_desc) {
                    return TEE_ERROR_BAD_PARAMETERS;
                }
                param_buffer = ta_model_param_buffer(model, param_desc);
                if (!param_buffer && param_desc->byte_size > 0) {
                    return TEE_ERROR_BAD_PARAMETERS;
                }

                param->param_id = param_desc->param_id;
                param->owner_layer_id = param_desc->owner_layer_id;
                param->owner_partition_id = param_desc->owner_partition_id;
                param->flags = param_desc->flags;
                param->role = param_desc->role;
                ta_data_init_from_param_desc(&param->data,
                                             param_desc,
                                             param_buffer);
            }
            param_base += dst->param_ref_count;
        }

        for (j = 0; j < dst->input_value_count; ++j) {
            if (!value_by_id(part, dst->input_refs[j].value_id)) {
                return TEE_ERROR_BAD_PARAMETERS;
            }
        }
        for (j = 0; j < dst->output_value_count; ++j) {
            if (!value_by_id(part, dst->output_refs[j].value_id)) {
                return TEE_ERROR_BAD_PARAMETERS;
            }
        }
    }

    return TEE_SUCCESS;
}

TEE_Result ta_model_materialize_partition(const ta_model_t *model,
                                          confinfer_partition_id_t partition_id,
                                          ta_partition_t *part)
{
    ta_partition_image_view_t view;
    TEE_Result res = TEE_SUCCESS;

    if (!model || !part || !model->is_registered) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    TEE_MemFill(part, 0, sizeof(*part));
    part->partition_id = CONFINFER_INVALID_PARTITION_ID;

    res = ta_model_open_partition_image(model, partition_id, &view);
    if (res != TEE_SUCCESS) {
        return res;
    }

    part->partition_id = partition_id;
    part->domain = CONFINFER_DATA_TEE;
    part->flags = view.header->flags;
    part->layer_count = view.header->layer_count;
    part->input_count = view.header->input_count;
    part->output_count = view.header->output_count;
    part->internal_count = view.header->internal_count;
    part->value_count = view.header->value_count;

    res = build_partition_values(part, &view);
    if (res != TEE_SUCCESS) {
        ta_partition_release(part);
        return res;
    }

    res = build_partition_layers(model, part, &view);
    if (res != TEE_SUCCESS) {
        ta_partition_release(part);
        return res;
    }

    return TEE_SUCCESS;
}

void ta_partition_release(ta_partition_t *part)
{
    uint32_t i = 0;

    if (!part) {
        return;
    }
    if (part->layers) {
        for (i = 0; i < part->layer_count; ++i) {
            ta_release_layer(&part->layers[i]);
        }
        TEE_Free(part->layers);
    }
    if (part->values) {
        for (i = 0; i < part->value_count; ++i) {
            ta_release_value(&part->values[i]);
        }
        TEE_Free(part->values);
    }
    if (part->params) {
        TEE_Free(part->params);
    }
    TEE_MemFill(part, 0, sizeof(*part));
    part->partition_id = CONFINFER_INVALID_PARTITION_ID;
}

ta_value_t *ta_partition_find_value_by_id(ta_partition_t *part,
                                          confinfer_value_id_t value_id)
{
    return value_by_id(part, value_id);
}

void ta_model_release(ta_model_t *model)
{
    ta_model_store_t *store = ta_runtime_store();
    uint32_t i = 0;

    if (!model) {
        return;
    }

    ta_release_model_image(model);

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
