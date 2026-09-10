#ifndef CONFINFER_TA_RUNTIME_H
#define CONFINFER_TA_RUNTIME_H

#include <tee_internal_api.h>

#include <confinfer_model_image.h>
#include <confinfer_protocol.h>

#define CONFINFER_TA_MAX_MODELS 8u

enum ta_value_role_flags {
    TA_VALUE_ROLE_NONE = 0,
    TA_VALUE_ROLE_INPUT = 1u << 0,
    TA_VALUE_ROLE_OUTPUT = 1u << 1,
    TA_VALUE_ROLE_INTERNAL = 1u << 2,
};

typedef struct ta_value ta_value_t;
typedef struct ta_layer ta_layer_t;
typedef struct ta_param ta_param_t;
typedef struct ta_partition ta_partition_t;
typedef struct ta_model ta_model_t;

/*
 * 先建立带类型的 image 视图 再展开 runtime 对象
 * 这样可以集中校验一次 offset
 * 后续从 image 支撑的指针建立运行时语义
 * 而不复制每一张结构表
 */
typedef struct {
    const confinfer_partition_image_header_t *header;
    const confinfer_model_image_layer_desc_t *layers;
    const confinfer_model_image_value_desc_t *values;
    const confinfer_model_image_layer_io_t *layer_ios;
    const confinfer_model_image_value_ref_t *input_refs;
    const confinfer_model_image_value_ref_t *output_refs;
    const confinfer_model_image_param_ref_t *param_refs;
    const uint8_t *attr_blob;
} ta_partition_image_view_t;

typedef struct {
    uint32_t elem_count;
    uint32_t ndim;
    uint32_t dims[CONFINFER_VALUE_MAX_DIMS];
} ta_shape_t;

typedef struct {
    ta_shape_t shape;
    uint32_t dtype;
    uint32_t location;
    uint32_t flags;
    uint32_t byte_size;
    void *ptr;
} ta_data_t;

struct ta_value {
    /*
     * 即使 data 可以指回 image 仍保留 ta_value 运行时语义对象
     * 因为执行辅助函数仍需要稳定且带类型的句柄
     */
    confinfer_value_id_t value_id;
    confinfer_layer_id_t producer_layer_id;
    uint32_t output_index;
    uint32_t kind;
    uint32_t role_flags;
    ta_data_t data;
};

typedef struct {
    confinfer_param_id_t param_id;
    uint32_t role;
} ta_param_ref_t;

typedef struct {
    uint32_t size;
    const void *ptr;
} ta_layer_attr_t;

struct ta_layer {
    /*
     * 此处不嵌入复制后的 input output param 数组
     * 而是保留 ref 与 param 切片起点
     * 从而让 layer 保持为 image 布局上的轻量运行时视图
     */
    confinfer_layer_id_t layer_id;
    uint32_t layer_type;
    uint32_t layer_flags;
    uint32_t attr_flags;
    uint32_t input_value_count;
    uint32_t output_value_count;
    uint32_t param_ref_count;
    const confinfer_model_image_value_ref_t *input_refs;
    const confinfer_model_image_value_ref_t *output_refs;
    const confinfer_model_image_param_ref_t *param_refs;
    uint32_t param_begin;
    ta_layer_attr_t attr;
};

struct ta_param {
    confinfer_param_id_t param_id;
    confinfer_layer_id_t owner_layer_id;
    confinfer_partition_id_t owner_partition_id;
    uint32_t flags;
    uint32_t role;
    ta_data_t data;
};

struct ta_partition {
    /*
     * 每个 partition 都需要展开 因为执行以 partition 为作用域
     * 此对象向 backend 提供清晰的运行时图视图
     * 同时静态字节仍保留在 model 所拥有的 image 内
     */
    confinfer_partition_id_t partition_id;
    uint32_t domain;
    uint32_t unit_type;
    uint32_t flags;
    uint32_t layer_count;
    uint32_t input_count;
    uint32_t output_count;
    uint32_t internal_count;
    uint32_t value_count;
    uint32_t param_count;
    ta_layer_t *layers;
    ta_value_t *values;
    ta_param_t *params;
};

struct ta_model {
    /*
     * model 拥有完整 image 与 partition 运行时视图
     * 这一所有权边界使 session 上传缓冲区保持短暂
     * 并让 unload 逻辑保持明确
     */
    confinfer_model_id_t model_id;
    uint32_t flags;
    uint32_t partition_count;
    uint32_t param_count;
    uint32_t is_registered;
    ta_partition_t *partitions;
    void *image_data;
    uint32_t image_size;
    uint32_t image_owned;
    size_t image_mapping_size;
};

typedef struct {
    uint32_t model_count;
    ta_model_t *models[CONFINFER_TA_MAX_MODELS];
} ta_model_store_t;

/*
 * runtime API 分为两层
 * model image 访问函数回答字节位于哪里
 * 展开函数回答如何由这些字节建立执行对象
 */
void ta_runtime_init(void);
void ta_runtime_deinit(void);
ta_model_store_t *ta_runtime_store(void);

ta_model_t *ta_model_find(confinfer_model_id_t model_id);
TEE_Result ta_model_ensure(confinfer_model_id_t model_id, ta_model_t **out_model);
TEE_Result ta_model_load_image(ta_model_t *model,
                               const void *image_data,
                               size_t image_size);
TEE_Result ta_model_attach_image(ta_model_t *model,
                                 void *image_data,
                                 size_t image_size,
                                 size_t image_mapping_size);
void ta_model_release(ta_model_t *model);

const confinfer_model_image_header_t *ta_model_image_header(const ta_model_t *model);
const confinfer_model_image_partition_entry_t *ta_model_partition_entry(
    const ta_model_t *model,
    confinfer_partition_id_t partition_id);
const confinfer_model_image_param_desc_t *ta_model_param_descs(const ta_model_t *model);
const uint8_t *ta_model_param_data(const ta_model_t *model);
const uint8_t *ta_model_partition_image_data(
    const ta_model_t *model,
    const confinfer_model_image_partition_entry_t *entry);
const confinfer_model_image_param_desc_t *ta_model_find_param_desc(
    const ta_model_t *model,
    confinfer_param_id_t param_id);
const void *ta_model_param_buffer(const ta_model_t *model,
                                  const confinfer_model_image_param_desc_t *desc);
TEE_Result ta_model_open_partition_image(const ta_model_t *model,
                                         confinfer_partition_id_t partition_id,
                                         ta_partition_image_view_t *view);

TEE_Result ta_model_materialize_partition(const ta_model_t *model,
                                          confinfer_partition_id_t partition_id,
                                          ta_partition_t *part);
ta_partition_t *ta_model_find_partition(ta_model_t *model,
                                        confinfer_partition_id_t partition_id);
void ta_partition_release(ta_partition_t *part);
ta_value_t *ta_partition_find_value_by_id(ta_partition_t *part,
                                          confinfer_value_id_t value_id);

#endif
