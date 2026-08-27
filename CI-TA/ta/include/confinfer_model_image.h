#ifndef CONFINFER_MODEL_IMAGE_H
#define CONFINFER_MODEL_IMAGE_H

#include <stdint.h>

#include <confinfer_protocol.h>

#define CONFINFER_MODEL_IMAGE_MAGIC UINT32_C(0x43494d47)
#define CONFINFER_MODEL_IMAGE_VERSION_MAJOR UINT16_C(1)
#define CONFINFER_MODEL_IMAGE_VERSION_MINOR UINT16_C(0)
#define CONFINFER_PARTITION_IMAGE_MAGIC UINT32_C(0x4350494d)

enum confinfer_model_image_exec_mode {
    CONFINFER_MODEL_IMAGE_EXEC_TEE_SINGLE = 0,
    CONFINFER_MODEL_IMAGE_EXEC_TEE_PARALLEL = 1,
    CONFINFER_MODEL_IMAGE_EXEC_TEE_PARALLEL_TRUSTSPAN = 2,
};

enum confinfer_model_image_value_role {
    CONFINFER_MODEL_IMAGE_VALUE_INPUT = 0,
    CONFINFER_MODEL_IMAGE_VALUE_OUTPUT = 1,
    CONFINFER_MODEL_IMAGE_VALUE_INTERNAL = 2,
};

typedef struct {
    uint32_t magic;
    uint16_t version_major;
    uint16_t version_minor;
    uint32_t header_size;
    uint32_t total_size;
    confinfer_model_id_t model_id;
    uint32_t exec_mode;
    uint32_t flags;
    uint32_t partition_count;
    uint32_t partition_table_off;
    uint32_t partition_table_size;
    uint32_t param_desc_count;
    uint32_t param_desc_off;
    uint32_t param_data_off;
    uint32_t param_data_size;
    uint64_t reserved_phys_base;
    uint64_t reserved_phys_size;
} confinfer_model_image_header_t;

typedef struct {
    confinfer_partition_id_t partition_id;
    uint32_t flags;
    uint32_t image_off;
    uint32_t image_size;
    uint32_t layer_count;
    uint32_t input_count;
    uint32_t output_count;
    uint32_t internal_count;
} confinfer_model_image_partition_entry_t;

typedef struct {
    confinfer_param_id_t param_id;
    confinfer_layer_id_t owner_layer_id;
    confinfer_partition_id_t owner_partition_id;
    uint32_t role;
    uint32_t dtype;
    uint32_t location;
    uint32_t flags;
    uint32_t elem_count;
    uint32_t byte_size;
    uint32_t data_offset;
    uint32_t ndim;
    uint32_t dims[CONFINFER_VALUE_MAX_DIMS];
} confinfer_model_image_param_desc_t;

typedef struct {
    uint32_t magic;
    uint16_t version_major;
    uint16_t version_minor;
    uint32_t total_size;
    confinfer_partition_id_t partition_id;
    uint32_t flags;
    uint32_t layer_count;
    uint32_t value_count;
    uint32_t input_count;
    uint32_t output_count;
    uint32_t internal_count;
    uint32_t input_ref_count;
    uint32_t output_ref_count;
    uint32_t param_ref_count;
    uint32_t layer_desc_off;
    uint32_t value_desc_off;
    uint32_t layer_io_off;
    uint32_t input_ref_off;
    uint32_t output_ref_off;
    uint32_t param_ref_off;
    uint32_t attr_blob_off;
    uint32_t attr_blob_size;
    uint32_t runtime_data_off;
    uint32_t runtime_data_size;
} confinfer_partition_image_header_t;

typedef struct {
    confinfer_layer_id_t layer_id;
    uint16_t layer_type;
    uint16_t reserved0;
    uint32_t flags;
    uint32_t workspace_bytes;
    uint32_t attr_off;
    uint32_t attr_size;
    uint32_t topo_index;
} confinfer_model_image_layer_desc_t;

typedef struct {
    confinfer_value_id_t value_id;
    uint16_t role;
    uint16_t kind;
    uint32_t flags;
    confinfer_layer_id_t producer_layer_id;
    uint32_t output_index;
    uint32_t dtype;
    uint32_t location;
    uint32_t elem_count;
    uint32_t byte_size;
    uint32_t data_offset;
    uint32_t ndim;
    uint32_t dims[CONFINFER_VALUE_MAX_DIMS];
} confinfer_model_image_value_desc_t;

typedef struct {
    confinfer_layer_id_t layer_id;
    uint32_t input_ref_begin;
    uint32_t input_ref_count;
    uint32_t output_ref_begin;
    uint32_t output_ref_count;
    uint32_t param_ref_begin;
    uint32_t param_ref_count;
} confinfer_model_image_layer_io_t;

typedef struct {
    confinfer_value_id_t value_id;
    uint32_t reserved0;
} confinfer_model_image_value_ref_t;

typedef struct {
    confinfer_param_id_t param_id;
    uint32_t role;
} confinfer_model_image_param_ref_t;

typedef struct {
    int32_t start_dim;
    int32_t end_dim;
} confinfer_model_image_flatten_attr_t;

typedef struct {
    int32_t dim;
    uint32_t reserved0;
} confinfer_model_image_axis_attr_t;

typedef struct {
    uint32_t in_channels;
    uint32_t out_channels;
    uint32_t groups;
    uint32_t has_bias;
    uint32_t spatial_dim;
    uint32_t kernel_size[CONFINFER_VALUE_MAX_DIMS];
    uint32_t stride[CONFINFER_VALUE_MAX_DIMS];
    int32_t padding[CONFINFER_VALUE_MAX_DIMS * 2];
    uint32_t padding_count;
    uint32_t dilation[CONFINFER_VALUE_MAX_DIMS];
} confinfer_model_image_conv_attr_t;

typedef struct {
    uint32_t spatial_dim;
    uint32_t kernel_size[CONFINFER_VALUE_MAX_DIMS];
    uint32_t stride[CONFINFER_VALUE_MAX_DIMS];
    int32_t padding[CONFINFER_VALUE_MAX_DIMS * 2];
    uint32_t padding_count;
    uint32_t dilation[CONFINFER_VALUE_MAX_DIMS];
    uint32_t return_indices;
    uint32_t ceil_mode;
    uint32_t count_include_pad;
    uint32_t divisor_override;
} confinfer_model_image_pool_attr_t;

typedef struct {
    uint32_t output_ndim;
    uint32_t output_size[CONFINFER_VALUE_MAX_DIMS];
    uint32_t return_indices;
    uint32_t reserved0;
} confinfer_model_image_adaptive_pool_attr_t;

typedef struct {
    float eps;
    uint32_t num_features;
    uint32_t affine;
    uint32_t track_running_stats;
    float momentum;
} confinfer_model_image_batchnorm_attr_t;

typedef struct {
    float eps;
    uint32_t affine;
    uint32_t num_groups;
    uint32_t num_channels;
    uint32_t normalized_ndim;
    uint32_t normalized_shape[CONFINFER_VALUE_MAX_DIMS];
} confinfer_model_image_norm_attr_t;

typedef struct {
    float p;
    uint32_t reserved0;
} confinfer_model_image_dropout_attr_t;

typedef struct {
    uint32_t in_features;
    uint32_t out_features;
    uint32_t has_bias;
    uint32_t reserved0;
} confinfer_model_image_linear_attr_t;

typedef struct {
    float alpha;
} confinfer_model_image_add_attr_t;

typedef struct {
    uint32_t size;
    int32_t dim;
} confinfer_model_image_bias_add_attr_t;

#endif
