#ifndef CONFINFER_PROTOCOL_H
#define CONFINFER_PROTOCOL_H

#include <stdint.h>

#define CONFINFER_PROTOCOL_VERSION 1u
#define CONFINFER_VALUE_MAX_DIMS 6u

typedef uint32_t confinfer_model_id_t;
typedef uint32_t confinfer_partition_id_t;
typedef uint32_t confinfer_layer_id_t;
typedef uint32_t confinfer_value_id_t;
typedef uint32_t confinfer_param_id_t;

#define CONFINFER_INVALID_MODEL_ID UINT32_C(0xffffffff)
#define CONFINFER_INVALID_PARTITION_ID UINT32_C(0xffffffff)
#define CONFINFER_INVALID_LAYER_ID UINT32_C(0xffffffff)
#define CONFINFER_INVALID_PARAM_ID UINT32_C(0xffffffff)

enum confinfer_command_id {
    CONFINFER_CMD_PREPARE_MODEL_IMAGE = 0,
    CONFINFER_CMD_EXEC_PARTITION = 1,
    CONFINFER_CMD_UNLOAD_MODEL = 2,
    CONFINFER_CMD_PREPARE_MODEL_IMAGE_BEGIN = 0x10,
    CONFINFER_CMD_PREPARE_MODEL_IMAGE_CHUNK = 0x11,
    CONFINFER_CMD_PREPARE_MODEL_IMAGE_END = 0x12,
    CONFINFER_CMD_EXEC_PARTITION_BEGIN = 0x20,
    CONFINFER_CMD_EXEC_PARTITION_INPUT_CHUNK = 0x21,
    CONFINFER_CMD_EXEC_PARTITION_RUN = 0x22,
    CONFINFER_CMD_EXEC_PARTITION_OUTPUT_CHUNK = 0x23,
    CONFINFER_CMD_EXEC_PARTITION_END = 0x24,
    CONFINFER_CMD_DEBUG_INC_VALUE = 0x100,
    CONFINFER_CMD_DEBUG_DEC_VALUE = 0x101,
};

enum confinfer_status_code {
    CONFINFER_STATUS_OK = 0,
    CONFINFER_STATUS_BAD_REQUEST = 1,
    CONFINFER_STATUS_NOT_FOUND = 2,
    CONFINFER_STATUS_NOT_READY = 3,
    CONFINFER_STATUS_INTERNAL_ERROR = 4,
};

enum confinfer_data_type_id {
    CONFINFER_DTYPE_FP32 = 0,
    CONFINFER_DTYPE_FP16 = 1,
    CONFINFER_DTYPE_INT8 = 2,
    CONFINFER_DTYPE_INT32 = 3,
};

enum confinfer_data_location_id {
    CONFINFER_DATA_CPU = 0,
    CONFINFER_DATA_TEE = 1,
};

enum confinfer_param_role_id {
    CONFINFER_PARAM_ROLE_WEIGHT = 0,
    CONFINFER_PARAM_ROLE_BIAS = 1,
    CONFINFER_PARAM_ROLE_RUNNING_MEAN = 2,
    CONFINFER_PARAM_ROLE_RUNNING_VAR = 3,
    CONFINFER_PARAM_ROLE_UNKNOWN = 255,
};

enum confinfer_layer_type_id {
    CONFINFER_LAYER_GRAPH_INPUT = 0,
    CONFINFER_LAYER_GRAPH_OUTPUT = 1,
    CONFINFER_LAYER_CONV2D = 2,
    CONFINFER_LAYER_MAXPOOL2D = 3,
    CONFINFER_LAYER_AVGPOOL2D = 4,
    CONFINFER_LAYER_ADAPTIVEAVGPOOL2D = 5,
    CONFINFER_LAYER_ADAPTIVEMAXPOOL2D = 6,
    CONFINFER_LAYER_BATCHNORM2D = 7,
    CONFINFER_LAYER_LAYERNORM = 8,
    CONFINFER_LAYER_GROUPNORM = 9,
    CONFINFER_LAYER_RELU = 10,
    CONFINFER_LAYER_SIGMOID = 11,
    CONFINFER_LAYER_DROPOUT = 12,
    CONFINFER_LAYER_FLATTEN = 13,
    CONFINFER_LAYER_BIASADD = 14,
    CONFINFER_LAYER_MATMUL = 15,
    CONFINFER_LAYER_LINEAR = 16,
    CONFINFER_LAYER_SOFTMAX = 17,
    CONFINFER_LAYER_ADD = 18,
    CONFINFER_LAYER_MUL = 19,
    CONFINFER_LAYER_CONCAT = 20,
};

typedef struct {
    uint32_t version;
    confinfer_model_id_t model_id;
    uint32_t image_size;
    uint32_t flags;
} confinfer_prepare_model_image_req_t;

typedef struct {
    uint32_t version;
    confinfer_model_id_t model_id;
    uint32_t total_image_size;
    uint32_t flags;
} confinfer_prepare_model_image_begin_req_t;

typedef struct {
    uint32_t version;
    uint32_t status;
    confinfer_model_id_t model_id;
    uint32_t accepted_bytes;
    uint32_t reserved0;
    uint32_t reserved1;
} confinfer_prepare_model_image_begin_rsp_t;

typedef struct {
    uint32_t version;
    confinfer_model_id_t model_id;
    uint32_t chunk_offset;
    uint32_t chunk_size;
    uint32_t total_image_size;
    uint32_t flags;
} confinfer_prepare_model_image_chunk_req_t;

typedef struct {
    uint32_t version;
    uint32_t status;
    confinfer_model_id_t model_id;
    uint32_t next_offset;
    uint32_t accepted_bytes;
    uint32_t reserved0;
} confinfer_prepare_model_image_chunk_rsp_t;

typedef struct {
    uint32_t version;
    confinfer_model_id_t model_id;
    uint32_t total_image_size;
    uint32_t flags;
} confinfer_prepare_model_image_end_req_t;

typedef struct {
    uint32_t version;
    uint32_t status;
    confinfer_model_id_t model_id;
    uint32_t loaded_image_size;
    uint32_t reserved0;
    uint32_t reserved1;
} confinfer_prepare_model_image_rsp_t;

typedef struct {
    uint32_t version;
    confinfer_model_id_t model_id;
    confinfer_partition_id_t partition_id;
    uint32_t input_count;
    uint32_t output_count;
    uint32_t input_bytes;
    uint32_t output_bytes;
    uint32_t flags;
} confinfer_exec_partition_req_t;

typedef struct {
    uint32_t version;
    confinfer_model_id_t model_id;
    confinfer_partition_id_t partition_id;
    uint32_t input_count;
    uint32_t output_count;
    uint32_t total_input_bytes;
    uint32_t total_output_bytes;
    uint32_t flags;
} confinfer_exec_partition_begin_req_t;

typedef struct {
    uint32_t version;
    uint32_t status;
    confinfer_model_id_t model_id;
    confinfer_partition_id_t partition_id;
    uint32_t accepted_input_bytes;
    uint32_t reserved0;
    uint32_t reserved1;
} confinfer_exec_partition_begin_rsp_t;

typedef struct {
    uint32_t version;
    confinfer_model_id_t model_id;
    confinfer_partition_id_t partition_id;
    uint32_t chunk_offset;
    uint32_t chunk_size;
    uint32_t total_input_bytes;
    uint32_t flags;
} confinfer_exec_partition_input_chunk_req_t;

typedef struct {
    uint32_t version;
    uint32_t status;
    confinfer_model_id_t model_id;
    confinfer_partition_id_t partition_id;
    uint32_t next_offset;
    uint32_t accepted_bytes;
    uint32_t reserved0;
} confinfer_exec_partition_input_chunk_rsp_t;

typedef struct {
    uint32_t version;
    confinfer_model_id_t model_id;
    confinfer_partition_id_t partition_id;
    uint32_t input_count;
    uint32_t output_count;
    uint32_t total_input_bytes;
    uint32_t total_output_bytes;
    uint32_t flags;
} confinfer_exec_partition_run_req_t;

typedef struct {
    uint32_t version;
    confinfer_model_id_t model_id;
    confinfer_partition_id_t partition_id;
    uint32_t chunk_offset;
    uint32_t chunk_size;
    uint32_t total_output_bytes;
    uint32_t flags;
} confinfer_exec_partition_output_chunk_req_t;

typedef struct {
    uint32_t version;
    uint32_t status;
    confinfer_model_id_t model_id;
    confinfer_partition_id_t partition_id;
    uint32_t next_offset;
    uint32_t copied_bytes;
    uint32_t reserved0;
} confinfer_exec_partition_output_chunk_rsp_t;

typedef struct {
    uint32_t version;
    confinfer_model_id_t model_id;
    confinfer_partition_id_t partition_id;
    uint32_t flags;
} confinfer_exec_partition_end_req_t;

typedef struct {
    uint32_t version;
    uint32_t status;
    confinfer_model_id_t model_id;
    confinfer_partition_id_t partition_id;
    uint32_t consumed_inputs;
    uint32_t produced_outputs;
    uint32_t output_bytes;
    uint32_t reserved0;
} confinfer_exec_partition_rsp_t;

typedef struct {
    uint32_t version;
    confinfer_model_id_t model_id;
    uint32_t flags;
    uint32_t reserved0;
} confinfer_unload_model_req_t;

typedef struct {
    uint32_t version;
    uint32_t status;
    confinfer_model_id_t model_id;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
} confinfer_unload_model_rsp_t;

#endif
