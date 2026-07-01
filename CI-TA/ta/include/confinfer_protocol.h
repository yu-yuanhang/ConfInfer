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

#define CONFINFER_INVALID_MODEL_ID      UINT32_C(0xffffffff)
#define CONFINFER_INVALID_PARTITION_ID  UINT32_C(0xffffffff)
#define CONFINFER_INVALID_LAYER_ID      UINT32_C(0xffffffff)
#define CONFINFER_INVALID_PARAM_ID      UINT32_C(0xffffffff)

enum confinfer_command_id {
    CONFINFER_CMD_REGISTER_MODEL = 0,
    CONFINFER_CMD_LOAD_PARAMS = 1,
    CONFINFER_CMD_REGISTER_PARTITION = 2,
    CONFINFER_CMD_EXEC_PARTITION = 3,
    CONFINFER_CMD_UNLOAD_MODEL = 4,

    // 下面保留给当前 smoke/debug 路径使用，不属于长期推理命令体系主体。
    CONFINFER_CMD_DEBUG_INC_VALUE = 0x100,
    CONFINFER_CMD_DEBUG_DEC_VALUE = 0x101,
};

enum confinfer_exec_domain_id {
    CONFINFER_DOMAIN_DEFAULT = 0,
    CONFINFER_DOMAIN_CPU_REE = 1,
    CONFINFER_DOMAIN_CPU_TEE = 2,
};

enum confinfer_exec_unit_type_id {
    CONFINFER_UNIT_LAYER = 0,
    CONFINFER_UNIT_PARTITION = 1,
};

// 必须与 REE 侧 LayerType 枚举保持一致
enum confinfer_layer_type_id {
    CONFINFER_LAYER_GRAPH_INPUT = 0,
    CONFINFER_LAYER_GRAPH_OUTPUT,
    CONFINFER_LAYER_CONV2D,
    CONFINFER_LAYER_MAXPOOL2D,
    CONFINFER_LAYER_AVGPOOL2D,
    CONFINFER_LAYER_ADAPTIVEAVGPOOL2D,
    CONFINFER_LAYER_ADAPTIVEMAXPOOL2D,
    CONFINFER_LAYER_BATCHNORM2D,
    CONFINFER_LAYER_LAYERNORM,
    CONFINFER_LAYER_GROUPNORM,
    CONFINFER_LAYER_RELU,
    CONFINFER_LAYER_SIGMOID,
    CONFINFER_LAYER_DROPOUT,
    CONFINFER_LAYER_FLATTEN,
    CONFINFER_LAYER_BIASADD,
    CONFINFER_LAYER_MATMUL,
    CONFINFER_LAYER_LINEAR,
    CONFINFER_LAYER_SOFTMAX,
    CONFINFER_LAYER_ADD,
    CONFINFER_LAYER_MUL,
    CONFINFER_LAYER_CONCAT,
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

enum confinfer_partition_status_code {
    CONFINFER_PART_OK = 0,
    CONFINFER_PART_UNSUPPORTED_DOMAIN = 1,
    CONFINFER_PART_UNSUPPORTED_UNIT = 2,
    CONFINFER_PART_BAD_LAYER_DESC = 3,
    CONFINFER_PART_BAD_DATA_DESC = 4,
};

enum confinfer_model_status_code {
    CONFINFER_MODEL_OK = 0,
    CONFINFER_MODEL_BAD_DESC = 1,
};

enum confinfer_param_status_code {
    CONFINFER_PARAM_OK = 0,
    CONFINFER_PARAM_BAD_DESC = 1,
};

enum confinfer_unload_model_status_code {
    CONFINFER_UNLOAD_MODEL_OK = 0,
    CONFINFER_UNLOAD_MODEL_NOT_FOUND = 1,
};

typedef struct {
    uint32_t version;
    confinfer_model_id_t model_id;
    uint32_t flags;
    uint32_t expected_partition_count;
    uint32_t expected_param_count;
    uint32_t reserved0;
    uint32_t reserved1;
} confinfer_model_desc_t;
// 关于模型上下文的注册描述信息

typedef struct {
    uint32_t version;
    uint32_t status;
    confinfer_model_id_t model_id;
    uint32_t flags;
    uint32_t partition_count;
    uint32_t param_count;
    uint32_t reserved0;
    uint32_t reserved1;
} confinfer_model_rsp_t;
// 专门用来表示模型上下文注册后的反馈
// 对于 TA 没有什么意义 主要用来输出一下 

typedef struct {
    uint32_t version;
    confinfer_model_id_t model_id;
    uint32_t param_count;
    uint32_t total_param_bytes;
    uint32_t flags;
    uint32_t reserved0;
    uint32_t reserved1;
} confinfer_load_params_req_t;
// 关于本次参数加载的请求头

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
} confinfer_param_desc_t;
// 关于单个常驻参数的元信息描述

typedef struct {
    uint32_t version;
    uint32_t status;
    confinfer_model_id_t model_id;
    uint32_t loaded_param_count;
    uint32_t total_param_bytes;
    uint32_t reserved0;
    uint32_t reserved1;
} confinfer_load_params_rsp_t;
// 专门用来表示参数加载后的反馈

typedef struct {
    uint32_t version;
    confinfer_model_id_t model_id;
    uint32_t flags;
    uint32_t reserved0;
    uint32_t reserved1;
} confinfer_unload_model_req_t;
// 关于模型卸载的请求头

typedef struct {
    uint32_t version;
    uint32_t status;
    confinfer_model_id_t model_id;
    uint32_t released_partition_count;
    uint32_t released_param_count;
    uint32_t reserved0;
    uint32_t reserved1;
} confinfer_unload_model_rsp_t;
// 专门用来表示模型卸载后的反馈

typedef struct {
    uint32_t version;
    uint32_t domain;
    uint32_t unit_type;
    confinfer_model_id_t model_id;
    confinfer_partition_id_t partition_id;
    uint32_t layer_count;
    uint32_t input_count;
    uint32_t output_count;
    uint32_t flags;
    uint32_t reserved;
} confinfer_partition_req_t;
// 关于 partition 的描述信息
// 这个 partition 本身就是 调用对象所以也可以作为协议头

typedef struct {
    confinfer_layer_id_t layer_id;
    uint32_t layer_type;
    uint32_t layer_flags;
    uint32_t attr_offset;
    uint32_t attr_size;
    uint32_t reserved;
} confinfer_layer_desc_t;
// 关于 Layer 的描述信息
// 一个 partition 后承载多个 Layer 

typedef struct {
    int32_t start_dim;
    int32_t end_dim;
} confinfer_flatten_attr_t;

typedef struct {
    int32_t dim;
    uint32_t reserved;
} confinfer_axis_attr_t;

typedef struct {
    uint32_t in_channels;
    uint32_t out_channels;
    uint32_t groups;
    uint32_t has_bias;
    uint32_t padding_mode;
    uint32_t spatial_dim;
    uint32_t kernel_size[CONFINFER_VALUE_MAX_DIMS];
    uint32_t stride[CONFINFER_VALUE_MAX_DIMS];
    int32_t padding[CONFINFER_VALUE_MAX_DIMS * 2];
    uint32_t padding_count;
    uint32_t dilation[CONFINFER_VALUE_MAX_DIMS];
} confinfer_conv_attr_t;

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
} confinfer_pool_attr_t;

typedef struct {
    uint32_t output_ndim;
    uint32_t output_size[CONFINFER_VALUE_MAX_DIMS];
    uint32_t return_indices;
    uint32_t reserved;
} confinfer_adaptive_pool_attr_t;

typedef struct {
    float eps;
    uint32_t num_features;
    uint32_t affine;
    uint32_t track_running_stats;
    float momentum;
} confinfer_batchnorm_attr_t;

typedef struct {
    float eps;
    uint32_t affine;
    uint32_t num_groups;
    uint32_t num_channels;
    uint32_t normalized_ndim;
    uint32_t normalized_shape[CONFINFER_VALUE_MAX_DIMS];
} confinfer_norm_attr_t;

typedef struct {
    float p;
    uint32_t inplace;
} confinfer_dropout_attr_t;

typedef struct {
    uint32_t in_features;
    uint32_t out_features;
    uint32_t has_bias;
    uint32_t reserved;
} confinfer_linear_attr_t;

typedef struct {
    float alpha;
} confinfer_add_attr_t;

typedef struct {
    uint32_t size;
    int32_t dim;
} confinfer_bias_add_attr_t;

typedef struct {
    uint32_t version;
    uint32_t input_count;
    uint32_t output_count;
    uint32_t internal_count;
    uint32_t layer_io_count;
    uint32_t input_ref_count;
    uint32_t output_ref_count;
    uint32_t param_ref_count;
    uint32_t total_input_bytes;
    uint32_t total_output_bytes;
    uint32_t flags;
    uint32_t reserved0;
    uint32_t reserved1;
} confinfer_partition_data_req_t;
// 关于本次执行的数据面描述信息

typedef struct {
    confinfer_value_id_t value_id;
    uint32_t producer_layer_id;
    uint32_t output_index;
    uint32_t kind;
    uint32_t dtype;
    uint32_t location;
    uint32_t flags;
    uint32_t elem_count;
    uint32_t byte_size;
    uint32_t ndim;
    uint32_t dims[CONFINFER_VALUE_MAX_DIMS];
} confinfer_value_desc_t;
// 关于单个输入输出 Value 的元信息描述

typedef struct {
    confinfer_value_id_t value_id;
    uint32_t reserved;
} confinfer_layer_value_ref_t;
// 直接引用稳定 value_id

typedef struct {
    confinfer_param_id_t param_id;
    uint32_t role;
} confinfer_layer_param_ref_t;
// 某个 layer 用到一个参数 参数身份是 param_id，角色是 WEIGHT/BIAS/...

typedef struct {
    uint32_t layer_id;
    // 在引用表中查找 
    // input_refs[start] - input_refs[start + count - 1]
    // 至于这个位置上的连续性 由 append_layer_refs 函数保证
    uint32_t input_ref_start;
    uint32_t input_ref_count;
    uint32_t output_ref_start;
    uint32_t output_ref_count;
    uint32_t param_ref_start;
    uint32_t param_ref_count;
} confinfer_layer_io_desc_t;
// 记录某个 Layer 的输入/输出/参数引用区间信息
// 这个 layer 的输入引用从哪开始 输出引用从哪开始 参数引用从哪开始

typedef struct {
    uint32_t version;
    uint32_t status;
    uint32_t domain;
    confinfer_model_id_t model_id;
    confinfer_partition_id_t partition_id;
    uint32_t executed_layers;
    uint32_t consumed_inputs;
    uint32_t produced_outputs;
    uint32_t reserved0;
    uint32_t reserved1;
} confinfer_partition_rsp_t;
// 专门用来表示 TA 执行完后的反馈

#endif
