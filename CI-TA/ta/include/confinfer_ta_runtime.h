#ifndef CONFINFER_TA_RUNTIME_H
#define CONFINFER_TA_RUNTIME_H

#include <tee_internal_api.h>

#include <confinfer_protocol.h>

#define CONFINFER_TA_MAX_MODELS 8u

enum ta_value_role_flags {
    TA_VALUE_ROLE_NONE         = 0,
    TA_VALUE_ROLE_INPUT        = 1u << 0,
    TA_VALUE_ROLE_OUTPUT       = 1u << 1,
    TA_VALUE_ROLE_INTERNAL     = 1u << 2,
};

enum ta_param_role_id {
    TA_PARAM_ROLE_UNKNOWN = 0,
    TA_PARAM_ROLE_WEIGHT,
    TA_PARAM_ROLE_BIAS,
    TA_PARAM_ROLE_RUNNING_MEAN,
    TA_PARAM_ROLE_RUNNING_VAR,
};

typedef struct ta_value ta_value_t;
typedef struct ta_layer ta_layer_t;
typedef struct ta_param ta_param_t;
typedef struct ta_partition ta_partition_t;
typedef struct ta_model ta_model_t;
typedef struct ta_value_index ta_value_index_t;
typedef struct ta_param_index ta_param_index_t;
typedef struct ta_partition_index ta_partition_index_t;

typedef struct {
    uint32_t elem_count;
    uint32_t ndim;
    uint32_t dims[CONFINFER_VALUE_MAX_DIMS];
} ta_shape_t;

// 对应 REE 侧 Data_t，承接底层 shape / dtype / location / buffer 描述。
typedef struct {
    ta_shape_t shape;
    uint32_t dtype;
    uint32_t location;
    uint32_t flags;
    uint32_t byte_size;
    void *ptr;
} ta_data_t;

// 对应 REE 侧 Value_t，保存 value 本身的全局语义与底层数据描述。
struct ta_value {
    confinfer_value_id_t value_id;
    // 全局模型语义上的真实生产者 layer_id。
    // 即使该 producer 不在当前 partition 内，这里也应保留。
    confinfer_layer_id_t producer_layer_id;
    uint32_t output_index;
    uint32_t kind;
    uint32_t role_flags;
    ta_data_t data;
};

typedef struct {
    uint32_t start_dim;
    uint32_t end_dim;
} ta_flatten_attr_t;

typedef struct {
    int32_t dim;
} ta_axis_attr_t;

typedef struct {
    uint32_t stride_h;
    uint32_t stride_w;
    uint32_t pad_h;
    uint32_t pad_w;
    uint32_t dilation_h;
    uint32_t dilation_w;
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t groups;
    uint32_t has_bias;
} ta_conv2d_attr_t;

typedef struct {
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t stride_h;
    uint32_t stride_w;
    uint32_t pad_h;
    uint32_t pad_w;
    uint32_t ceil_mode;
    uint32_t count_include_pad;
} ta_pool2d_attr_t;

typedef struct {
    float eps;
    uint32_t groups;
    uint32_t normalized_ndim;
    uint32_t normalized_dims[CONFINFER_VALUE_MAX_DIMS];
} ta_norm_attr_t;

typedef struct {
    float p;
} ta_dropout_attr_t;

typedef struct {
    uint32_t size;
    void *ptr;
} ta_layer_attr_t;

typedef struct {
    confinfer_param_id_t param_id;
    uint32_t role;
} ta_param_ref_t;

struct ta_value_index {
    confinfer_value_id_t value_id;
    ta_value_t *value;
};

struct ta_param_index {
    confinfer_param_id_t param_id;
    ta_param_t *param;
};

struct ta_partition_index {
    confinfer_partition_id_t partition_id;
    ta_partition_t *partition;
};
// 上面两个索引表只用于初始化/绑定阶段按 id 快速解析对象；
// 真正执行阶段仍然直接走 layer->inputs / outputs / params 指针关系。

// ======================================================================
// 对应 REE 侧 Layer 不引入 LayerSlice 直接按单核串行执行语义建模
struct ta_layer {
    confinfer_layer_id_t layer_id;
    uint32_t layer_type;
    uint32_t layer_flags;
    uint32_t attr_flags;
    uint32_t input_value_count;
    uint32_t output_value_count;
    uint32_t param_ref_count;
    confinfer_value_id_t *input_value_ids;
    confinfer_value_id_t *output_value_ids;
    ta_value_t **inputs;
    ta_value_t **outputs;
    ta_param_ref_t *param_refs;
    ta_param_t **params;
    ta_layer_attr_t attr;
};

// 对应 REE 侧常驻参数对象，保存参数本身的全局语义。
struct ta_param {
    confinfer_param_id_t param_id;
    confinfer_layer_id_t owner_layer_id;
    confinfer_partition_id_t owner_partition_id;
    uint32_t flags;
    uint32_t role;
    ta_data_t data;
};

// 对应 REE 侧 ExecutionPartition，本质上是 model 中一组 layer/value 的执行视图。
struct ta_partition {
    confinfer_partition_id_t partition_id;
    uint32_t domain;
    uint32_t unit_type;
    uint32_t flags;

    // model 视角下的引用集合
    uint32_t layer_count;
    uint32_t input_count;
    uint32_t output_count;
    uint32_t internal_count;
    confinfer_layer_id_t *layer_ids;
    ta_layer_t **topo;
    ta_value_t **inputs;
    ta_value_t **outputs;
    ta_value_t **internals;

    // 当前 backend 直接消费分区内的运行时对象
    uint32_t value_count;
    ta_layer_t *layers;
    ta_value_t *values;
    // value_id -> ta_value_t* 的加速索引，不承载额外语义
    ta_value_index_t *value_index;
};

// 对应 REE 侧模型上下文，作为 TA 内长期对象的唯一根。
struct ta_model {
    confinfer_model_id_t model_id;
    uint32_t flags;
    uint32_t expected_partition_count;
    uint32_t expected_param_count;
    uint32_t is_registered;

    // model 级长期对象。
    // layers / params 是长期注册表；
    // 真正执行时仍然以 partition 内已绑定好的指针关系为准。
    uint32_t layer_count;
    uint32_t partition_count;
    uint32_t param_count;
    ta_layer_t *layers;
    ta_partition_t *partitions;
    ta_param_t *params;
    // partition_id -> ta_partition_t* 的加速索引，不承载额外语义
    ta_partition_index_t *partition_index;
    // param_id -> ta_param_t* 的加速索引，不承载额外语义
    ta_param_index_t *param_index;
};

typedef struct {
    uint32_t model_count;
    ta_model_t *models[CONFINFER_TA_MAX_MODELS];
} ta_model_store_t;
// 其实现在一般都是 单个模型的 
// 但是还是先预留了多个模型的 假设
// ta_model_store_t 这里就作为两个世界交互种保存的 全局模型 
// 里面挂多个 ta_model_t
// 每个 ta_model_t 再挂 layers / values / partitions / params

void ta_runtime_init(void);
void ta_runtime_deinit(void);
ta_model_store_t *ta_runtime_store(void);

void ta_partition_reset(ta_partition_t *part);
void ta_partition_release(ta_partition_t *part);
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
    const confinfer_layer_param_ref_t *param_refs);

ta_model_t *ta_model_find(confinfer_model_id_t model_id);
ta_partition_t *ta_model_find_partition(ta_model_t *model,
                                        confinfer_partition_id_t partition_id);
ta_value_t *ta_partition_find_value_by_id(ta_partition_t *part, confinfer_value_id_t value_id);
ta_param_t *ta_model_find_param(ta_model_t *model, confinfer_param_id_t param_id);
// 按 model_id 创建或查找模型
TEE_Result ta_model_ensure(confinfer_model_id_t model_id,
                           ta_model_t **out_model);
// 正式注册模型上下文，重置旧内容并写入模型级元信息
TEE_Result ta_model_register(const confinfer_model_desc_t *desc,
                             ta_model_t **out_model);
TEE_Result ta_model_load_params(ta_model_t *model,
                                const confinfer_load_params_req_t *req,
                                const confinfer_param_desc_t *param_descs,
                                const void *param_blob,
                                size_t param_blob_size);
// 把 partition 挂到 model 下面
TEE_Result ta_model_upsert_partition(ta_model_t *model,
                                     const ta_partition_t *part);
void ta_model_release(ta_model_t *model);

#endif
