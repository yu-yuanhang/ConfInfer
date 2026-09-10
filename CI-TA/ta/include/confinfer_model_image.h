#ifndef CONFINFER_MODEL_IMAGE_H
#define CONFINFER_MODEL_IMAGE_H

#include <stdint.h>

#include <confinfer_protocol.h>

/*
 * ModelImage 是 REE -> TEE 之间传递“TEE 子模型”的统一连续镜像格式。
 *
 * 这份格式的目标不是表达完整训练框架语义，而是表达：
 * 1. 当前模型里需要交给 TEE 的所有 partition
 * 2. 每个 partition 的 layer/value/param 引用关系
 * 3. 参数数据在整份镜像中的连续存放方式
 * 4. partition 内运行时 value buffer 的预留布局
 *
 * 当前设计的核心思想：
 * - REE 侧先把模型结构信息和参数信息打包成一块连续的 image
 * - TEE 侧拿到 image 后，以“偏移 + 视图”的方式解析
 * - 运行时语义对象可以存在，但尽量不要再复制静态结构和参数数据
 * - 默认 bridge 下如果共享内存不足，允许按 chunk 逐步传输；
 *   但 chunk 只是传输手段，不改变 image 本身的逻辑布局
 *
 * 整体布局分两层：
 *
 * model image
 * +--------------------------------------------------------------+
 * | confinfer_model_image_header_t                               |
 * +--------------------------------------------------------------+
 * | partition table : confinfer_model_image_partition_entry_t[]  |
 * +--------------------------------------------------------------+
 * | param desc table : confinfer_model_image_param_desc_t[]      |
 * +--------------------------------------------------------------+
 * | partition image blob #0                                      |
 * +--------------------------------------------------------------+
 * | partition image blob #1                                      |
 * +--------------------------------------------------------------+
 * | ...                                                          |
 * +--------------------------------------------------------------+
 * | contiguous param data area                                   |
 * +--------------------------------------------------------------+
 *
 * 其中：
 * - 顶层 header 描述整份 image 的边界、版本、分区数、参数区位置等
 * - partition table 给出 partition_id -> 子镜像偏移/大小 的映射
 * - param desc table 描述全局参数元数据
 * - param data area 顺序保存所有参数字节
 *
 * 每个 partition image 仍然是一块线性连续布局：
 *
 * partition image
 * +--------------------------------------------------------------+
 * | confinfer_partition_image_header_t                           |
 * +--------------------------------------------------------------+
 * | layer desc table : confinfer_model_image_layer_desc_t[]      |
 * +--------------------------------------------------------------+
 * | value desc table : confinfer_model_image_value_desc_t[]      |
 * +--------------------------------------------------------------+
 * | layer io table : confinfer_model_image_layer_io_t[]          |
 * +--------------------------------------------------------------+
 * | input refs : confinfer_model_image_value_ref_t[]             |
 * +--------------------------------------------------------------+
 * | output refs : confinfer_model_image_value_ref_t[]            |
 * +--------------------------------------------------------------+
 * | param refs : confinfer_model_image_param_ref_t[]             |
 * +--------------------------------------------------------------+
 * | attr blob                                                    |
 * +--------------------------------------------------------------+
 * | runtime data area                                            |
 * +--------------------------------------------------------------+
 *
 * 当前 partition image 的语义边界是：
 * - 前半部分是静态结构描述区，只读
 * - attr blob 是各类 layer attr 的原始连续字节区，只读
 * - runtime data area 是 value 的运行期可写数据区
 *
 * 也就是说，当前实现里：
 * - param 数据来自 model image 顶层参数区
 * - value 的 data.ptr 指向 partition image 的 runtime data area
 * - layer 的 input/output/param 关系通过 ref table 建立
 *
 * 因此 TEE 在解析 image 时，主要做的是：
 * - 校验各 offset / size 是否合法
 * - 建立 ta_model / ta_partition / ta_layer / ta_value / ta_param 这些语义对象
 * - 让这些对象尽量“指向 image”，而不是把 image 内容拆成多份复制
 *
 * 这个头文件只负责“镜像格式定义”。
 * 它不负责：
 * - CA/TA 命令调用协议
 * - bridge 传输策略
 * - TEE 运行时对象实现
 *
 * 这三者分别属于：
 * - confinfer_protocol.h
 * - bridge / host 层
 * - confinfer_ta_runtime.h / .c
 */

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
    /* 预留给后续 TrustSpan / 物理连续映射场景。默认 bridge 下不使用。 */
    uint64_t reserved_phys_base;
    uint64_t reserved_phys_size;
} confinfer_model_image_header_t;

/* partition table 的每个表项描述一块 partition 子镜像在顶层 image 中的位置。 */
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

/* 全局参数描述表。真正的参数字节统一存放在顶层 param data area 中。 */
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

/* partition 子镜像头，给出 partition 内部各静态区和运行区的偏移。 */
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

/* 每个 layer 的基础描述，不直接带输入输出数组，关系由 ref table 表达。 */
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

/* value 描述 partition 内 tensor 语义，data_offset 指向 runtime_data area。 */
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

/* layer 到 input/output/param ref 表的切片索引。 */
typedef struct {
    confinfer_layer_id_t layer_id;
    uint32_t input_ref_begin;
    uint32_t input_ref_count;
    uint32_t output_ref_begin;
    uint32_t output_ref_count;
    uint32_t param_ref_begin;
    uint32_t param_ref_count;
} confinfer_model_image_layer_io_t;

/* layer input/output 通过 value_id 关联到 partition value 表。 */
typedef struct {
    confinfer_value_id_t value_id;
    uint32_t reserved0;
} confinfer_model_image_value_ref_t;

/* layer 参数通过 param_id 关联到顶层 param desc / param data。 */
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
