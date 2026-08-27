#ifndef __MODEL_IMAGE_FORMAT_H_CA__
#define __MODEL_IMAGE_FORMAT_H_CA__

#include <confinfer_model_image.h>

namespace Kernel {
namespace image {

enum class ModelImageExecMode : uint32_t {
    TEE_SINGLE = CONFINFER_MODEL_IMAGE_EXEC_TEE_SINGLE,
    TEE_PARALLEL = CONFINFER_MODEL_IMAGE_EXEC_TEE_PARALLEL,
    TEE_PARALLEL_TRUSTSPAN = CONFINFER_MODEL_IMAGE_EXEC_TEE_PARALLEL_TRUSTSPAN,
};

enum class ModelImageValueRole : uint16_t {
    PARTITION_INPUT = CONFINFER_MODEL_IMAGE_VALUE_INPUT,
    PARTITION_OUTPUT = CONFINFER_MODEL_IMAGE_VALUE_OUTPUT,
    INTERNAL = CONFINFER_MODEL_IMAGE_VALUE_INTERNAL,
};

using ModelImageHeader = ::confinfer_model_image_header_t;
using ModelImagePartitionEntry = ::confinfer_model_image_partition_entry_t;
using ModelImageParamDesc = ::confinfer_model_image_param_desc_t;
using ModelPartitionImageHeader = ::confinfer_partition_image_header_t;
using ModelImageLayerDesc = ::confinfer_model_image_layer_desc_t;
using ModelImageValueDesc = ::confinfer_model_image_value_desc_t;
using ModelImageLayerIO = ::confinfer_model_image_layer_io_t;
using ModelImageValueRef = ::confinfer_model_image_value_ref_t;
using ModelImageParamRef = ::confinfer_model_image_param_ref_t;
using ModelImageFlattenAttr = ::confinfer_model_image_flatten_attr_t;
using ModelImageAxisAttr = ::confinfer_model_image_axis_attr_t;
using ModelImageConvAttr = ::confinfer_model_image_conv_attr_t;
using ModelImagePoolAttr = ::confinfer_model_image_pool_attr_t;
using ModelImageAdaptivePoolAttr = ::confinfer_model_image_adaptive_pool_attr_t;
using ModelImageBatchNormAttr = ::confinfer_model_image_batchnorm_attr_t;
using ModelImageNormAttr = ::confinfer_model_image_norm_attr_t;
using ModelImageDropoutAttr = ::confinfer_model_image_dropout_attr_t;
using ModelImageLinearAttr = ::confinfer_model_image_linear_attr_t;
using ModelImageAddAttr = ::confinfer_model_image_add_attr_t;
using ModelImageBiasAddAttr = ::confinfer_model_image_bias_add_attr_t;

} // namespace image
} // namespace Kernel

#endif
