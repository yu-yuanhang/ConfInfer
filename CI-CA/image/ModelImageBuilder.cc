#include <image/ModelImage.h>

#include <core/ops/activation.h>
#include <core/ops/arithmetic.h>
#include <core/ops/convolution.h>
#include <core/ops/linear.h>
#include <core/ops/normalization.h>
#include <core/ops/pool.h>
#include <core/ops/reshape.h>

#include <algorithm>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Kernel {
namespace image {

namespace {

uint32_t data_type_to_proto(Kernel::core::DataType dtype)
{
    switch (dtype) {
    case Kernel::core::DataType::FP16:
        return CONFINFER_DTYPE_FP16;
    case Kernel::core::DataType::INT8:
        return CONFINFER_DTYPE_INT8;
    case Kernel::core::DataType::INT32:
        return CONFINFER_DTYPE_INT32;
    case Kernel::core::DataType::FP32:
    default:
        return CONFINFER_DTYPE_FP32;
    }
}

uint32_t data_location_to_proto(Kernel::core::DataLocation location)
{
    switch (location) {
    case Kernel::core::DataLocation::TEE:
        return CONFINFER_DATA_TEE;
    case Kernel::core::DataLocation::CPU:
    default:
        return CONFINFER_DATA_CPU;
    }
}

uint32_t param_role_to_proto(Kernel::core::ParamRole role)
{
    switch (role) {
    case Kernel::core::ParamRole::WEIGHT:
        return CONFINFER_PARAM_ROLE_WEIGHT;
    case Kernel::core::ParamRole::BIAS:
        return CONFINFER_PARAM_ROLE_BIAS;
    case Kernel::core::ParamRole::RUNNING_MEAN:
        return CONFINFER_PARAM_ROLE_RUNNING_MEAN;
    case Kernel::core::ParamRole::RUNNING_VAR:
        return CONFINFER_PARAM_ROLE_RUNNING_VAR;
    case Kernel::core::ParamRole::UNKNOWN:
    default:
        return CONFINFER_PARAM_ROLE_UNKNOWN;
    }
}

uint16_t value_role_to_proto(ModelImageValueRole role)
{
    return static_cast<uint16_t>(role);
}

struct CollectedParams {
    std::vector<ModelImageParamDesc> descs;
    std::vector<uint8_t> blob;
};

struct PartitionImageData {
    ModelPartitionImageHeader header;
    std::vector<ModelImageLayerDesc> layers;
    std::vector<ModelImageValueDesc> values;
    std::vector<ModelImageLayerIO> layer_ios;
    std::vector<ModelImageValueRef> input_refs;
    std::vector<ModelImageValueRef> output_refs;
    std::vector<ModelImageParamRef> param_refs;
    std::vector<uint8_t> attr_blob;
};

uint32_t count_tee_partitions(const std::vector<Kernel::core::ExecPartition>& parts)
{
    uint32_t count = 0;

    for (const auto& part : parts) {
        if (part.domain() == Kernel::core::ExecutionDomain::ED_CPU_TEE) {
            ++count;
        }
    }

    return count;
}

void append_param_desc(std::unordered_set<confinfer_param_id_t>& seen,
                       CollectedParams& collected,
                       confinfer_partition_id_t partition_id,
                       const Kernel::core::Layer *layer,
                       Kernel::core::ParamRole role,
                       const Kernel::core::Data_t *param)
{
    ModelImageParamDesc desc{};
    const UINT param_id = layer ? layer->paramId(role) : INVALID_VALUE_U;
    const uint32_t byte_size = param->shape.size * param->getTypeSize();

    EXIT_ERROR_CHECK_EQ(nullptr, layer, "Layer is nullptr");
    EXIT_ERROR_CHECK_EQ(nullptr, param, "Param is nullptr");
    EXIT_ERROR_CHECK_EQ(nullptr, param->ptr, "Param data.ptr is nullptr");
    EXIT_ERROR_CHECK_EQ(INVALID_VALUE_U, param_id, "Layer param_id is invalid");

    desc.param_id = static_cast<confinfer_param_id_t>(param_id);
    if (seen.find(desc.param_id) != seen.end()) {
        return;
    }

    desc.owner_layer_id = layer->id();
    desc.owner_partition_id = partition_id;
    desc.role = param_role_to_proto(role);
    desc.dtype = data_type_to_proto(param->dtype);
    desc.location = data_location_to_proto(param->location);
    desc.flags = param->flags;
    desc.elem_count = param->shape.size;
    desc.byte_size = byte_size;
    desc.data_offset = static_cast<uint32_t>(collected.blob.size());
    desc.ndim = param->shape.ndim;
    EXIT_ERROR_CHECK_EQ(true, desc.ndim > CONFINFER_VALUE_MAX_DIMS,
                        "Param ndim exceeds protocol max dims");
    for (uint32_t i = 0; i < desc.ndim; ++i) {
        desc.dims[i] = param->shape.dims[i];
    }

    const uint8_t *src = static_cast<const uint8_t *>(param->ptr);
    collected.blob.insert(collected.blob.end(), src, src + byte_size);
    collected.descs.push_back(desc);
    seen.insert(desc.param_id);
}

CollectedParams collect_tee_params(const std::vector<Kernel::core::ExecPartition>& parts)
{
    CollectedParams collected;
    std::unordered_set<confinfer_param_id_t> seen;
    const Kernel::core::ParamRole roles[] = {
        Kernel::core::ParamRole::WEIGHT,
        Kernel::core::ParamRole::BIAS,
        Kernel::core::ParamRole::RUNNING_MEAN,
        Kernel::core::ParamRole::RUNNING_VAR,
    };

    for (const auto& part : parts) {
        if (part.domain() != Kernel::core::ExecutionDomain::ED_CPU_TEE) {
            continue;
        }

        for (Kernel::core::Layer *layer : part.topo()) {
            EXIT_ERROR_CHECK_EQ(nullptr, layer, "TEE partition layer is nullptr");
            for (Kernel::core::ParamRole role : roles) {
                const Kernel::core::Data_t *param = layer->param(role);
                if (nullptr == param) {
                    continue;
                }
                append_param_desc(seen, collected,
                                  static_cast<confinfer_partition_id_t>(part.id()),
                                  layer, role, param);
            }
        }
    }

    return collected;
}

template<typename T>
void append_attr_blob(ModelImageLayerDesc& desc, const T& attr, std::vector<uint8_t>& blob)
{
    desc.attr_off = static_cast<uint32_t>(blob.size());
    desc.attr_size = static_cast<uint32_t>(sizeof(T));
    const uint8_t *src = reinterpret_cast<const uint8_t *>(&attr);
    blob.insert(blob.end(), src, src + sizeof(T));
}

void append_axis_attr(ModelImageLayerDesc& desc, INT dim, std::vector<uint8_t>& blob)
{
    ModelImageAxisAttr attr{};
    attr.dim = dim;
    attr.reserved0 = 0;
    append_attr_blob(desc, attr, blob);
}

void append_flatten_attr(ModelImageLayerDesc& desc,
                         const Kernel::core::Flatten_L *flatten,
                         std::vector<uint8_t>& blob)
{
    ModelImageFlattenAttr attr{};
    attr.start_dim = flatten->startDim();
    attr.end_dim = flatten->endDim();
    append_attr_blob(desc, attr, blob);
}

void append_conv_attr(ModelImageLayerDesc& desc,
                      const Kernel::core::ConvNd_L *conv,
                      std::vector<uint8_t>& blob)
{
    ModelImageConvAttr attr{};
    attr.in_channels = conv->inChannels();
    attr.out_channels = conv->outChannels();
    attr.groups = conv->groups();
    attr.has_bias = conv->biasEnabled();
    attr.spatial_dim = conv->spatialDim();
    for (uint32_t i = 0; i < conv->kernelSize().size() && i < CONFINFER_VALUE_MAX_DIMS; ++i)
        attr.kernel_size[i] = conv->kernelSize()[i];
    for (uint32_t i = 0; i < conv->stride().size() && i < CONFINFER_VALUE_MAX_DIMS; ++i)
        attr.stride[i] = conv->stride()[i];
    attr.padding_count = static_cast<uint32_t>(conv->padding().size());
    for (uint32_t i = 0; i < conv->padding().size() && i < CONFINFER_VALUE_MAX_DIMS * 2; ++i)
        attr.padding[i] = conv->padding()[i];
    for (uint32_t i = 0; i < conv->dilation().size() && i < CONFINFER_VALUE_MAX_DIMS; ++i)
        attr.dilation[i] = conv->dilation()[i];
    append_attr_blob(desc, attr, blob);
}

void append_pool_attr(ModelImageLayerDesc& desc,
                      const Kernel::core::PoolNd_L *pool,
                      std::vector<uint8_t>& blob)
{
    ModelImagePoolAttr attr{};
    attr.spatial_dim = pool->spatialDim();
    for (uint32_t i = 0; i < pool->kernelSize().size() && i < CONFINFER_VALUE_MAX_DIMS; ++i)
        attr.kernel_size[i] = pool->kernelSize()[i];
    for (uint32_t i = 0; i < pool->stride().size() && i < CONFINFER_VALUE_MAX_DIMS; ++i)
        attr.stride[i] = pool->stride()[i];
    attr.padding_count = static_cast<uint32_t>(pool->padding().size());
    for (uint32_t i = 0; i < pool->padding().size() && i < CONFINFER_VALUE_MAX_DIMS * 2; ++i)
        attr.padding[i] = pool->padding()[i];
    for (uint32_t i = 0; i < pool->dilation().size() && i < CONFINFER_VALUE_MAX_DIMS; ++i)
        attr.dilation[i] = pool->dilation()[i];
    attr.return_indices = pool->returnIndices();
    attr.ceil_mode = pool->ceilMode();
    attr.count_include_pad = pool->countIncludePad();
    attr.divisor_override = pool->divisorOverride();
    append_attr_blob(desc, attr, blob);
}

void append_adaptive_pool_attr(ModelImageLayerDesc& desc,
                               const Kernel::core::AdaptivePool2d_L *pool,
                               std::vector<uint8_t>& blob)
{
    ModelImageAdaptivePoolAttr attr{};
    attr.output_ndim = static_cast<uint32_t>(pool->outputSize().size());
    for (uint32_t i = 0; i < pool->outputSize().size() && i < CONFINFER_VALUE_MAX_DIMS; ++i)
        attr.output_size[i] = pool->outputSize()[i];
    attr.return_indices = pool->returnIndices();
    attr.reserved0 = 0;
    append_attr_blob(desc, attr, blob);
}

void append_norm_attr(ModelImageLayerDesc& desc,
                      const Kernel::core::Layer *layer,
                      std::vector<uint8_t>& blob)
{
    ModelImageNormAttr attr{};
    if (auto *ln = dynamic_cast<const Kernel::core::LayerNorm_L *>(layer)) {
        attr.eps = ln->eps();
        attr.affine = ln->elementwiseAffine();
        attr.normalized_ndim = static_cast<uint32_t>(ln->normalizedShape().size());
        for (uint32_t i = 0; i < ln->normalizedShape().size() && i < CONFINFER_VALUE_MAX_DIMS; ++i)
            attr.normalized_shape[i] = ln->normalizedShape()[i];
    } else if (auto *gn = dynamic_cast<const Kernel::core::GroupNorm_L *>(layer)) {
        attr.eps = gn->eps();
        attr.affine = gn->affine();
        attr.num_groups = gn->numGroups();
        attr.num_channels = gn->numChannels();
    }
    append_attr_blob(desc, attr, blob);
}

void append_batchnorm_attr(ModelImageLayerDesc& desc,
                           const Kernel::core::BatchNorm2d_L *bn,
                           std::vector<uint8_t>& blob)
{
    ModelImageBatchNormAttr attr{};
    attr.eps = bn->eps();
    attr.num_features = bn->numFeatures();
    attr.affine = bn->affine();
    attr.track_running_stats = bn->trackRunningStats();
    attr.momentum = 0.0f;
    append_attr_blob(desc, attr, blob);
}

void append_dropout_attr(ModelImageLayerDesc& desc,
                         const Kernel::core::Dropout *dropout,
                         std::vector<uint8_t>& blob)
{
    ModelImageDropoutAttr attr{};
    attr.p = dropout->p();
    attr.reserved0 = 0;
    append_attr_blob(desc, attr, blob);
}

void append_linear_attr(ModelImageLayerDesc& desc,
                        const Kernel::core::Linear_L *linear,
                        std::vector<uint8_t>& blob)
{
    ModelImageLinearAttr attr{};
    attr.in_features = linear->inFeatures();
    attr.out_features = linear->outFeatures();
    attr.has_bias = linear->biasEnabled();
    attr.reserved0 = 0;
    append_attr_blob(desc, attr, blob);
}

void append_add_attr(ModelImageLayerDesc& desc,
                     const Kernel::core::Add_L *add,
                     std::vector<uint8_t>& blob)
{
    ModelImageAddAttr attr{};
    attr.alpha = add->alpha();
    append_attr_blob(desc, attr, blob);
}

void append_bias_add_attr(ModelImageLayerDesc& desc,
                          const Kernel::core::BiasAdd_L *biasadd,
                          std::vector<uint8_t>& blob)
{
    ModelImageBiasAddAttr attr{};
    attr.size = biasadd->size();
    attr.dim = biasadd->dim();
    append_attr_blob(desc, attr, blob);
}

void append_layer_attr(const Kernel::core::Layer *layer,
                       ModelImageLayerDesc& desc,
                       std::vector<uint8_t>& blob)
{
    switch (layer->type()) {
    case Kernel::core::LayerType::CONV2D:
        append_conv_attr(desc, dynamic_cast<const Kernel::core::ConvNd_L *>(layer), blob);
        break;
    case Kernel::core::LayerType::MAXPOOL2D:
    case Kernel::core::LayerType::AVGPOOL2D:
        append_pool_attr(desc, dynamic_cast<const Kernel::core::PoolNd_L *>(layer), blob);
        break;
    case Kernel::core::LayerType::ADAPTIVEAVGPOOL2D:
    case Kernel::core::LayerType::ADAPTIVEMAXPOOL2D:
        append_adaptive_pool_attr(desc, dynamic_cast<const Kernel::core::AdaptivePool2d_L *>(layer), blob);
        break;
    case Kernel::core::LayerType::BATCHNORM2D:
        append_batchnorm_attr(desc, dynamic_cast<const Kernel::core::BatchNorm2d_L *>(layer), blob);
        break;
    case Kernel::core::LayerType::LAYERNORM:
    case Kernel::core::LayerType::GROUPNORM:
        append_norm_attr(desc, layer, blob);
        break;
    case Kernel::core::LayerType::DROPOUT:
        append_dropout_attr(desc, dynamic_cast<const Kernel::core::Dropout *>(layer->opSignature()), blob);
        break;
    case Kernel::core::LayerType::FLATTEN:
        append_flatten_attr(desc, dynamic_cast<const Kernel::core::Flatten_L *>(layer), blob);
        break;
    case Kernel::core::LayerType::BIASADD:
        append_bias_add_attr(desc, dynamic_cast<const Kernel::core::BiasAdd_L *>(layer), blob);
        break;
    case Kernel::core::LayerType::LINEAR:
        append_linear_attr(desc, dynamic_cast<const Kernel::core::Linear_L *>(layer), blob);
        break;
    case Kernel::core::LayerType::SOFTMAX:
        append_axis_attr(desc, dynamic_cast<const Kernel::core::Softmax_L *>(layer)->dim(), blob);
        break;
    case Kernel::core::LayerType::CONCAT:
        append_axis_attr(desc, dynamic_cast<const Kernel::core::Concat_L *>(layer)->dim(), blob);
        break;
    case Kernel::core::LayerType::ADD:
        append_add_attr(desc, dynamic_cast<const Kernel::core::Add_L *>(layer), blob);
        break;
    default:
        desc.attr_off = 0;
        desc.attr_size = 0;
        break;
    }
}

void append_dims(uint32_t *dst, uint32_t ndim, const Kernel::core::DataShape_t& shape)
{
    for (uint32_t i = 0; i < ndim; ++i) {
        dst[i] = shape.dims[i];
    }
}

ModelImageValueRole classify_value(const Kernel::core::ExecPartition& part,
                                   const Kernel::core::Value_t *value)
{
    if (std::find(part.inputs().begin(), part.inputs().end(), value) != part.inputs().end()) {
        return ModelImageValueRole::PARTITION_INPUT;
    }
    if (std::find(part.outputs().begin(), part.outputs().end(), value) != part.outputs().end()) {
        return ModelImageValueRole::PARTITION_OUTPUT;
    }
    return ModelImageValueRole::INTERNAL;
}

std::vector<Kernel::core::Value_t *> collect_partition_values(const Kernel::core::ExecPartition& part)
{
    std::vector<Kernel::core::Value_t *> values;

    values.insert(values.end(), part.inputs().begin(), part.inputs().end());
    values.insert(values.end(), part.outputs().begin(), part.outputs().end());
    values.insert(values.end(), part.internals().begin(), part.internals().end());
    return values;
}

PartitionImageData build_partition_image(const Kernel::core::ExecPartition& part)
{
    PartitionImageData out{};
    std::unordered_map<confinfer_value_id_t, size_t> value_index;
    const Kernel::core::ParamRole param_roles[] = {
        Kernel::core::ParamRole::WEIGHT,
        Kernel::core::ParamRole::BIAS,
        Kernel::core::ParamRole::RUNNING_MEAN,
        Kernel::core::ParamRole::RUNNING_VAR,
    };
    const std::vector<Kernel::core::Value_t *> values = collect_partition_values(part);

    out.header.magic = CONFINFER_PARTITION_IMAGE_MAGIC;
    out.header.version_major = CONFINFER_MODEL_IMAGE_VERSION_MAJOR;
    out.header.version_minor = CONFINFER_MODEL_IMAGE_VERSION_MINOR;
    out.header.partition_id = static_cast<confinfer_partition_id_t>(part.id());
    out.header.flags = 0;
    out.header.layer_count = static_cast<uint32_t>(part.topo().size());
    out.header.value_count = static_cast<uint32_t>(values.size());
    out.header.input_count = static_cast<uint32_t>(part.inputs().size());
    out.header.output_count = static_cast<uint32_t>(part.outputs().size());
    out.header.internal_count = static_cast<uint32_t>(part.internals().size());
    out.header.input_ref_count = 0;
    out.header.output_ref_count = 0;
    out.header.param_ref_count = 0;

    out.values.reserve(values.size());
    uint32_t runtime_data_offset = 0;
    for (size_t i = 0; i < values.size(); ++i) {
        Kernel::core::Value_t *value = values[i];
        ModelImageValueDesc desc{};

        EXIT_ERROR_CHECK_EQ(nullptr, value, "Partition value is nullptr");
        desc.value_id = static_cast<confinfer_value_id_t>(value->id);
        desc.role = value_role_to_proto(classify_value(part, value));
        desc.kind = static_cast<uint16_t>(value->kind);
        desc.flags = value->data.flags;
        desc.producer_layer_id = value->producer ? static_cast<confinfer_layer_id_t>(value->producer->id())
                                                 : CONFINFER_INVALID_LAYER_ID;
        desc.output_index = value->output_index;
        desc.dtype = data_type_to_proto(value->data.dtype);
        desc.location = data_location_to_proto(value->data.location);
        desc.elem_count = value->data.shape.size;
        desc.byte_size = value->data.shape.size * value->data.getTypeSize();
        desc.data_offset = runtime_data_offset;
        desc.ndim = value->data.shape.ndim;
        append_dims(desc.dims, desc.ndim, value->data.shape);
        out.values.push_back(desc);
        value_index[desc.value_id] = i;
        runtime_data_offset += desc.byte_size;
    }
    out.header.runtime_data_size = runtime_data_offset;

    out.layers.reserve(part.topo().size());
    out.layer_ios.reserve(part.topo().size());
    for (size_t topo_index = 0; topo_index < part.topo().size(); ++topo_index) {
        Kernel::core::Layer *layer = part.topo()[topo_index];
        ModelImageLayerDesc layer_desc{};
        ModelImageLayerIO layer_io{};

        EXIT_ERROR_CHECK_EQ(nullptr, layer, "Partition layer is nullptr");
        layer_desc.layer_id = static_cast<confinfer_layer_id_t>(layer->id());
        layer_desc.layer_type = static_cast<uint16_t>(layer->type());
        layer_desc.reserved0 = 0;
        layer_desc.flags = layer->lf();
        layer_desc.workspace_bytes = layer->workspaceSize();
        layer_desc.topo_index = static_cast<uint32_t>(topo_index);
        append_layer_attr(layer, layer_desc, out.attr_blob);
        out.layers.push_back(layer_desc);

        layer_io.layer_id = layer_desc.layer_id;
        layer_io.input_ref_begin = static_cast<uint32_t>(out.input_refs.size());
        layer_io.input_ref_count = layer->inputNum();
        out.header.input_ref_count += layer_io.input_ref_count;
        for (UINT i = 0; i < layer->inputNum(); ++i) {
            ModelImageValueRef ref{};
            ref.value_id = static_cast<confinfer_value_id_t>(layer->input(i).id);
            ref.reserved0 = 0;
            out.input_refs.push_back(ref);
        }

        layer_io.output_ref_begin = static_cast<uint32_t>(out.output_refs.size());
        layer_io.output_ref_count = layer->outputNum();
        out.header.output_ref_count += layer_io.output_ref_count;
        for (UINT i = 0; i < layer->outputNum(); ++i) {
            ModelImageValueRef ref{};
            ref.value_id = static_cast<confinfer_value_id_t>(layer->output(i).id);
            ref.reserved0 = 0;
            out.output_refs.push_back(ref);
        }

        layer_io.param_ref_begin = static_cast<uint32_t>(out.param_refs.size());
        layer_io.param_ref_count = 0;
        for (Kernel::core::ParamRole role : param_roles) {
            const Kernel::core::Data_t *param = layer->param(role);
            const UINT param_id = layer->paramId(role);

            if (nullptr == param || INVALID_VALUE_U == param_id) {
                continue;
            }
            ModelImageParamRef ref{};
            ref.param_id = static_cast<confinfer_param_id_t>(param_id);
            ref.role = param_role_to_proto(role);
            out.param_refs.push_back(ref);
            layer_io.param_ref_count += 1;
            out.header.param_ref_count += 1;
        }

        out.layer_ios.push_back(layer_io);
    }

    return out;
}

std::vector<uint8_t> serialize_partition_image(PartitionImageData& part)
{
    const uint32_t header_size = static_cast<uint32_t>(sizeof(ModelPartitionImageHeader));
    const uint32_t layer_desc_size = static_cast<uint32_t>(part.layers.size() * sizeof(ModelImageLayerDesc));
    const uint32_t value_desc_size = static_cast<uint32_t>(part.values.size() * sizeof(ModelImageValueDesc));
    const uint32_t layer_io_size = static_cast<uint32_t>(part.layer_ios.size() * sizeof(ModelImageLayerIO));
    const uint32_t input_ref_size = static_cast<uint32_t>(part.input_refs.size() * sizeof(ModelImageValueRef));
    const uint32_t output_ref_size = static_cast<uint32_t>(part.output_refs.size() * sizeof(ModelImageValueRef));
    const uint32_t param_ref_size = static_cast<uint32_t>(part.param_refs.size() * sizeof(ModelImageParamRef));
    const uint32_t total_size = header_size + layer_desc_size + value_desc_size + layer_io_size +
                                input_ref_size + output_ref_size + param_ref_size +
                                static_cast<uint32_t>(part.attr_blob.size()) +
                                part.header.runtime_data_size;
    std::vector<uint8_t> bytes(total_size, 0);
    uint8_t *base = bytes.data();

    part.header.total_size = total_size;
    part.header.layer_desc_off = header_size;
    part.header.value_desc_off = part.header.layer_desc_off + layer_desc_size;
    part.header.layer_io_off = part.header.value_desc_off + value_desc_size;
    part.header.input_ref_off = part.header.layer_io_off + layer_io_size;
    part.header.output_ref_off = part.header.input_ref_off + input_ref_size;
    part.header.param_ref_off = part.header.output_ref_off + output_ref_size;
    part.header.attr_blob_off = part.header.param_ref_off + param_ref_size;
    part.header.attr_blob_size = static_cast<uint32_t>(part.attr_blob.size());
    part.header.runtime_data_off = part.header.attr_blob_off + part.header.attr_blob_size;

    std::memcpy(base, &part.header, sizeof(part.header));
    if (!part.layers.empty()) {
        std::memcpy(base + part.header.layer_desc_off,
                    part.layers.data(),
                    part.layers.size() * sizeof(ModelImageLayerDesc));
    }
    if (!part.values.empty()) {
        std::memcpy(base + part.header.value_desc_off,
                    part.values.data(),
                    part.values.size() * sizeof(ModelImageValueDesc));
    }
    if (!part.layer_ios.empty()) {
        std::memcpy(base + part.header.layer_io_off,
                    part.layer_ios.data(),
                    part.layer_ios.size() * sizeof(ModelImageLayerIO));
    }
    if (!part.input_refs.empty()) {
        std::memcpy(base + part.header.input_ref_off,
                    part.input_refs.data(),
                    part.input_refs.size() * sizeof(ModelImageValueRef));
    }
    if (!part.output_refs.empty()) {
        std::memcpy(base + part.header.output_ref_off,
                    part.output_refs.data(),
                    part.output_refs.size() * sizeof(ModelImageValueRef));
    }
    if (!part.param_refs.empty()) {
        std::memcpy(base + part.header.param_ref_off,
                    part.param_refs.data(),
                    part.param_refs.size() * sizeof(ModelImageParamRef));
    }
    if (!part.attr_blob.empty()) {
        std::memcpy(base + part.header.attr_blob_off,
                    part.attr_blob.data(),
                    part.attr_blob.size());
    }
    return bytes;
}

} // namespace

ModelImage ModelImageBuilder::build(confinfer_model_id_t model_id,
                                    const std::vector<Kernel::core::ExecPartition>& parts,
                                    ModelImageExecMode exec_mode,
                                    uint64_t reserved_phys_base,
                                    uint64_t reserved_phys_size) const
{
    const uint32_t partition_count = count_tee_partitions(parts);
    const CollectedParams params = collect_tee_params(parts);
    std::vector<PartitionImageData> partition_images;
    std::vector<std::vector<uint8_t>> serialized_partitions;
    const uint32_t partition_table_size =
        partition_count * static_cast<uint32_t>(sizeof(ModelImagePartitionEntry));
    const uint32_t param_desc_size =
        static_cast<uint32_t>(params.descs.size() * sizeof(ModelImageParamDesc));
    const uint32_t header_size = static_cast<uint32_t>(sizeof(ModelImageHeader));
    uint32_t partition_blob_size = 0;
    std::vector<uint8_t> bytes;
    ModelImageHeader header{};
    ModelImage image;
    uint32_t partition_data_cursor = 0;
    uint32_t partition_index = 0;

    partition_images.reserve(partition_count);
    serialized_partitions.reserve(partition_count);
    for (const auto& part : parts) {
        if (part.domain() != Kernel::core::ExecutionDomain::ED_CPU_TEE) {
            continue;
        }
        partition_images.push_back(build_partition_image(part));
        serialized_partitions.push_back(serialize_partition_image(partition_images.back()));
        partition_blob_size += static_cast<uint32_t>(serialized_partitions.back().size());
    }

    header.magic = CONFINFER_MODEL_IMAGE_MAGIC;
    header.version_major = CONFINFER_MODEL_IMAGE_VERSION_MAJOR;
    header.version_minor = CONFINFER_MODEL_IMAGE_VERSION_MINOR;
    header.header_size = header_size;
    header.model_id = model_id;
    header.exec_mode = static_cast<uint32_t>(exec_mode);
    header.flags = 0;
    header.partition_count = partition_count;
    header.partition_table_off = header_size;
    header.partition_table_size = partition_table_size;
    header.param_desc_count = static_cast<uint32_t>(params.descs.size());
    header.param_desc_off = header.partition_table_off + partition_table_size;
    header.param_data_off = header.param_desc_off + param_desc_size;
    header.param_data_size = static_cast<uint32_t>(params.blob.size());
    header.reserved_phys_base = reserved_phys_base;
    header.reserved_phys_size = reserved_phys_size;
    header.total_size = header.param_data_off + header.param_data_size + partition_blob_size;

    bytes.resize(header.total_size, 0);
    std::memcpy(bytes.data(), &header, sizeof(header));

    if (partition_count > 0) {
        auto *entries = reinterpret_cast<ModelImagePartitionEntry *>(bytes.data() + header.partition_table_off);
        partition_data_cursor = header.param_data_off + header.param_data_size;
        for (size_t i = 0; i < partition_images.size(); ++i) {
            const PartitionImageData& partition_image = partition_images[i];
            const std::vector<uint8_t>& partition_blob = serialized_partitions[i];
            ModelImagePartitionEntry entry{};
            entry.partition_id = partition_image.header.partition_id;
            entry.flags = partition_image.header.flags;
            entry.image_off = partition_data_cursor;
            entry.image_size = static_cast<uint32_t>(partition_blob.size());
            entry.layer_count = partition_image.header.layer_count;
            entry.input_count = partition_image.header.input_count;
            entry.output_count = partition_image.header.output_count;
            entry.internal_count = partition_image.header.internal_count;
            entries[partition_index++] = entry;
            std::memcpy(bytes.data() + partition_data_cursor,
                        partition_blob.data(),
                        partition_blob.size());
            partition_data_cursor += static_cast<uint32_t>(partition_blob.size());
        }
    }

    if (!params.descs.empty()) {
        std::memcpy(bytes.data() + header.param_desc_off,
                    params.descs.data(),
                    params.descs.size() * sizeof(ModelImageParamDesc));
    }
    if (!params.blob.empty()) {
        std::memcpy(bytes.data() + header.param_data_off,
                    params.blob.data(),
                    params.blob.size());
    }

    image.reset(std::move(bytes));
    return image;
}

} // namespace image
} // namespace Kernel
