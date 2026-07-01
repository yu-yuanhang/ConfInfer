#include <core/ExecUnitProto.h>
#include <core/ops/activation.h>
#include <core/ops/arithmetic.h>
#include <core/ops/convolution.h>
#include <core/ops/linear.h>
#include <core/ops/normalization.h>
#include <core/ops/pool.h>
#include <core/ops/reshape.h>

namespace Kernel {
namespace core {

namespace {

uint32_t param_role_to_proto(ParamRole role) {
    switch (role) {
    case ParamRole::WEIGHT:
        return CONFINFER_PARAM_ROLE_WEIGHT;
    case ParamRole::BIAS:
        return CONFINFER_PARAM_ROLE_BIAS;
    case ParamRole::RUNNING_MEAN:
        return CONFINFER_PARAM_ROLE_RUNNING_MEAN;
    case ParamRole::RUNNING_VAR:
        return CONFINFER_PARAM_ROLE_RUNNING_VAR;
    case ParamRole::UNKNOWN:
    default:
        return CONFINFER_PARAM_ROLE_UNKNOWN;
    }
}

template<typename T>
void append_attr_blob(confinfer_layer_desc_t& desc,
                      const T& attr,
                      std::vector<uint8_t>& blob) {
    desc.attr_offset = static_cast<uint32_t>(blob.size());
    desc.attr_size = static_cast<uint32_t>(sizeof(T));
    const uint8_t *src = reinterpret_cast<const uint8_t *>(&attr);
    blob.insert(blob.end(), src, src + sizeof(T));
}

void append_axis_attr(confinfer_layer_desc_t& desc, INT dim, std::vector<uint8_t>& blob) {
    confinfer_axis_attr_t attr{};
    attr.dim = dim;
    attr.reserved = 0;
    append_attr_blob(desc, attr, blob);
}

void append_flatten_attr(confinfer_layer_desc_t& desc,
                         const Flatten_L *flatten,
                         std::vector<uint8_t>& blob) {
    confinfer_flatten_attr_t attr{};
    attr.start_dim = flatten->startDim();
    attr.end_dim = flatten->endDim();
    append_attr_blob(desc, attr, blob);
}

void append_conv_attr(confinfer_layer_desc_t& desc,
                      const ConvNd_L *conv,
                      std::vector<uint8_t>& blob) {
    confinfer_conv_attr_t attr{};
    attr.in_channels = conv->inChannels();
    attr.out_channels = conv->outChannels();
    attr.groups = conv->groups();
    attr.has_bias = conv->biasEnabled();
    attr.spatial_dim = conv->spatialDim();
    for (uint32_t i = 0; i < conv->kernelSize().size() && i < CONFINFER_VALUE_MAX_DIMS; ++i) {
        attr.kernel_size[i] = conv->kernelSize()[i];
    }
    for (uint32_t i = 0; i < conv->stride().size() && i < CONFINFER_VALUE_MAX_DIMS; ++i) {
        attr.stride[i] = conv->stride()[i];
    }
    attr.padding_count = static_cast<uint32_t>(conv->padding().size());
    for (uint32_t i = 0; i < conv->padding().size() &&
                         i < CONFINFER_VALUE_MAX_DIMS * 2; ++i) {
        attr.padding[i] = conv->padding()[i];
    }
    for (uint32_t i = 0; i < conv->dilation().size() && i < CONFINFER_VALUE_MAX_DIMS; ++i) {
        attr.dilation[i] = conv->dilation()[i];
    }
    append_attr_blob(desc, attr, blob);
}

void append_pool_attr(confinfer_layer_desc_t& desc,
                      const PoolNd_L *pool,
                      std::vector<uint8_t>& blob) {
    confinfer_pool_attr_t attr{};
    attr.spatial_dim = pool->spatialDim();
    for (uint32_t i = 0; i < pool->kernelSize().size() && i < CONFINFER_VALUE_MAX_DIMS; ++i) {
        attr.kernel_size[i] = pool->kernelSize()[i];
    }
    for (uint32_t i = 0; i < pool->stride().size() && i < CONFINFER_VALUE_MAX_DIMS; ++i) {
        attr.stride[i] = pool->stride()[i];
    }
    attr.padding_count = static_cast<uint32_t>(pool->padding().size());
    for (uint32_t i = 0; i < pool->padding().size() &&
                         i < CONFINFER_VALUE_MAX_DIMS * 2; ++i) {
        attr.padding[i] = pool->padding()[i];
    }
    for (uint32_t i = 0; i < pool->dilation().size() && i < CONFINFER_VALUE_MAX_DIMS; ++i) {
        attr.dilation[i] = pool->dilation()[i];
    }
    attr.return_indices = pool->returnIndices();
    attr.ceil_mode = pool->ceilMode();
    attr.count_include_pad = pool->countIncludePad();
    attr.divisor_override = pool->divisorOverride();
    append_attr_blob(desc, attr, blob);
}

void append_adaptive_pool_attr(confinfer_layer_desc_t& desc,
                               const AdaptivePool2d_L *pool,
                               std::vector<uint8_t>& blob) {
    confinfer_adaptive_pool_attr_t attr{};
    attr.output_ndim = static_cast<uint32_t>(pool->outputSize().size());
    for (uint32_t i = 0; i < pool->outputSize().size() && i < CONFINFER_VALUE_MAX_DIMS; ++i) {
        attr.output_size[i] = pool->outputSize()[i];
    }
    attr.return_indices = pool->returnIndices();
    append_attr_blob(desc, attr, blob);
}

void append_norm_attr(confinfer_layer_desc_t& desc,
                      const Layer *layer,
                      std::vector<uint8_t>& blob) {
    confinfer_norm_attr_t attr{};
    if (auto *ln = dynamic_cast<const LayerNorm_L *>(layer)) {
        attr.eps = ln->eps();
        attr.affine = ln->elementwiseAffine();
        attr.normalized_ndim = static_cast<uint32_t>(ln->normalizedShape().size());
        for (uint32_t i = 0; i < ln->normalizedShape().size() && i < CONFINFER_VALUE_MAX_DIMS; ++i) {
            attr.normalized_shape[i] = ln->normalizedShape()[i];
        }
    } else if (auto *gn = dynamic_cast<const GroupNorm_L *>(layer)) {
        attr.eps = gn->eps();
        attr.affine = gn->affine();
        attr.num_groups = gn->numGroups();
        attr.num_channels = gn->numChannels();
    }
    append_attr_blob(desc, attr, blob);
}

void append_batchnorm_attr(confinfer_layer_desc_t& desc,
                           const BatchNorm2d_L *bn,
                           std::vector<uint8_t>& blob) {
    confinfer_batchnorm_attr_t attr{};
    attr.eps = bn->eps();
    attr.num_features = bn->numFeatures();
    attr.affine = bn->affine();
    attr.track_running_stats = bn->trackRunningStats();
    attr.momentum = 0.0f;
    append_attr_blob(desc, attr, blob);
}

void append_dropout_attr(confinfer_layer_desc_t& desc,
                         const Dropout *dropout,
                         std::vector<uint8_t>& blob) {
    confinfer_dropout_attr_t attr{};
    attr.p = dropout->p();
    attr.inplace = 0;
    append_attr_blob(desc, attr, blob);
}

void append_linear_attr(confinfer_layer_desc_t& desc,
                        const Linear_L *linear,
                        std::vector<uint8_t>& blob) {
    confinfer_linear_attr_t attr{};
    attr.in_features = linear->inFeatures();
    attr.out_features = linear->outFeatures();
    attr.has_bias = linear->biasEnabled();
    append_attr_blob(desc, attr, blob);
}

void append_add_attr(confinfer_layer_desc_t& desc,
                     const Add_L *add,
                     std::vector<uint8_t>& blob) {
    confinfer_add_attr_t attr{};
    attr.alpha = add->alpha();
    append_attr_blob(desc, attr, blob);
}

void append_bias_add_attr(confinfer_layer_desc_t& desc,
                          const BiasAdd_L *biasadd,
                          std::vector<uint8_t>& blob) {
    confinfer_bias_add_attr_t attr{};
    attr.size = biasadd->size();
    attr.dim = biasadd->dim();
    append_attr_blob(desc, attr, blob);
}

void append_layer_attr(const Layer *layer,
                       confinfer_layer_desc_t& desc,
                       std::vector<uint8_t>& blob) {
    switch (layer->type()) {
    case LayerType::CONV2D:
        append_conv_attr(desc, dynamic_cast<const ConvNd_L *>(layer), blob);
        break;
    case LayerType::MAXPOOL2D:
    case LayerType::AVGPOOL2D:
        append_pool_attr(desc, dynamic_cast<const PoolNd_L *>(layer), blob);
        break;
    case LayerType::ADAPTIVEAVGPOOL2D:
    case LayerType::ADAPTIVEMAXPOOL2D:
        append_adaptive_pool_attr(desc, dynamic_cast<const AdaptivePool2d_L *>(layer), blob);
        break;
    case LayerType::BATCHNORM2D:
        append_batchnorm_attr(desc, dynamic_cast<const BatchNorm2d_L *>(layer), blob);
        break;
    case LayerType::LAYERNORM:
    case LayerType::GROUPNORM:
        append_norm_attr(desc, layer, blob);
        break;
    case LayerType::DROPOUT:
        append_dropout_attr(desc, dynamic_cast<const Dropout *>(layer->opSignature()), blob);
        break;
    case LayerType::FLATTEN:
        append_flatten_attr(desc, dynamic_cast<const Flatten_L *>(layer), blob);
        break;
    case LayerType::BIASADD:
        append_bias_add_attr(desc, dynamic_cast<const BiasAdd_L *>(layer), blob);
        break;
    case LayerType::LINEAR:
        append_linear_attr(desc, dynamic_cast<const Linear_L *>(layer), blob);
        break;
    case LayerType::SOFTMAX:
        append_axis_attr(desc, dynamic_cast<const Softmax_L *>(layer)->dim(), blob);
        break;
    case LayerType::CONCAT:
        append_axis_attr(desc, dynamic_cast<const Concat_L *>(layer)->dim(), blob);
        break;
    case LayerType::ADD:
        append_add_attr(desc, dynamic_cast<const Add_L *>(layer), blob);
        break;
    default:
        desc.attr_offset = 0;
        desc.attr_size = 0;
        break;
    }
}

uint32_t dtype_size(DataType dtype) {
    switch (dtype) {
    case DataType::INT8:
        return sizeof(int8_t);
    case DataType::INT32:
        return sizeof(int32_t);
    case DataType::FP16:
        return sizeof(fp16_t);
    case DataType::FP32:
    default:
        return sizeof(float32);
    }
}

uint32_t data_type_to_proto(DataType dtype) {
    switch (dtype) {
    case DataType::FP16:
        return CONFINFER_DTYPE_FP16;
    case DataType::INT8:
        return CONFINFER_DTYPE_INT8;
    case DataType::INT32:
        return CONFINFER_DTYPE_INT32;
    case DataType::FP32:
    default:
        return CONFINFER_DTYPE_FP32;
    }
}

uint32_t data_location_to_proto(DataLocation location) {
    switch (location) {
    case DataLocation::TEE:
        return CONFINFER_DATA_TEE;
    case DataLocation::CPU:
    default:
        return CONFINFER_DATA_CPU;
    }
}

void fill_value_desc(const Value_t *value,
                     confinfer_value_desc_t& desc) {
    EXIT_ERROR_CHECK_EQ(nullptr, value, "Value_t is nullptr");
    desc.value_id = value->id;
    // Value 描述里保留全局真实 producer。
    // 当前 partition 内是否能解析出这个 producer，应由 TA runtime 的本地绑定结果决定。
    desc.producer_layer_id = nullptr == value->producer ?
        CONFINFER_INVALID_LAYER_ID :
        static_cast<uint32_t>(value->producer->id());
    desc.output_index = value->output_index;
    desc.kind = static_cast<uint32_t>(value->kind);
    desc.dtype = data_type_to_proto(value->data.dtype);
    desc.location = data_location_to_proto(value->data.location);
    desc.flags = value->data.flags;
    desc.elem_count = value->data.shape.size;
    desc.byte_size = value->data.shape.size * dtype_size(value->data.dtype);
    desc.ndim = value->data.shape.ndim;
    EXIT_ERROR_CHECK_EQ(true, desc.ndim > CONFINFER_VALUE_MAX_DIMS,
        "Value ndim exceeds protocol max dims");
    for (uint32_t i = 0; i < desc.ndim; ++i) {
        desc.dims[i] = value->data.shape.dims[i];
    }
}

void append_value_group(const std::vector<Value_t *>& values,
                        std::vector<ExecValueProto>& out,
                        std::unordered_set<confinfer_value_id_t>& seen) {
    out.reserve(values.size());
    for (auto it = values.begin(); it != values.end(); ++it) {
        Value_t *value = *it;
        EXIT_ERROR_CHECK_EQ(nullptr, value, "Value_t is nullptr");
        EXIT_ERROR_CHECK_EQ(INVALID_VALUE_U, value->id, "Value_t id is invalid");
        if (seen.find(static_cast<confinfer_value_id_t>(value->id)) != seen.end()) {
            continue;
        }

        ExecValueProto proto;
        fill_value_desc(value, proto.desc);
        out.push_back(proto);
        seen.insert(static_cast<confinfer_value_id_t>(value->id));
    }
}

void append_layer_refs(const Layer *layer,
                       confinfer_layer_io_desc_t& io_desc,
                       std::vector<confinfer_layer_value_ref_t>& input_refs,
                       std::vector<confinfer_layer_value_ref_t>& output_refs,
                       std::vector<confinfer_layer_param_ref_t>& param_refs) {
    Layer *mutable_layer = const_cast<Layer *>(layer);
    EXIT_ERROR_CHECK_EQ(nullptr, layer, "Layer is nullptr");

    io_desc.layer_id = layer->id();
    io_desc.input_ref_start = static_cast<uint32_t>(input_refs.size());
    io_desc.input_ref_count = 0;
    io_desc.output_ref_start = static_cast<uint32_t>(output_refs.size());
    io_desc.output_ref_count = 0;
    io_desc.param_ref_start = static_cast<uint32_t>(param_refs.size());
    io_desc.param_ref_count = 0;

    for (uint32_t i = 0; i < layer->inputNum(); ++i) {
        const Value_t *value = &mutable_layer->input(i);
        EXIT_ERROR_CHECK_EQ(INVALID_VALUE_U, value->id, "Layer input value id is invalid");

        confinfer_layer_value_ref_t ref{};
        ref.value_id = static_cast<confinfer_value_id_t>(value->id);
        input_refs.push_back(ref);
        ++io_desc.input_ref_count;
    }

    for (uint32_t i = 0; i < layer->outputNum(); ++i) {
        const Value_t *value = &mutable_layer->output(i);
        EXIT_ERROR_CHECK_EQ(INVALID_VALUE_U, value->id, "Layer output value id is invalid");

        confinfer_layer_value_ref_t ref{};
        ref.value_id = static_cast<confinfer_value_id_t>(value->id);
        output_refs.push_back(ref);
        ++io_desc.output_ref_count;
    }

    const ParamRole roles[] = {
        ParamRole::WEIGHT,
        ParamRole::BIAS,
        ParamRole::RUNNING_MEAN,
        ParamRole::RUNNING_VAR,
    };
    for (ParamRole role : roles) {
        const Data_t *param = layer->param(role);
        const UINT param_id = layer->paramId(role);
        if (nullptr == param) {
            continue;
        }
        EXIT_ERROR_CHECK_EQ(INVALID_VALUE_U, param_id, "Layer param_id is invalid");
        confinfer_layer_param_ref_t ref{};
        ref.param_id = static_cast<confinfer_param_id_t>(param_id);
        ref.role = param_role_to_proto(role);
        param_refs.push_back(ref);
        ++io_desc.param_ref_count;
        (void)param;
    }
}

} // namespace

uint32_t exec_domain_to_proto(ExecutionDomain domain) {
    switch (domain) {
    case ExecutionDomain::ED_CPU_REE:
        return CONFINFER_DOMAIN_CPU_REE;
    case ExecutionDomain::ED_CPU_TEE:
        return CONFINFER_DOMAIN_CPU_TEE;
    case ExecutionDomain::ED_DEFAULT:
    default:
        return CONFINFER_DOMAIN_DEFAULT;
    }
}

uint32_t exec_unit_type_to_proto(ExecUnitType type) {
    switch (type) {
    case ExecUnitType::EU_PARTITION:
        return CONFINFER_UNIT_PARTITION;
    case ExecUnitType::EU_LAYER:
    default:
        return CONFINFER_UNIT_LAYER;
    }
}

ExecUnitProto make_exec_unit_proto(const ExecUnit& unit) {
    ExecUnitProto proto;
    const ExecutionPartition *part = unit.part();
    std::unordered_set<confinfer_value_id_t> seen_values;

    proto.req.version = CONFINFER_PROTOCOL_VERSION;
    proto.req.domain = exec_domain_to_proto(unit.domain());
    proto.req.unit_type = exec_unit_type_to_proto(unit.type());
    // model_id 先留空
    proto.req.model_id = CONFINFER_INVALID_MODEL_ID;
    proto.req.partition_id = nullptr == part ?
        CONFINFER_INVALID_PARTITION_ID :
        static_cast<confinfer_partition_id_t>(part->id());
    proto.req.layer_count = static_cast<uint32_t>(unit.slices().size());
    proto.req.input_count = static_cast<uint32_t>(unit.inputs().size());
    proto.req.output_count = static_cast<uint32_t>(unit.outputs().size());
    proto.req.flags = 0;
    proto.req.reserved = 0;

    proto.data.req.version = CONFINFER_PROTOCOL_VERSION;
    proto.data.req.input_count = proto.req.input_count;
    proto.data.req.output_count = proto.req.output_count;
    proto.data.req.internal_count = nullptr == part ? 0 : static_cast<uint32_t>(part->internals().size());
    proto.data.req.layer_io_count = 0;
    proto.data.req.input_ref_count = 0;
    proto.data.req.output_ref_count = 0;
    proto.data.req.param_ref_count = 0;
    proto.data.req.total_input_bytes = 0;
    proto.data.req.total_output_bytes = 0;
    proto.data.req.flags = 0;
    proto.data.req.reserved0 = 0;
    proto.data.req.reserved1 = 0;

    proto.layers.reserve(unit.slices().size());
    // 为每个 layer 生成 confinfer_layer_desc_t
    for (auto it = unit.slices().begin(); it != unit.slices().end(); ++it) {
        LayerSlice *slice = *it;
        EXIT_ERROR_CHECK_EQ(nullptr, slice, "ExecUnit contains nullptr LayerSlice");

        Layer *layer = slice->layer();
        EXIT_ERROR_CHECK_EQ(nullptr, layer, "LayerSlice contains nullptr Layer");

        confinfer_layer_desc_t desc{};
        desc.layer_id = layer->id();
        desc.layer_type = static_cast<uint32_t>(layer->type());
        desc.layer_flags = layer->lf();
        desc.attr_offset = 0;
        desc.attr_size = 0;
        desc.reserved = 0;
        // 然后通过 append_layer_attr() 把 layer 的算子属性 blob 也拼进去
        append_layer_attr(layer, desc, proto.layer_attrs);
        proto.layers.push_back(desc);
    }

    append_value_group(unit.inputs(), proto.data.inputs, seen_values);
    for (auto it = proto.data.inputs.begin(); it != proto.data.inputs.end(); ++it) {
        proto.data.req.total_input_bytes += it->desc.byte_size;
    }

    if (nullptr != part) {
        append_value_group(part->internals(), proto.data.internals, seen_values);
    }

    append_value_group(unit.outputs(), proto.data.outputs, seen_values);
    for (auto it = proto.data.outputs.begin(); it != proto.data.outputs.end(); ++it) {
        proto.data.req.total_output_bytes += it->desc.byte_size;
    }

    if (nullptr != part) {
        proto.data.layer_ios.reserve(part->topo().size());
        for (auto it = part->topo().begin(); it != part->topo().end(); ++it) {
            confinfer_layer_io_desc_t io_desc{};
            append_layer_refs(*it, io_desc,
                              proto.data.input_refs,
                              proto.data.output_refs,
                              proto.data.param_refs);
            proto.data.layer_ios.push_back(io_desc);
        }
    }

    proto.data.req.layer_io_count = static_cast<uint32_t>(proto.data.layer_ios.size());
    proto.data.req.input_ref_count = static_cast<uint32_t>(proto.data.input_refs.size());
    proto.data.req.output_ref_count = static_cast<uint32_t>(proto.data.output_refs.size());
    proto.data.req.param_ref_count = static_cast<uint32_t>(proto.data.param_refs.size());

    return proto;
}

} // namespace core
} // namespace Kernel
