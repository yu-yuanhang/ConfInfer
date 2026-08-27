#include <backend/backend.h>
#include <core/Network.h>
#include <cstring>
#include <vector>
#if ENABLE_TEE_BRIDGE
#include <bridges/ExecBridge_TEE.h>
#endif

namespace Kernel {
namespace backend {

namespace {

uint32_t total_value_bytes(const std::vector<Value_t *>& values)
{
    uint32_t total = 0;

    for (Value_t *value : values) {
        EXIT_ERROR_CHECK_EQ(nullptr, value, "Value_t is nullptr");
        EXIT_ERROR_CHECK_EQ(nullptr, value->data.ptr, "Value_t data.ptr is nullptr");
        total += value->data.shape.size * value->data.getTypeSize();
    }
    return total;
}

std::vector<uint8_t> pack_values(const std::vector<Value_t *>& values)
{
    std::vector<uint8_t> blob(total_value_bytes(values), 0);
    uint8_t *cursor = blob.data();

    for (Value_t *value : values) {
        const size_t byte_size = value->data.shape.size * value->data.getTypeSize();

        EXIT_ERROR_CHECK_EQ(nullptr, value, "Value_t is nullptr");
        EXIT_ERROR_CHECK_EQ(nullptr, value->data.ptr, "Value_t data.ptr is nullptr");
        if (byte_size > 0) {
            std::memcpy(cursor, value->data.ptr, byte_size);
            cursor += byte_size;
        }
    }
    return blob;
}

void unpack_values(const std::vector<Value_t *>& values, const std::vector<uint8_t>& blob)
{
    const uint8_t *cursor = blob.data();
    size_t remain = blob.size();

    for (Value_t *value : values) {
        const size_t byte_size = value->data.shape.size * value->data.getTypeSize();

        EXIT_ERROR_CHECK_EQ(nullptr, value, "Value_t is nullptr");
        EXIT_ERROR_CHECK_EQ(nullptr, value->data.ptr, "Value_t data.ptr is nullptr");
        EXIT_ERROR_CHECK_EQ(true, byte_size > remain, "TEE output blob size is too small");
        if (byte_size > 0) {
            std::memcpy(value->data.ptr, cursor, byte_size);
            cursor += byte_size;
            remain -= byte_size;
        }
    }

    EXIT_ERROR_CHECK_NE(0u, static_cast<UINT>(remain), "TEE output blob has trailing bytes");
}

Kernel::bridges::ExecIOBlob make_exec_blob(const ExecPartition& part,
                                           const std::vector<uint8_t>& input_blob,
                                           std::vector<uint8_t>& output_blob)
{
    Kernel::bridges::ExecIOBlob io_blob;

    io_blob.input_data = input_blob.empty() ? nullptr : input_blob.data();
    io_blob.input_bytes = static_cast<uint32_t>(input_blob.size());
    io_blob.input_count = static_cast<uint32_t>(part.inputs().size());
    io_blob.output_data = output_blob.empty() ? nullptr : output_blob.data();
    io_blob.output_bytes = static_cast<uint32_t>(output_blob.size());
    io_blob.output_count = static_cast<uint32_t>(part.outputs().size());
    return io_blob;
}

UINT count_tee_partitions(const std::vector<ExecPartition>& parts)
{
    UINT count = 0;

    for (const auto& part : parts) {
        if (part.domain() == ExecutionDomain::ED_CPU_TEE) {
            ++count;
        }
    }

    return count;
}

void validate_prepare_inputs(const ExecPartition& part, const ExecContext_t *ctx)
{
    EXIT_ERROR_CHECK_EQ(true, part.domain() != ExecutionDomain::ED_CPU_TEE,
                        "Backend_CPU_TEE can only prepare TEE partitions");
    EXIT_ERROR_CHECK_EQ(nullptr, ctx, "ExecContext is nullptr");
    EXIT_ERROR_CHECK_EQ(nullptr, ctx->parts, "ExecContext parts is nullptr");
    EXIT_ERROR_CHECK_EQ(CONFINFER_INVALID_MODEL_ID, ctx->modelId,
                        "ExecContext modelId is invalid");
}

} // namespace

Backend_CPU_TEE::Backend_CPU_TEE()
#if ENABLE_TEE_BRIDGE
    : _bridge(Kernel::bridges::createCompiledTEEBridge()),
#else
    : _bridge(nullptr),
#endif
      _imageBuilder(),
      _runtimeImage(),
      _runtimeLoaded(false),
      _loadedModelId(CONFINFER_INVALID_MODEL_ID) {}

Backend_CPU_TEE::~Backend_CPU_TEE()
{
    resetRuntime(nullptr, false);
#if ENABLE_TEE_BRIDGE
    if (_bridge) {
        _bridge->close();
    }
#endif
}

bool Backend_CPU_TEE::openBridge(uint32_t *err_origin)
{
#if ENABLE_TEE_BRIDGE
    EXIT_ERROR_CHECK_EQ(nullptr, _bridge, "TEE bridge is nullptr");
    return _bridge->open(err_origin);
#else
    if (nullptr != err_origin) {
        *err_origin = 0;
    }
    return false;
#endif
}

bool Backend_CPU_TEE::isBridgeOpened() const
{
#if ENABLE_TEE_BRIDGE
    return _bridge && _bridge->ready();
#else
    return false;
#endif
}

bool Backend_CPU_TEE::hasRuntime(confinfer_model_id_t model_id) const
{
    return _runtimeLoaded && _loadedModelId == model_id;
}

void Backend_CPU_TEE::prepare(Layer* layer)
{
    EXIT_ERROR_CHECK_EQ(nullptr, layer, "Layer is nullptr");
    EXIT_ERROR("Backend_CPU_TEE only supports partition-granularity prepare");
}

void Backend_CPU_TEE::execute(Layer* layer, ExecContext_t* ctx)
{
    EXIT_ERROR_CHECK_EQ(nullptr, layer, "Layer is nullptr");
    (void)ctx;
    EXIT_ERROR("Backend_CPU_TEE only supports partition-granularity execute");
}

void Backend_CPU_TEE::prepare(const ExecPartition& part, ExecContext_t* ctx)
{
    uint32_t err_origin = 0;

    validate_prepare_inputs(part, ctx);
    EXIT_ERROR_CHECK_EQ(false, openBridge(&err_origin),
                        "Backend_CPU_TEE failed to open TEE bridge");
    if (hasRuntime(ctx->modelId)) {
        return;
    }

    if (_runtimeLoaded && _loadedModelId != ctx->modelId) {
        resetRuntime(ctx, false);
    }

    EXIT_ERROR_CHECK_EQ(0U, count_tee_partitions(*ctx->parts),
                        "ExecContext parts contains no TEE partitions");
    EXIT_ERROR_CHECK_EQ(false, loadRuntime(ctx->modelId, *ctx->parts),
                        "Backend_CPU_TEE loadRuntime failed");
}

void Backend_CPU_TEE::execute(const ExecPartition& part, ExecContext_t* ctx)
{
    EXIT_ERROR_CHECK_EQ(true, part.domain() != ExecutionDomain::ED_CPU_TEE,
                        "Backend_CPU_TEE can only execute TEE partitions");
#if ENABLE_TEE_BRIDGE
    std::vector<uint8_t> input_blob;
    std::vector<uint8_t> output_blob;
    Kernel::bridges::ExecIOBlob io_blob;

    if (!isBridgeOpened() || !hasRuntime(ctx ? ctx->modelId : CONFINFER_INVALID_MODEL_ID)) {
        prepare(part, ctx);
    }
    EXIT_ERROR_CHECK_EQ(nullptr, _bridge, "Backend_CPU_TEE bridge is nullptr");
    input_blob = pack_values(part.inputs());
    output_blob.resize(total_value_bytes(part.outputs()), 0);
    io_blob = make_exec_blob(part, input_blob, output_blob);
    EXIT_ERROR_CHECK_EQ(false,
                        _bridge->executePartition(ctx->modelId,
                                                  static_cast<confinfer_partition_id_t>(part.id()),
                                                  io_blob),
                        "Backend_CPU_TEE remote partition execute failed");
    unpack_values(part.outputs(), output_blob);
#else
    (void)ctx;
    EXIT_ERROR("Backend_CPU_TEE execute is unavailable because ENABLE_TEE_BRIDGE=0");
#endif
}

bool Backend_CPU_TEE::loadRuntime(confinfer_model_id_t model_id,
                                  const std::vector<ExecPartition>& parts)
{
#if ENABLE_TEE_BRIDGE
    if (0 == count_tee_partitions(parts)) {
        return true;
    }

    EXIT_ERROR_CHECK_EQ(false, openBridge(nullptr),
                        "Backend_CPU_TEE failed to open TEE bridge");
    EXIT_ERROR_CHECK_EQ(nullptr, _bridge, "Backend_CPU_TEE bridge is nullptr");

    _runtimeImage = _imageBuilder.build(model_id, parts);
    EXIT_ERROR_CHECK_EQ(false,
                        _bridge->loadModelImage(model_id, _runtimeImage),
                        "Backend_CPU_TEE loadModelImage failed");
    _runtimeLoaded = true;
    _loadedModelId = model_id;
    return true;
#else
    (void)model_id;
    (void)parts;
    return false;
#endif
}

bool Backend_CPU_TEE::unloadRuntime(confinfer_model_id_t model_id, bool strict)
{
#if ENABLE_TEE_BRIDGE
    bool ok = false;

    if (nullptr == _bridge) {
        if (strict) {
            EXIT_ERROR("Backend_CPU_TEE bridge is nullptr for unload");
        }
        _runtimeLoaded = false;
        _loadedModelId = CONFINFER_INVALID_MODEL_ID;
        _runtimeImage.reset();
        return false;
    }

    ok = _bridge->unloadModel(model_id, strict);
    if (strict) {
        EXIT_ERROR_CHECK_EQ(false, ok, "Backend_CPU_TEE unloadModel invoke failed");
    }
    if (ok || !strict) {
        _runtimeImage.reset();
        _runtimeLoaded = false;
        _loadedModelId = CONFINFER_INVALID_MODEL_ID;
    }
    return ok;
#else
    (void)model_id;
    (void)strict;
    return false;
#endif
}

void Backend_CPU_TEE::resetRuntime(ExecContext_t* ctx, bool strict)
{
    confinfer_model_id_t model_id = _loadedModelId;

    if (nullptr != ctx && ctx->modelId != CONFINFER_INVALID_MODEL_ID) {
        model_id = ctx->modelId;
    }
    if (!_runtimeLoaded && model_id == CONFINFER_INVALID_MODEL_ID) {
        return;
    }
    if (model_id == CONFINFER_INVALID_MODEL_ID) {
        _runtimeImage.reset();
        _runtimeLoaded = false;
        return;
    }
    unloadRuntime(model_id, strict);
}

} // namespace backend
} // namespace Kernel
