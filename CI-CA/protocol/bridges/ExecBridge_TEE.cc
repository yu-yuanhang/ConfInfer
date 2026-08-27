#include <bridges/ExecBridge_TEE.h>

namespace Kernel {
namespace bridges {

ExecBridge_TEE::ExecBridge_TEE()
    : _client(),
      _opened(false),
      _loaded_model_id(CONFINFER_INVALID_MODEL_ID) {}

ExecBridge_TEE::~ExecBridge_TEE()
{
    close();
}

bool ExecBridge_TEE::open(uint32_t *err_origin)
{
    TEEC_Result res = TEEC_SUCCESS;
    uint32_t origin = 0;

    if (_opened) {
        if (nullptr != err_origin) {
            *err_origin = 0;
        }
        return true;
    }

    res = confinfer_teec_open(&_client, &origin);
    if (nullptr != err_origin) {
        *err_origin = origin;
    }
    if (res != TEEC_SUCCESS) {
        return false;
    }

    _opened = true;
    _loaded_model_id = CONFINFER_INVALID_MODEL_ID;
    return true;
}

bool ExecBridge_TEE::ready() const
{
    return _opened;
}

void ExecBridge_TEE::close()
{
    if (!_opened) {
        return;
    }
    confinfer_teec_close(&_client);
    _opened = false;
    _loaded_model_id = CONFINFER_INVALID_MODEL_ID;
}

bool ExecBridge_TEE_Default::loadModelImage(confinfer_model_id_t model_id,
                                            const Kernel::image::ModelImage& image)
{
    confinfer_prepare_model_image_req_t req{};
    confinfer_prepare_model_image_rsp_t rsp{};
    uint32_t err_origin = 0;
    TEEC_Result res = TEEC_SUCCESS;

    EXIT_ERROR_CHECK_EQ(false, ready(), "ExecBridge_TEE_Default session is not open");
    if (loadedModelId() == model_id) {
        return true;
    }

    req.version = CONFINFER_PROTOCOL_VERSION;
    req.model_id = model_id;
    req.image_size = static_cast<uint32_t>(image.size());
    req.flags = 0;

    res = confinfer_teec_prepare_model_image(client(), &req,
                                             image.data(), image.size(),
                                             &rsp, &err_origin);
    if (res != TEEC_SUCCESS) {
        return false;
    }

    EXIT_ERROR_CHECK_NE(CONFINFER_PROTOCOL_VERSION, rsp.version,
                        "ExecBridge_TEE_Default prepare image response version mismatch");
    EXIT_ERROR_CHECK_NE(model_id, rsp.model_id,
                        "ExecBridge_TEE_Default prepare image response model_id mismatch");
    EXIT_ERROR_CHECK_NE(CONFINFER_STATUS_OK, rsp.status,
                        "ExecBridge_TEE_Default prepare image remote failed");
    markModelLoaded(model_id);
    return true;
}

bool ExecBridge_TEE_Default::unloadModel(confinfer_model_id_t model_id, bool strict)
{
    confinfer_unload_model_req_t req{};
    confinfer_unload_model_rsp_t rsp{};
    uint32_t err_origin = 0;
    TEEC_Result res = TEEC_SUCCESS;

    if (!ready() || loadedModelId() != model_id) {
        return !strict;
    }

    req.version = CONFINFER_PROTOCOL_VERSION;
    req.model_id = model_id;
    req.flags = 0;
    req.reserved0 = 0;

    res = confinfer_teec_unload_model(client(), &req, &rsp, &err_origin);
    if (strict) {
        EXIT_ERROR_CHECK_NE(TEEC_SUCCESS, res, "ExecBridge_TEE_Default unload model failed");
        EXIT_ERROR_CHECK_NE(CONFINFER_PROTOCOL_VERSION, rsp.version,
                            "ExecBridge_TEE_Default unload response version mismatch");
        EXIT_ERROR_CHECK_NE(model_id, rsp.model_id,
                            "ExecBridge_TEE_Default unload response model_id mismatch");
        EXIT_ERROR_CHECK_NE(CONFINFER_STATUS_OK, rsp.status,
                            "ExecBridge_TEE_Default unload remote failed");
    }
    if (res == TEEC_SUCCESS) {
        clearLoadedModel();
        return true;
    }
    return false;
}

bool ExecBridge_TEE_Default::executePartition(
    confinfer_model_id_t model_id,
    confinfer_partition_id_t partition_id,
    const ExecIOBlob& io_blob)
{
    confinfer_exec_partition_req_t req{};
    confinfer_exec_partition_rsp_t rsp{};
    uint32_t err_origin = 0;
    TEEC_Result res = TEEC_SUCCESS;

    EXIT_ERROR_CHECK_EQ(false, ready(), "ExecBridge_TEE_Default session is not open");
    EXIT_ERROR_CHECK_NE(model_id, loadedModelId(),
                        "ExecBridge_TEE_Default execute with unloaded model");

    req.version = CONFINFER_PROTOCOL_VERSION;
    req.model_id = model_id;
    req.partition_id = partition_id;
    req.input_count = io_blob.input_count;
    req.output_count = io_blob.output_count;
    req.input_bytes = io_blob.input_bytes;
    req.output_bytes = io_blob.output_bytes;
    req.flags = 0;

    res = confinfer_teec_exec_partition(client(), &req,
                                        io_blob.input_data,
                                        io_blob.input_bytes,
                                        io_blob.output_data,
                                        io_blob.output_bytes,
                                        &rsp,
                                        &err_origin);
    if (res != TEEC_SUCCESS) {
        return false;
    }

    EXIT_ERROR_CHECK_NE(CONFINFER_PROTOCOL_VERSION, rsp.version,
                        "ExecBridge_TEE_Default exec response version mismatch");
    EXIT_ERROR_CHECK_NE(model_id, rsp.model_id,
                        "ExecBridge_TEE_Default exec response model_id mismatch");
    EXIT_ERROR_CHECK_NE(partition_id, rsp.partition_id,
                        "ExecBridge_TEE_Default exec response partition_id mismatch");
    EXIT_ERROR_CHECK_NE(CONFINFER_STATUS_OK, rsp.status,
                        "ExecBridge_TEE_Default exec remote failed");
    return true;
}

bool ExecBridge_TEE_TrustSpan::loadModelImage(confinfer_model_id_t model_id,
                                              const Kernel::image::ModelImage& image)
{
    (void)model_id;
    (void)image;
    EXIT_ERROR("ExecBridge_TEE_TrustSpan is not implemented yet");
    return false;
}

bool ExecBridge_TEE_TrustSpan::unloadModel(confinfer_model_id_t model_id, bool strict)
{
    (void)model_id;
    (void)strict;
    EXIT_ERROR("ExecBridge_TEE_TrustSpan is not implemented yet");
    return false;
}

bool ExecBridge_TEE_TrustSpan::executePartition(
    confinfer_model_id_t model_id,
    confinfer_partition_id_t partition_id,
    const ExecIOBlob& io_blob)
{
    (void)model_id;
    (void)partition_id;
    (void)io_blob;
    EXIT_ERROR("ExecBridge_TEE_TrustSpan is not implemented yet");
    return false;
}

std::unique_ptr<ExecBridge_TEE> createCompiledTEEBridge()
{
#if ENABLE_TEE_BRIDGE && TRUSTSPAN
    return std::unique_ptr<ExecBridge_TEE>(new ExecBridge_TEE_TrustSpan());
#elif ENABLE_TEE_BRIDGE
    return std::unique_ptr<ExecBridge_TEE>(new ExecBridge_TEE_Default());
#else
    return nullptr;
#endif
}

} // namespace bridges
} // namespace Kernel
