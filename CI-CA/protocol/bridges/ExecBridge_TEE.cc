#include <bridges/ExecBridge_TEE.h>

#include <cstring>

#if TRUSTSPAN
#include <trustspan_mem_test.h>

#include <fcntl.h>
#include <limits>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace Kernel {
namespace bridges {

namespace {

#if TRUSTSPAN
constexpr const char *kTrustSpanDevicePath = "/dev/trustspan_mem_test";
#endif

} // namespace

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

bool ExecBridge_TEE_TrustSpan::allocateRegion(size_t size)
{
#if TRUSTSPAN
    trustspan_mem_info info{};
    int fd = -1;

    if (size == 0 || size > std::numeric_limits<uint32_t>::max()) {
        return false;
    }

    releaseRegion();

    fd = ::open(kTrustSpanDevicePath, O_RDWR);
    if (fd < 0) {
        return false;
    }

    info.size = static_cast<__u64>(size);
    if (::ioctl(fd, TRUSTSPAN_MEM_TEST_IOC_ALLOC, &info) < 0 ||
        !info.allocated || info.size < size ||
        info.size > std::numeric_limits<uint32_t>::max()) {
        ::close(fd);
        return false;
    }

    _model_region.fd = fd;
    _model_region.phys_addr = info.phys_addr;
    _model_region.size = static_cast<uint32_t>(info.size);
    if (!mapRegion()) {
        (void)::ioctl(fd, TRUSTSPAN_MEM_TEST_IOC_FREE);
        (void)::close(fd);
        _model_region = TrustSpanModelRegion{};
        return false;
    }
    return true;
#else
    (void)size;
    return false;
#endif
}

ExecBridge_TEE_TrustSpan::~ExecBridge_TEE_TrustSpan()
{
    if (loadedModelId() != CONFINFER_INVALID_MODEL_ID) {
        (void)unloadModel(loadedModelId(), false);
    }
    releaseRegion();
}

bool ExecBridge_TEE_TrustSpan::copyToRegion(
    const Kernel::image::ModelImage& image)
{
    if (!image.valid() || image.empty() ||
        _model_region.ree_addr == nullptr ||
        _model_region.size < image.size()) {
        return false;
    }

    std::memcpy(_model_region.ree_addr, image.data(), image.size());
    return true;
}

bool ExecBridge_TEE_TrustSpan::mapRegion()
{
#if TRUSTSPAN
    void *ree_addr = nullptr;

    if (_model_region.fd < 0 || _model_region.size == 0 ||
        _model_region.ree_addr != nullptr) {
        return false;
    }

    ree_addr = ::mmap(nullptr, _model_region.size, PROT_READ | PROT_WRITE,
                      MAP_SHARED, _model_region.fd, 0);
    if (ree_addr == MAP_FAILED) {
        return false;
    }
    _model_region.ree_addr = static_cast<uint8_t *>(ree_addr);
    return true;
#else
    return false;
#endif
}

bool ExecBridge_TEE_TrustSpan::unmapRegion()
{
#if TRUSTSPAN
    if (_model_region.ree_addr == nullptr) {
        return true;
    }
    if (::munmap(_model_region.ree_addr, _model_region.size) != 0) {
        return false;
    }
    _model_region.ree_addr = nullptr;
    return true;
#else
    return false;
#endif
}

bool ExecBridge_TEE_TrustSpan::cancelSecureRegion()
{
#if TRUSTSPAN
    if (_model_region.fd < 0 ||
        ::ioctl(_model_region.fd, TRUSTSPAN_MEM_TEST_IOC_CANCEL_SECURE) < 0) {
        return false;
    }
    return mapRegion();
#else
    return false;
#endif
}

void ExecBridge_TEE_TrustSpan::releaseRegion()
{
#if TRUSTSPAN
    (void)unmapRegion();
    if (_model_region.fd >= 0) {
        (void)::ioctl(_model_region.fd, TRUSTSPAN_MEM_TEST_IOC_FREE);
        (void)::close(_model_region.fd);
    }
#endif
    _model_region = TrustSpanModelRegion{};
}

// ExecBridge_TEE_TrustSpan::loadModelImage() 的职责是：
// 把 CA 侧已有的通用 ModelImage 复制到 TrustSpan 物理连续区
// 然后将该物理区安全交给 TA 并让 TA 直接解析这份 Image
bool ExecBridge_TEE_TrustSpan::loadModelImage(confinfer_model_id_t model_id,
                                              const Kernel::image::ModelImage& image)
{
    confinfer_prepare_model_image_trustspan_req_t req{};
    confinfer_prepare_model_image_rsp_t rsp{};
    uint32_t err_origin = 0;
    TEEC_Result res = TEEC_SUCCESS;
    bool model_prepared = false;
    bool region_released = false;

    EXIT_ERROR_CHECK_EQ(false, ready(), "ExecBridge_TEE_TrustSpan session is not open");
    if (loadedModelId() == model_id) {
        return true;
    }
    if (!image.valid() || image.empty() ||
        image.size() > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    if (!allocateRegion(image.size()) || !copyToRegion(image)) {
        releaseRegion();
        return false;
    }

    // 在此前的流程还是 比较 容易处理的
    // CA ModelImage -> allocateRegion -> REE mmap 得到 ree_addr -> copyToRegion
    // 但是 设计到 TA 侧的 secure handoff 就比较复杂了
    // 需要考虑 prepare 失败的情况进行 重新映射

#if TRUSTSPAN
    // REE 完成写入后必须撤销自身映射
    // 驱动只会在无 REE 映射时允许进入 secure handoff
    if (!unmapRegion()) {
        releaseRegion();
        return false;
    }
    if (::ioctl(_model_region.fd, TRUSTSPAN_MEM_TEST_IOC_PREPARE_SECURE) < 0) {
        releaseRegion();
        return false;
    }
#endif

    req.version = CONFINFER_PROTOCOL_VERSION;
    req.model_id = model_id;
    req.image_size = static_cast<uint32_t>(image.size());
    req.flags = 0;
    req.phys_addr = _model_region.phys_addr;
    req.region_size = _model_region.size;

    res = confinfer_teec_prepare_model_image_trustspan(client(), &req, &rsp,
                                                        &err_origin);
    model_prepared = res == TEEC_SUCCESS &&
                     rsp.version == CONFINFER_PROTOCOL_VERSION &&
                     rsp.model_id == model_id &&
                     rsp.status == CONFINFER_STATUS_OK &&
                     rsp.loaded_image_size == image.size();
    if (!model_prepared) {  // prepare 失败了 必须重新映射 REE 才能继续使用这段 Region
        // 这里是 确认 TA 是否明确告诉 REE 自己已经没有使用这段 Region 了
        region_released = rsp.version == CONFINFER_PROTOCOL_VERSION &&
                          rsp.model_id == model_id &&
                          (rsp.flags &
                           CONFINFER_PREPARE_MODEL_IMAGE_RSP_FLAG_REGION_RELEASED);
        if (region_released) {
            (void)cancelSecureRegion();
        }
        return false;
    }

#if TRUSTSPAN
    if (::ioctl(_model_region.fd, TRUSTSPAN_MEM_TEST_IOC_MARK_PROTECTED) < 0) {
        // 到此为止 TA 已经成功 prepare
        // TA 已经持有 Image 的 Secure 映射
        // TF-A 已经完成 protect
        // 但 Linux 驱动执行 MARK_PROTECTED 失败
        // 后面处理就是 直接一次性全部卸载 res = confinfer_teec_unload_model(...);
        confinfer_unload_model_req_t unload_req{};
        confinfer_unload_model_rsp_t unload_rsp{};

        unload_req.version = CONFINFER_PROTOCOL_VERSION;
        unload_req.model_id = model_id;
        res = confinfer_teec_unload_model(client(), &unload_req, &unload_rsp,
                                          &err_origin);
        if (res == TEEC_SUCCESS &&
            unload_rsp.version == CONFINFER_PROTOCOL_VERSION &&
            unload_rsp.model_id == model_id &&
            unload_rsp.status == CONFINFER_STATUS_OK) { // 然后 重新 状态回滚到 unloaded 之前
            (void)cancelSecureRegion();
        }
        return false;
    }
#endif

    markModelLoaded(model_id);
    return true;
}

bool ExecBridge_TEE_TrustSpan::unloadModel(confinfer_model_id_t model_id, bool strict)
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
        EXIT_ERROR_CHECK_NE(TEEC_SUCCESS, res,
                            "ExecBridge_TEE_TrustSpan unload model failed");
        EXIT_ERROR_CHECK_NE(CONFINFER_PROTOCOL_VERSION, rsp.version,
                            "ExecBridge_TEE_TrustSpan unload response version mismatch");
        EXIT_ERROR_CHECK_NE(model_id, rsp.model_id,
                            "ExecBridge_TEE_TrustSpan unload response model_id mismatch");
        EXIT_ERROR_CHECK_NE(CONFINFER_STATUS_OK, rsp.status,
                            "ExecBridge_TEE_TrustSpan unload remote failed");
    }
    if (res != TEEC_SUCCESS ||
        rsp.version != CONFINFER_PROTOCOL_VERSION ||
        rsp.model_id != model_id ||
        rsp.status != CONFINFER_STATUS_OK) {
        return false;
    }

#if TRUSTSPAN
    // TA unload 已完成 Span PTA release 和 TF-A unprotect
    // 此时驱动才可以恢复页面的普通所有权并允许 FREE
    if (::ioctl(_model_region.fd, TRUSTSPAN_MEM_TEST_IOC_RELEASE_SECURE) < 0) {
        return false;
    }
#endif

    clearLoadedModel();
    releaseRegion();
    return true;
}

bool ExecBridge_TEE_TrustSpan::executePartition(
    confinfer_model_id_t model_id,
    confinfer_partition_id_t partition_id,
    const ExecIOBlob& io_blob)
{
    confinfer_exec_partition_req_t req{};
    confinfer_exec_partition_rsp_t rsp{};
    uint32_t err_origin = 0;
    TEEC_Result res = TEEC_SUCCESS;

    EXIT_ERROR_CHECK_EQ(false, ready(), "ExecBridge_TEE_TrustSpan session is not open");
    EXIT_ERROR_CHECK_NE(model_id, loadedModelId(),
                        "ExecBridge_TEE_TrustSpan execute with unloaded model");

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
                        "ExecBridge_TEE_TrustSpan exec response version mismatch");
    EXIT_ERROR_CHECK_NE(model_id, rsp.model_id,
                        "ExecBridge_TEE_TrustSpan exec response model_id mismatch");
    EXIT_ERROR_CHECK_NE(partition_id, rsp.partition_id,
                        "ExecBridge_TEE_TrustSpan exec response partition_id mismatch");
    EXIT_ERROR_CHECK_NE(CONFINFER_STATUS_OK, rsp.status,
                        "ExecBridge_TEE_TrustSpan exec remote failed");
    return true;
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
