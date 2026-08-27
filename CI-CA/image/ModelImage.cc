#include <image/ModelImage.h>

namespace Kernel {
namespace image {

namespace {

template<typename T>
bool range_is_inside(size_t offset, size_t span, size_t total)
{
    return offset <= total && span <= total - offset;
}

template<typename T>
T *offset_ptr(uint8_t *base, size_t total, size_t offset, size_t span)
{
    if (nullptr == base || !range_is_inside<T>(offset, span, total)) {
        return nullptr;
    }
    return reinterpret_cast<T *>(base + offset);
}

template<typename T>
const T *offset_ptr(const uint8_t *base, size_t total, size_t offset, size_t span)
{
    if (nullptr == base || !range_is_inside<T>(offset, span, total)) {
        return nullptr;
    }
    return reinterpret_cast<const T *>(base + offset);
}

} // namespace

ModelImage::ModelImage() : _buffer() {}

bool ModelImage::valid() const
{
    const ModelImageHeader *hdr = header();
    const uint32_t partition_table_span =
        hdr ? hdr->partition_count * static_cast<uint32_t>(sizeof(ModelImagePartitionEntry)) : 0;
    const uint32_t param_desc_span =
        hdr ? hdr->param_desc_count * static_cast<uint32_t>(sizeof(ModelImageParamDesc)) : 0;

    if (nullptr == hdr) {
        return false;
    }
    if (hdr->magic != CONFINFER_MODEL_IMAGE_MAGIC) {
        return false;
    }
    if (hdr->version_major != CONFINFER_MODEL_IMAGE_VERSION_MAJOR) {
        return false;
    }
    if (hdr->header_size < sizeof(ModelImageHeader)) {
        return false;
    }
    if (hdr->total_size != _buffer.size()) {
        return false;
    }
    if (!range_is_inside<void>(hdr->partition_table_off, partition_table_span, _buffer.size())) {
        return false;
    }
    if (hdr->partition_table_size != partition_table_span) {
        return false;
    }
    if (!range_is_inside<void>(hdr->param_desc_off, param_desc_span, _buffer.size())) {
        return false;
    }
    if (!range_is_inside<void>(hdr->param_data_off, hdr->param_data_size, _buffer.size())) {
        return false;
    }

    const ModelImagePartitionEntry *entries = partitionTable();
    if (hdr->partition_count > 0 && nullptr == entries) {
        return false;
    }
    for (uint32_t i = 0; i < hdr->partition_count; ++i) {
        if (!range_is_inside<void>(entries[i].image_off, entries[i].image_size, _buffer.size())) {
            return false;
        }
    }
    return true;
}

const ModelImageHeader *ModelImage::header() const
{
    if (_buffer.size() < sizeof(ModelImageHeader)) {
        return nullptr;
    }
    return reinterpret_cast<const ModelImageHeader *>(_buffer.data());
}

ModelImageHeader *ModelImage::header()
{
    if (_buffer.size() < sizeof(ModelImageHeader)) {
        return nullptr;
    }
    return reinterpret_cast<ModelImageHeader *>(_buffer.data());
}

const ModelImagePartitionEntry *ModelImage::partitionTable() const
{
    const ModelImageHeader *hdr = header();
    if (nullptr == hdr) {
        return nullptr;
    }
    return offset_ptr<ModelImagePartitionEntry>(_buffer.data(),
                                                _buffer.size(),
                                                hdr->partition_table_off,
                                                hdr->partition_count * sizeof(ModelImagePartitionEntry));
}

ModelImagePartitionEntry *ModelImage::partitionTable()
{
    ModelImageHeader *hdr = header();
    if (nullptr == hdr) {
        return nullptr;
    }
    return offset_ptr<ModelImagePartitionEntry>(_buffer.data(),
                                                _buffer.size(),
                                                hdr->partition_table_off,
                                                hdr->partition_count * sizeof(ModelImagePartitionEntry));
}

const ModelImagePartitionEntry *ModelImage::findPartition(confinfer_partition_id_t partition_id) const
{
    const ModelImageHeader *hdr = header();
    const ModelImagePartitionEntry *entries = partitionTable();

    if (nullptr == hdr || nullptr == entries) {
        return nullptr;
    }
    for (uint32_t i = 0; i < hdr->partition_count; ++i) {
        if (entries[i].partition_id == partition_id) {
            return &entries[i];
        }
    }
    return nullptr;
}

ModelImagePartitionEntry *ModelImage::findPartition(confinfer_partition_id_t partition_id)
{
    ModelImageHeader *hdr = header();
    ModelImagePartitionEntry *entries = partitionTable();

    if (nullptr == hdr || nullptr == entries) {
        return nullptr;
    }
    for (uint32_t i = 0; i < hdr->partition_count; ++i) {
        if (entries[i].partition_id == partition_id) {
            return &entries[i];
        }
    }
    return nullptr;
}

const ModelImageParamDesc *ModelImage::paramDescs() const
{
    const ModelImageHeader *hdr = header();
    if (nullptr == hdr) {
        return nullptr;
    }
    return offset_ptr<ModelImageParamDesc>(_buffer.data(),
                                           _buffer.size(),
                                           hdr->param_desc_off,
                                           hdr->param_desc_count * sizeof(ModelImageParamDesc));
}

ModelImageParamDesc *ModelImage::paramDescs()
{
    ModelImageHeader *hdr = header();
    if (nullptr == hdr) {
        return nullptr;
    }
    return offset_ptr<ModelImageParamDesc>(_buffer.data(),
                                           _buffer.size(),
                                           hdr->param_desc_off,
                                           hdr->param_desc_count * sizeof(ModelImageParamDesc));
}

const uint8_t *ModelImage::paramData() const
{
    const ModelImageHeader *hdr = header();
    if (nullptr == hdr) {
        return nullptr;
    }
    return offset_ptr<uint8_t>(_buffer.data(),
                               _buffer.size(),
                               hdr->param_data_off,
                               hdr->param_data_size);
}

uint8_t *ModelImage::paramData()
{
    ModelImageHeader *hdr = header();
    if (nullptr == hdr) {
        return nullptr;
    }
    return offset_ptr<uint8_t>(_buffer.data(),
                               _buffer.size(),
                               hdr->param_data_off,
                               hdr->param_data_size);
}

const uint8_t *ModelImage::partitionImage(const ModelImagePartitionEntry& entry) const
{
    return offset_ptr<uint8_t>(_buffer.data(),
                               _buffer.size(),
                               entry.image_off,
                               entry.image_size);
}

uint8_t *ModelImage::partitionImage(const ModelImagePartitionEntry& entry)
{
    return offset_ptr<uint8_t>(_buffer.data(),
                               _buffer.size(),
                               entry.image_off,
                               entry.image_size);
}

void ModelImage::reset()
{
    _buffer.clear();
}

void ModelImage::reset(std::vector<uint8_t> bytes)
{
    _buffer = std::move(bytes);
}

} // namespace image
} // namespace Kernel
