#ifndef __MODEL_IMAGE_H_CA__
#define __MODEL_IMAGE_H_CA__

#include <confinfer_protocol.h>
#include <core/ExecutionPartition.h>
#include <image/ModelImageFormat.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace Kernel {
namespace image {

class ModelImage {
public:
    ModelImage();

    const uint8_t *data() const { return _buffer.empty() ? nullptr : _buffer.data(); }
    uint8_t *data() { return _buffer.empty() ? nullptr : _buffer.data(); }
    size_t size() const { return _buffer.size(); }
    bool empty() const { return _buffer.empty(); }
    bool valid() const;

    const ModelImageHeader *header() const;
    ModelImageHeader *header();
    const ModelImagePartitionEntry *partitionTable() const;
    ModelImagePartitionEntry *partitionTable();
    const ModelImagePartitionEntry *findPartition(confinfer_partition_id_t partition_id) const;
    ModelImagePartitionEntry *findPartition(confinfer_partition_id_t partition_id);
    const ModelImageParamDesc *paramDescs() const;
    ModelImageParamDesc *paramDescs();
    const uint8_t *paramData() const;
    uint8_t *paramData();
    const uint8_t *partitionImage(const ModelImagePartitionEntry& entry) const;
    uint8_t *partitionImage(const ModelImagePartitionEntry& entry);

    const std::vector<uint8_t>& buffer() const { return _buffer; }
    std::vector<uint8_t>& buffer() { return _buffer; }

    void reset();
    void reset(std::vector<uint8_t> bytes);

private:
    std::vector<uint8_t> _buffer;
};

class ModelImageBuilder {
public:
    ModelImage build(confinfer_model_id_t model_id,
                     const std::vector<Kernel::core::ExecPartition>& parts,
                     ModelImageExecMode exec_mode = ModelImageExecMode::TEE_SINGLE,
                     uint64_t reserved_phys_base = 0,
                     uint64_t reserved_phys_size = 0) const;
};

} // namespace image
} // namespace Kernel

#endif
