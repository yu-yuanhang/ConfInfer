#ifndef __IO_PACK_H_CA__
#define __IO_PACK_H_CA__

#include <All.h>
#include <core/Param.h>

namespace Kernel {
namespace core {

struct TensorView {
    void* ptr;
    DataType dtype;
    DataShape_t shape;
    DataLocation location;
    uint32_t flags;

    TensorView()
        : ptr(nullptr),
          dtype(DataType::FP32),
          shape(),
          location(DataLocation::CPU),
          flags(0) {}
};

struct InputPack {
    std::vector<TensorView> values;
};

struct OutputPack {
    std::vector<TensorView> values;
};

} // namespace core
} // namespace Kernel

#endif
