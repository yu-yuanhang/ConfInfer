#ifndef __CIFAR100_DATASET_H_CA__
#define __CIFAR100_DATASET_H_CA__

#include <All.h>
#include <core/Param.h>

namespace Kernel {
namespace core {

class CIFAR100Dataset {
public:
    enum class Split : uint8_t {
        TRAIN = 0,
        TEST = 1,
    };

    CIFAR100Dataset(const std::string& root_dir,
                    Split split,
                    bool normalize = true);

    size_t size() const { return fine_labels_.size(); }
    bool normalize() const { return normalize_; }
    UINT fineLabel(size_t index) const;
    UINT coarseLabel(size_t index) const;
    const std::string& filename(size_t index) const;

    // 将单个样本写入 NCHW = [1, 3, 32, 32] 的运行时输入。
    void loadInput(size_t index, Value_t& input) const;

private:
    static std::string splitFileName(Split split);
    static void checkIndex(size_t index, size_t limit, const char* field_name);

    void parsePayload(const std::vector<uint8_t>& payload);

private:
    std::string root_dir_;
    Split split_;
    bool normalize_;

    std::vector<std::string> filenames_;
    std::vector<UINT> fine_labels_;
    std::vector<UINT> coarse_labels_;
    std::vector<uint8_t> image_bytes_;
};

} // namespace core
} // namespace Kernel

#endif
