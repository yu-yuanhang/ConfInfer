#include <core/cifar100_dataset.h>

#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace Kernel {
namespace core {
namespace {

namespace fs = std::filesystem;

constexpr UINT CIFAR100_IMAGE_BYTES = 3u * 32u * 32u;
constexpr FLOAT CIFAR100_MEAN[3] = {0.5071f, 0.4867f, 0.4408f};
constexpr FLOAT CIFAR100_STD[3] = {0.2675f, 0.2565f, 0.2761f};

[[noreturn]] void fail_dataset(const std::string& message) {
    throw std::runtime_error(message);
}

uint8_t read_u8(const std::vector<uint8_t>& payload, size_t& pos) {
    if (pos >= payload.size()) {
        fail_dataset("Unexpected end of CIFAR100 pickle payload");
    }
    return payload[pos++];
}

uint16_t read_u16_le(const std::vector<uint8_t>& payload, size_t& pos) {
    const uint16_t b0 = read_u8(payload, pos);
    const uint16_t b1 = read_u8(payload, pos);
    return static_cast<uint16_t>(b0 | (b1 << 8));
}

uint32_t read_u32_le(const std::vector<uint8_t>& payload, size_t& pos) {
    const uint32_t b0 = read_u8(payload, pos);
    const uint32_t b1 = read_u8(payload, pos);
    const uint32_t b2 = read_u8(payload, pos);
    const uint32_t b3 = read_u8(payload, pos);
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

int32_t read_i32_le(const std::vector<uint8_t>& payload, size_t& pos) {
    return static_cast<int32_t>(read_u32_le(payload, pos));
}

void expect_opcode(const std::vector<uint8_t>& payload,
                   size_t& pos,
                   uint8_t expected,
                   const char* message) {
    const uint8_t opcode = read_u8(payload, pos);
    if (opcode != expected) {
        fail_dataset(std::string(message) + " at byte " + std::to_string(pos - 1));
    }
}

void skip_binput(const std::vector<uint8_t>& payload, size_t& pos) {
    const uint8_t opcode = read_u8(payload, pos);
    if (opcode == 'q') {
        (void)read_u8(payload, pos);
        return;
    }
    if (opcode == 'r') {
        (void)read_u32_le(payload, pos);
        return;
    }
    fail_dataset("Expected BINPUT/LONG_BINPUT in CIFAR100 pickle");
}

std::string read_short_binstring(const std::vector<uint8_t>& payload, size_t& pos) {
    expect_opcode(payload, pos, 'U', "Expected SHORT_BINSTRING");
    const uint8_t len = read_u8(payload, pos);
    if (pos + len > payload.size()) {
        fail_dataset("SHORT_BINSTRING exceeds payload bounds");
    }
    const std::string value(reinterpret_cast<const char*>(&payload[pos]), len);
    pos += len;
    return value;
}

std::string read_global(const std::vector<uint8_t>& payload, size_t& pos) {
    expect_opcode(payload, pos, 'c', "Expected GLOBAL");
    const size_t start = pos;
    while (pos < payload.size() && payload[pos] != '\n') {
        ++pos;
    }
    if (pos >= payload.size()) {
        fail_dataset("Malformed GLOBAL opcode");
    }
    const std::string module(reinterpret_cast<const char*>(&payload[start]), pos - start);
    ++pos;

    const size_t name_start = pos;
    while (pos < payload.size() && payload[pos] != '\n') {
        ++pos;
    }
    if (pos >= payload.size()) {
        fail_dataset("Malformed GLOBAL opcode");
    }
    const std::string name(reinterpret_cast<const char*>(&payload[name_start]), pos - name_start);
    ++pos;
    return module + " " + name;
}

UINT read_pickle_uint(const std::vector<uint8_t>& payload, size_t& pos) {
    const uint8_t opcode = read_u8(payload, pos);
    switch (opcode) {
        case 'K':
            return static_cast<UINT>(read_u8(payload, pos));
        case 'M':
            return static_cast<UINT>(read_u16_le(payload, pos));
        case 'J': {
            const int32_t value = read_i32_le(payload, pos);
            if (value < 0) {
                fail_dataset("Negative integer is not supported in CIFAR100 uint field");
            }
            return static_cast<UINT>(value);
        }
        default:
            fail_dataset("Unsupported integer opcode in CIFAR100 pickle");
    }
    return 0;
}

std::vector<UINT> parse_uint_list_after_key(const std::vector<uint8_t>& payload,
                                            size_t key_pos,
                                            const std::string& key) {
    size_t pos = key_pos;
    const std::string parsed_key = read_short_binstring(payload, pos);
    if (parsed_key != key) {
        fail_dataset("Unexpected key while parsing CIFAR100 integer list");
    }
    skip_binput(payload, pos);
    expect_opcode(payload, pos, ']', "Expected EMPTY_LIST");
    skip_binput(payload, pos);

    std::vector<UINT> values;
    while (pos < payload.size() && payload[pos] == '(') {
        ++pos;
        while (pos < payload.size()) {
            if (payload[pos] == 'e') {
                ++pos;
                break;
            }
            values.push_back(read_pickle_uint(payload, pos));
        }
    }
    return values;
}

std::vector<std::string> parse_string_list_after_key(const std::vector<uint8_t>& payload,
                                                     size_t key_pos,
                                                     const std::string& key) {
    size_t pos = key_pos;
    const std::string parsed_key = read_short_binstring(payload, pos);
    if (parsed_key != key) {
        fail_dataset("Unexpected key while parsing CIFAR100 string list");
    }
    skip_binput(payload, pos);
    expect_opcode(payload, pos, ']', "Expected EMPTY_LIST");
    skip_binput(payload, pos);

    std::vector<std::string> values;
    while (pos < payload.size() && payload[pos] == '(') {
        ++pos;
        while (pos < payload.size()) {
            if (payload[pos] == 'e') {
                ++pos;
                break;
            }
            values.push_back(read_short_binstring(payload, pos));
            skip_binput(payload, pos);
        }
    }
    return values;
}

std::vector<uint8_t> parse_image_bytes_after_key(const std::vector<uint8_t>& payload,
                                                 size_t key_pos,
                                                 const std::string& key,
                                                 size_t& sample_count) {
    size_t pos = key_pos;
    const std::string parsed_key = read_short_binstring(payload, pos);
    if (parsed_key != key) {
        fail_dataset("Unexpected key while parsing CIFAR100 image bytes");
    }
    skip_binput(payload, pos);

    const std::string reconstruct = read_global(payload, pos);
    if (reconstruct != "numpy.core.multiarray _reconstruct") {
        fail_dataset("Unexpected numpy reconstruct symbol in CIFAR100 pickle");
    }
    skip_binput(payload, pos);

    const std::string ndarray_symbol = read_global(payload, pos);
    if (ndarray_symbol != "numpy ndarray") {
        fail_dataset("Unexpected numpy ndarray symbol in CIFAR100 pickle");
    }
    skip_binput(payload, pos);

    const UINT zero_value = read_pickle_uint(payload, pos);
    if (zero_value != 0) {
        fail_dataset("Unexpected ndarray constructor value in CIFAR100 pickle");
    }
    expect_opcode(payload, pos, 0x85, "Expected TUPLE1");

    const std::string order_tag = read_short_binstring(payload, pos);
    if (order_tag != "b") {
        fail_dataset("Unexpected ndarray order tag in CIFAR100 pickle");
    }
    expect_opcode(payload, pos, 0x87, "Expected TUPLE3");
    expect_opcode(payload, pos, 'R', "Expected REDUCE");
    skip_binput(payload, pos);

    expect_opcode(payload, pos, '(', "Expected MARK before ndarray shape");
    const UINT ndim = read_pickle_uint(payload, pos);
    sample_count = read_pickle_uint(payload, pos);
    const UINT sample_width = read_pickle_uint(payload, pos);
    expect_opcode(payload, pos, 0x86, "Expected TUPLE2 for ndarray shape");
    if (ndim != 1 || sample_width != CIFAR100_IMAGE_BYTES) {
        fail_dataset("Unexpected CIFAR100 ndarray shape");
    }

    const std::string dtype_symbol = read_global(payload, pos);
    if (dtype_symbol != "numpy dtype") {
        fail_dataset("Unexpected numpy dtype symbol in CIFAR100 pickle");
    }
    skip_binput(payload, pos);

    const std::string dtype_name = read_short_binstring(payload, pos);
    if (dtype_name != "u1") {
        fail_dataset("Unexpected CIFAR100 dtype");
    }
    if (read_pickle_uint(payload, pos) != 0) {
        fail_dataset("Unexpected dtype arg #1 in CIFAR100 pickle");
    }
    if (read_pickle_uint(payload, pos) != 1) {
        fail_dataset("Unexpected dtype arg #2 in CIFAR100 pickle");
    }
    expect_opcode(payload, pos, 0x87, "Expected TUPLE3 for dtype reduce");
    expect_opcode(payload, pos, 'R', "Expected REDUCE for dtype");
    skip_binput(payload, pos);

    expect_opcode(payload, pos, '(', "Expected MARK before dtype BUILD");
    if (read_pickle_uint(payload, pos) != 3) {
        fail_dataset("Unexpected dtype version in CIFAR100 pickle");
    }
    const std::string endian_tag = read_short_binstring(payload, pos);
    if (endian_tag != "|") {
        fail_dataset("Unexpected dtype endian tag in CIFAR100 pickle");
    }
    expect_opcode(payload, pos, 'N', "Expected NONE");
    expect_opcode(payload, pos, 'N', "Expected NONE");
    expect_opcode(payload, pos, 'N', "Expected NONE");
    expect_opcode(payload, pos, 'J', "Expected BININT in dtype BUILD");
    const int32_t dim0 = read_i32_le(payload, pos);
    if (dim0 != -1) {
        fail_dataset("Unexpected dtype dim0 in CIFAR100 pickle");
    }
    expect_opcode(payload, pos, 'J', "Expected BININT in dtype BUILD");
    const int32_t dim1 = read_i32_le(payload, pos);
    if (dim1 != -1) {
        fail_dataset("Unexpected dtype dim1 in CIFAR100 pickle");
    }
    if (read_pickle_uint(payload, pos) != 0) {
        fail_dataset("Unexpected dtype alignment in CIFAR100 pickle");
    }
    expect_opcode(payload, pos, 't', "Expected TUPLE");
    expect_opcode(payload, pos, 'b', "Expected BUILD");
    expect_opcode(payload, pos, 0x89, "Expected NEWFALSE");

    expect_opcode(payload, pos, 'T', "Expected BINSTRING image payload");
    const uint32_t raw_size = read_u32_le(payload, pos);
    if (raw_size != sample_count * CIFAR100_IMAGE_BYTES) {
        fail_dataset("Unexpected CIFAR100 raw image byte size");
    }
    if (pos + raw_size > payload.size()) {
        fail_dataset("CIFAR100 raw image payload exceeds file bounds");
    }

    std::vector<uint8_t> image_bytes(raw_size);
    std::memcpy(image_bytes.data(), &payload[pos], raw_size);
    pos += raw_size;

    expect_opcode(payload, pos, 't', "Expected TUPLE after image payload");
    expect_opcode(payload, pos, 'b', "Expected final BUILD after image payload");
    return image_bytes;
}

size_t find_key_position(const std::vector<uint8_t>& payload, const std::string& key) {
    const size_t key_len = key.size();
    for (size_t pos = 0; pos + 2 + key_len <= payload.size(); ++pos) {
        if (payload[pos] != 'U' || payload[pos + 1] != key_len) {
            continue;
        }
        if (0 == std::memcmp(&payload[pos + 2], key.data(), key_len)) {
            return pos;
        }
    }
    fail_dataset("Failed to locate CIFAR100 key: " + key);
    return 0;
}

std::vector<uint8_t> read_file_bytes(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.good()) {
        fail_dataset("Failed to open CIFAR100 file: " + path);
    }
    ifs.seekg(0, std::ios::end);
    const std::streamoff file_size = ifs.tellg();
    if (file_size < 0) {
        fail_dataset("Failed to query CIFAR100 file size: " + path);
    }
    ifs.seekg(0, std::ios::beg);

    std::vector<uint8_t> bytes(static_cast<size_t>(file_size));
    if (!bytes.empty()) {
        ifs.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (static_cast<size_t>(ifs.gcount()) != bytes.size()) {
            fail_dataset("Failed to read complete CIFAR100 file: " + path);
        }
    }
    return bytes;
}

} // namespace

CIFAR100Dataset::CIFAR100Dataset(const std::string& root_dir,
                                 Split split,
                                 bool normalize)
    : root_dir_(root_dir),
      split_(split),
      normalize_(normalize),
      filenames_(),
      fine_labels_(),
      coarse_labels_(),
      image_bytes_() {
    const fs::path path = fs::path(root_dir_) / splitFileName(split_);
    parsePayload(read_file_bytes(path.string()));
}

UINT CIFAR100Dataset::fineLabel(size_t index) const {
    checkIndex(index, fine_labels_.size(), "fine label");
    return fine_labels_[index];
}

UINT CIFAR100Dataset::coarseLabel(size_t index) const {
    checkIndex(index, coarse_labels_.size(), "coarse label");
    return coarse_labels_[index];
}

const std::string& CIFAR100Dataset::filename(size_t index) const {
    checkIndex(index, filenames_.size(), "filename");
    return filenames_[index];
}

void CIFAR100Dataset::loadInput(size_t index, Value_t& input) const {
    checkIndex(index, fine_labels_.size(), "image");
    const size_t offset = index * CIFAR100_IMAGE_BYTES;
    if (offset + CIFAR100_IMAGE_BYTES > image_bytes_.size()) {
        fail_dataset("CIFAR100 image offset exceeds payload bounds");
    }

    if (input.data.shape.ndim == 0 && input.data.shape.size == 0) {
        input.data.shape = DataShape_t({1, 3, 32, 32});
    }
    EXIT_ERROR_CHECK_NE(input.data.shape.ndim, 4, "CIFAR100 input ndim must be 4");
    EXIT_ERROR_CHECK_NE(input.data.shape.dims[0], 1, "CIFAR100 batch size must be 1");
    EXIT_ERROR_CHECK_NE(input.data.shape.dims[1], 3, "CIFAR100 channel must be 3");
    EXIT_ERROR_CHECK_NE(input.data.shape.dims[2], 32, "CIFAR100 height must be 32");
    EXIT_ERROR_CHECK_NE(input.data.shape.dims[3], 32, "CIFAR100 width must be 32");
    EXIT_ERROR_CHECK_NE(input.data.dtype, DataType::FP32, "CIFAR100 input dtype must be FP32");

    if (nullptr == input.data.ptr) {
        input.alloc();
    }

    FLOAT* dst = static_cast<FLOAT*>(input.data.ptr);
    for (UINT c = 0; c < 3; ++c) {
        const FLOAT mean = CIFAR100_MEAN[c];
        const FLOAT std = CIFAR100_STD[c];
        for (UINT hw = 0; hw < 32u * 32u; ++hw) {
            FLOAT value = static_cast<FLOAT>(image_bytes_[offset + c * 1024u + hw]) / 255.0f;
            if (normalize_) {
                value = (value - mean) / std;
            }
            dst[c * 1024u + hw] = value;
        }
    }
}

std::string CIFAR100Dataset::splitFileName(Split split) {
    return split == Split::TRAIN ? "train" : "test";
}

void CIFAR100Dataset::checkIndex(size_t index, size_t limit, const char* field_name) {
    if (index >= limit) {
        fail_dataset(std::string("CIFAR100 ") + field_name + " index out of range");
    }
}

void CIFAR100Dataset::parsePayload(const std::vector<uint8_t>& payload) {
    size_t sample_count = 0;
    filenames_ = parse_string_list_after_key(payload, find_key_position(payload, "filenames"), "filenames");
    fine_labels_ = parse_uint_list_after_key(payload, find_key_position(payload, "fine_labels"), "fine_labels");
    coarse_labels_ = parse_uint_list_after_key(payload, find_key_position(payload, "coarse_labels"), "coarse_labels");
    image_bytes_ = parse_image_bytes_after_key(payload, find_key_position(payload, "data"), "data", sample_count);

    if (filenames_.size() != sample_count ||
        fine_labels_.size() != sample_count ||
        coarse_labels_.size() != sample_count) {
        fail_dataset("CIFAR100 payload field sizes are inconsistent: filenames="
            + std::to_string(filenames_.size())
            + " fine_labels=" + std::to_string(fine_labels_.size())
            + " coarse_labels=" + std::to_string(coarse_labels_.size())
            + " sample_count=" + std::to_string(sample_count));
    }
}

} // namespace core
} // namespace Kernel
