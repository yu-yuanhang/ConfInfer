#include <core/model_param_loader.h>

#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace Kernel {
namespace core {
namespace {

namespace fs = std::filesystem;

[[noreturn]] void fail_json(const std::string& message) {
    throw std::runtime_error(message);
}

enum class JsonType : uint8_t {
    NIL,
    BOOL,
    NUMBER,
    STRING,
    ARRAY,
    OBJECT,
};

struct JsonValue {
    JsonType type;
    bool bool_value;
    double number_value;
    std::string string_value;
    std::vector<std::shared_ptr<JsonValue>> array_value;
    std::unordered_map<std::string, std::shared_ptr<JsonValue>> object_value;

    JsonValue() : type(JsonType::NIL), bool_value(false), number_value(0.0) {}
};

class JsonParser {
public:
    explicit JsonParser(const std::string& text) : text_(text), pos_(0) {}

    JsonValue parse() {
        skipWhitespace();
        JsonValue value = parseValue();
        skipWhitespace();
        if (pos_ != text_.size()) {
            fail_json("Unexpected trailing content in JSON at position " + std::to_string(pos_));
        }
        return value;
    }

private:
    JsonValue parseValue() {
        if (pos_ >= text_.size()) {
            fail_json("Unexpected end of JSON");
        }
        const char ch = text_[pos_];
        if ('{' == ch) {
            return parseObject();
        }
        if ('[' == ch) {
            return parseArray();
        }
        if ('\"' == ch) {
            JsonValue value;
            value.type = JsonType::STRING;
            value.string_value = parseString();
            return value;
        }
        if ('t' == ch || 'f' == ch) {
            return parseBool();
        }
        if ('n' == ch) {
            return parseNull();
        }
        if ('-' == ch || std::isdigit(static_cast<unsigned char>(ch))) {
            return parseNumber();
        }
        fail_json("Unsupported JSON token at position " + std::to_string(pos_));
        return JsonValue();
    }

    JsonValue parseObject() {
        JsonValue object;
        object.type = JsonType::OBJECT;
        expect('{');
        skipWhitespace();
        if (consume('}')) {
            return object;
        }

        while (true) {
            skipWhitespace();
            if (pos_ >= text_.size()) {
                fail_json("Unexpected end of JSON while parsing object key");
            }
            if (text_[pos_] != '\"') {
                fail_json("JSON object key must be string at position " + std::to_string(pos_));
            }
            const std::string key = parseString();
            skipWhitespace();
            expect(':');
            skipWhitespace();
            object.object_value[key] = std::make_shared<JsonValue>(parseValue());
            skipWhitespace();
            if (consume('}')) {
                break;
            }
            expect(',');
        }
        return object;
    }

    JsonValue parseArray() {
        JsonValue array;
        array.type = JsonType::ARRAY;
        expect('[');
        skipWhitespace();
        if (consume(']')) {
            return array;
        }

        while (true) {
            skipWhitespace();
            array.array_value.push_back(std::make_shared<JsonValue>(parseValue()));
            skipWhitespace();
            if (consume(']')) {
                break;
            }
            expect(',');
        }
        return array;
    }

    std::string parseString() {
        expect('\"');
        std::string value;
        while (pos_ < text_.size()) {
            const char ch = text_[pos_++];
            if ('\"' == ch) {
                return value;
            }
            if ('\\' == ch) {
                if (pos_ >= text_.size()) {
                    fail_json("Invalid JSON escape at end of string");
                }
                const char escaped = text_[pos_++];
                switch (escaped) {
                    case '\"': value.push_back('\"'); break;
                    case '\\': value.push_back('\\'); break;
                    case '/': value.push_back('/'); break;
                    case 'b': value.push_back('\b'); break;
                    case 'f': value.push_back('\f'); break;
                    case 'n': value.push_back('\n'); break;
                    case 'r': value.push_back('\r'); break;
                    case 't': value.push_back('\t'); break;
                    default:
                        fail_json(std::string("Unsupported JSON escape \\") + escaped);
                }
                continue;
            }
            value.push_back(ch);
        }
        fail_json("Unterminated JSON string");
        return "";
    }

    JsonValue parseBool() {
        JsonValue value;
        value.type = JsonType::BOOL;
        if (match("true")) {
            value.bool_value = true;
            return value;
        }
        if (match("false")) {
            value.bool_value = false;
            return value;
        }
        fail_json("Invalid JSON boolean at position " + std::to_string(pos_));
        return JsonValue();
    }

    JsonValue parseNull() {
        EXIT_ERROR_CHECK_EQ(false, match("null"), "Invalid JSON null at position %zu", pos_);
        return JsonValue();
    }

    JsonValue parseNumber() {
        const size_t start = pos_;
        if ('-' == text_[pos_]) {
            ++pos_;
        }
        while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
            ++pos_;
        }
        if (pos_ < text_.size() && '.' == text_[pos_]) {
            ++pos_;
            while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
                ++pos_;
            }
        }
        if (pos_ < text_.size() && ('e' == text_[pos_] || 'E' == text_[pos_])) {
            ++pos_;
            if (pos_ < text_.size() && ('+' == text_[pos_] || '-' == text_[pos_])) {
                ++pos_;
            }
            while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
                ++pos_;
            }
        }

        JsonValue value;
        value.type = JsonType::NUMBER;
        value.number_value = std::stod(text_.substr(start, pos_ - start));
        return value;
    }

    void skipWhitespace() {
        while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_]))) {
            ++pos_;
        }
    }

    void expect(char expected) {
        if (pos_ >= text_.size()) {
            fail_json(std::string("Unexpected end of JSON, expected '") + expected + "'");
        }
        if (text_[pos_] != expected) {
            fail_json("Unexpected JSON character at position " + std::to_string(pos_)
                + ", expected '" + std::string(1, expected)
                + "', got '" + std::string(1, text_[pos_]) + "'");
        }
        ++pos_;
    }

    bool consume(char expected) {
        if (pos_ < text_.size() && text_[pos_] == expected) {
            ++pos_;
            return true;
        }
        return false;
    }

    bool match(const char* literal) {
        const size_t len = std::strlen(literal);
        if (text_.compare(pos_, len, literal) == 0) {
            pos_ += len;
            return true;
        }
        return false;
    }

    const std::string& text_;
    size_t pos_;
};

struct ManifestParamEntry {
    std::string name;
    std::vector<UINT> shape;
    DataType dtype;
    std::string file;
    size_t numel;
    size_t bytes;
};

struct ParamManifest {
    UINT format_version;
    std::string model_file;
    std::string source_format;
    std::string export_dtype;
    UINT entry_count;
    std::unordered_map<std::string, ManifestParamEntry> entries;
};

const JsonValue& getRequiredField(const JsonValue& object,
                                  const std::string& key,
                                  JsonType expected_type) {
    EXIT_ERROR_CHECK_NE(object.type, JsonType::OBJECT, "JSON value is not object");
    auto it = object.object_value.find(key);
    EXIT_ERROR_CHECK_EQ(it, object.object_value.end(), "JSON field missing: %s", key.c_str());
    EXIT_ERROR_CHECK_NE(it->second->type, expected_type, "JSON field type mismatch: %s", key.c_str());
    return *(it->second);
}

bool hasField(const JsonValue& object, const std::string& key) {
    EXIT_ERROR_CHECK_NE(object.type, JsonType::OBJECT, "JSON value is not object");
    return object.object_value.find(key) != object.object_value.end();
}

DataType parseDataType(const std::string& dtype) {
    if ("float32" == dtype || "fp32" == dtype) {
        return DataType::FP32;
    }
    if ("float16" == dtype || "fp16" == dtype) {
        return DataType::FP16;
    }
    if ("int8" == dtype) {
        return DataType::INT8;
    }
    if ("int32" == dtype) {
        return DataType::INT32;
    }
    EXIT_ERROR("Unsupported dtype in param manifest: %s", dtype.c_str());
    return DataType::FP32;
}

size_t getTypeBytes(DataType dtype) {
    switch (dtype) {
        case DataType::INT8:
            return sizeof(int8_t);
        case DataType::INT32:
            return sizeof(int32_t);
        case DataType::FP16:
            return sizeof(fp16_t);
        case DataType::FP32:
            return sizeof(float32);
    }
    EXIT_ERROR("Unsupported DataType");
    return 0;
}

size_t calcNumel(const std::vector<UINT>& shape) {
    if (shape.empty()) {
        return 1;
    }
    size_t numel = 1;
    for (UINT dim : shape) {
        numel *= static_cast<size_t>(dim);
    }
    return numel;
}

std::vector<UINT> parseShape(const JsonValue& array) {
    EXIT_ERROR_CHECK_NE(array.type, JsonType::ARRAY, "Param shape is not array");
    std::vector<UINT> shape;
    shape.reserve(array.array_value.size());
    for (const std::shared_ptr<JsonValue>& dim : array.array_value) {
        EXIT_ERROR_CHECK_EQ(nullptr, dim, "Shape dim is nullptr");
        EXIT_ERROR_CHECK_NE(dim->type, JsonType::NUMBER, "Shape dim is not number");
        EXIT_ERROR_CHECK_EQ(true, dim->number_value < 0.0, "Shape dim must be non-negative");
        shape.push_back(static_cast<UINT>(dim->number_value));
    }
    return shape;
}

ParamManifest parseManifest(const std::string& manifest_path) {
    std::ifstream ifs(manifest_path);
    EXIT_ERROR_CHECK_EQ(false, ifs.good(), "Failed to open manifest file: %s", manifest_path.c_str());

    std::stringstream buffer;
    buffer << ifs.rdbuf();
    const std::string json_text = buffer.str();
    std::cout << "[param_loader] manifest bytes read=" << json_text.size() << std::endl;
    JsonValue root;
    try {
        JsonParser parser(json_text);
        root = parser.parse();
    } catch (const std::exception& e) {
        std::cerr << "[param_loader] json parse failed: " << e.what() << std::endl;
        throw;
    }
    std::cout << "[param_loader] manifest json root parsed" << std::endl;

    ParamManifest manifest;
    manifest.format_version = static_cast<UINT>(getRequiredField(root, "format_version", JsonType::NUMBER).number_value);
    manifest.model_file = getRequiredField(root, "model_file", JsonType::STRING).string_value;
    manifest.source_format = getRequiredField(root, "source_format", JsonType::STRING).string_value;
    manifest.export_dtype = getRequiredField(root, "export_dtype", JsonType::STRING).string_value;
    manifest.entry_count = static_cast<UINT>(getRequiredField(root, "param_count", JsonType::NUMBER).number_value);

    const JsonValue& manifest_entries = getRequiredField(root, "params", JsonType::OBJECT);
    for (const auto& item : manifest_entries.object_value) {
        EXIT_ERROR_CHECK_EQ(nullptr, item.second, "Manifest entry is nullptr");
        const JsonValue& entry_json = *(item.second);
        ManifestParamEntry entry;
        entry.name = item.first;
        entry.shape = parseShape(getRequiredField(entry_json, "shape", JsonType::ARRAY));
        entry.dtype = parseDataType(getRequiredField(entry_json, "dtype", JsonType::STRING).string_value);
        entry.file = getRequiredField(entry_json, "file", JsonType::STRING).string_value;
        entry.numel = calcNumel(entry.shape);
        entry.bytes = entry.numel * getTypeBytes(entry.dtype);

        if (hasField(entry_json, "numel")) {
            const size_t json_numel =
                static_cast<size_t>(getRequiredField(entry_json, "numel", JsonType::NUMBER).number_value);
            EXIT_ERROR_CHECK_NE(json_numel, entry.numel, "Param numel mismatch: %s", entry.name.c_str());
        }
        if (hasField(entry_json, "bytes")) {
            const size_t json_bytes =
                static_cast<size_t>(getRequiredField(entry_json, "bytes", JsonType::NUMBER).number_value);
            EXIT_ERROR_CHECK_NE(json_bytes, entry.bytes, "Param bytes mismatch: %s", entry.name.c_str());
        }

        manifest.entries[entry.name] = entry;
    }

    EXIT_ERROR_CHECK_NE(manifest.entry_count, manifest.entries.size(), "manifest entry count mismatch");
    return manifest;
}

void assignShape(Data_t& data, const std::vector<UINT>& shape) {
    EXIT_ERROR_CHECK_EQ(true, shape.size() > PARAM_MAX_DIMS, "Shape ndim exceeds PARAM_MAX_DIMS");
    data.shape.ndim = static_cast<UINT>(shape.size());
    data.shape.size = static_cast<UINT>(calcNumel(shape));
    for (UINT i = 0; i < PARAM_MAX_DIMS; ++i) {
        data.shape.dims[i] = 0;
    }
    for (UINT i = 0; i < shape.size(); ++i) {
        data.shape.dims[i] = shape[i];
    }
}

bool shapeIsEmpty(const Data_t& data) {
    return 0 == data.shape.ndim && 0 == data.shape.size;
}

void validateOrInitTarget(Data_t& target,
                          const ManifestParamEntry& entry,
                          bool init_empty_target_meta) {
    if (shapeIsEmpty(target)) {
        EXIT_ERROR_CHECK_EQ(false, init_empty_target_meta,
            "Target shape is empty for param: %s", entry.name.c_str());
        assignShape(target, entry.shape);
        target.dtype = entry.dtype;
    } else {
        EXIT_ERROR_CHECK_NE(target.shape.ndim, entry.shape.size(),
            "Target ndim mismatch for param: %s", entry.name.c_str());
        for (UINT i = 0; i < entry.shape.size(); ++i) {
            EXIT_ERROR_CHECK_NE(target.shape.dims[i], entry.shape[i],
                "Target shape mismatch for param: %s", entry.name.c_str());
        }
        EXIT_ERROR_CHECK_NE(target.shape.size, entry.numel,
            "Target size mismatch for param: %s", entry.name.c_str());
        EXIT_ERROR_CHECK_NE(target.dtype, entry.dtype,
            "Target dtype mismatch for param: %s", entry.name.c_str());
    }

    if (nullptr == target.ptr) {
        target.alloc();
    }
}

void loadParamFile(const fs::path& param_path,
                    const ManifestParamEntry& entry,
                    Data_t& target) {
    std::ifstream ifs(param_path, std::ios::binary);
    EXIT_ERROR_CHECK_EQ(false, ifs.good(), "Failed to open param file: %s", param_path.c_str());

    ifs.seekg(0, std::ios::end);
    const std::streamoff file_size = ifs.tellg();
    EXIT_ERROR_CHECK_EQ(true, file_size < 0, "Failed to query param file size: %s", param_path.c_str());
    EXIT_ERROR_CHECK_NE(static_cast<size_t>(file_size), entry.bytes,
        "Param file size mismatch: %s", entry.name.c_str());
    ifs.seekg(0, std::ios::beg);
    ifs.read(static_cast<char*>(target.ptr), static_cast<std::streamsize>(entry.bytes));
    EXIT_ERROR_CHECK_NE(static_cast<size_t>(ifs.gcount()), entry.bytes,
        "Failed to read complete param file: %s", entry.name.c_str());
}

} // namespace

void ParamBindingTable::add(const std::string& external_name,
                            Data_t* data,
                            const std::string& target_name) {
    EXIT_ERROR_CHECK_EQ(nullptr, data, "ParamBinding data is nullptr");
    EXIT_ERROR_CHECK_NE(nullptr, find(external_name), "Duplicate external param name: %s", external_name.c_str());
    bindings_.push_back(ParamBinding(external_name, data, target_name));
}

const ParamBinding* ParamBindingTable::find(const std::string& external_name) const {
    for (const ParamBinding& binding : bindings_) {
        if (binding.external_name == external_name) {
            return &binding;
        }
    }
    return nullptr;
}

ParamLoadReport loadModelParams(const std::string& manifest_path,
                                const ParamBindingTable& bindings,
                                const ParamLoadOptions& options) {
    std::cout << "[param_loader] manifest_path=" << manifest_path << std::endl;
    const ParamManifest manifest = parseManifest(manifest_path);
    std::cout << "[param_loader] manifest parsed, entry_count=" << manifest.entry_count << std::endl;
    const fs::path manifest_dir = fs::absolute(fs::path(manifest_path)).parent_path();

    ParamLoadReport report;
    std::unordered_set<std::string> loaded_names;

    for (const auto& item : manifest.entries) {
        const ManifestParamEntry& entry = item.second;
        const ParamBinding* binding = bindings.find(entry.name);
        if (nullptr == binding) {
            if (options.allow_unused_manifest_entries) {
                ++report.skipped_manifest_entry_count;
                continue;
            }
            EXIT_ERROR("Unbound manifest param entry: %s", entry.name.c_str());
        }

        EXIT_ERROR_CHECK_EQ(nullptr, binding->data, "Binding target data is nullptr");
        validateOrInitTarget(*(binding->data), entry, options.init_empty_target_meta);
        loadParamFile(manifest_dir / entry.file, entry, *(binding->data));

        loaded_names.insert(entry.name);
        ++report.loaded_param_count;
        report.loaded_bytes += entry.bytes;
    }
    std::cout << "[param_loader] manifest entries processed" << std::endl;

    if (options.require_all_bindings) {
        for (const ParamBinding& binding : bindings.items()) {
            EXIT_ERROR_CHECK_EQ(loaded_names.find(binding.external_name), loaded_names.end(),
                "Bound param missing in manifest: %s", binding.external_name.c_str());
        }
    }

    return report;
}

} // namespace core
} // namespace Kernel
