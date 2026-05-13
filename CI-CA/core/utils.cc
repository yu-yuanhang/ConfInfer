#include <generic/utils.h>
#include <core/Layer.h>

using Kernel::core::DataType;

namespace Kernel {

unsigned int getCoreCount() {
    unsigned int cores = std::thread::hardware_concurrency();
    if (cores == 0) {
        // 返回 0 表示无法确定 可以选择抛异常或给个默认值
        std::cerr << "Warning: unable to determine hardware concurrency" << std::endl;
    }
    return cores;
}

void print_shape(const char *name, const core::DataShape_t &shape) {
    std::cout << name << " ndim=" << shape.ndim << " dims=[";
    for (uint32_t i = 0; i < shape.ndim; ++i) {
        std::cout << shape.dims[i];
        if (i + 1 < shape.ndim) {
            std::cout << ", ";
        }
    }
    std::cout << "] size=" << shape.size << std::endl;
}

bool is_zero_filled(core::Value_t &value) {
    core::Data_t &data = value.data;
    if (!data.ptr || data.shape.size == 0) {
        return false;
    }

    const std::size_t bytes = data.shape.size * data.getTypeSize();
    const auto *raw = static_cast<const unsigned char *>(data.ptr);
    for (std::size_t i = 0; i < bytes; ++i) {
        if (raw[i] != 0) {
            return false;
        }
    }
    return true;
}

bool print_zero_check(const char *name, core::Value_t &value) {
    const bool ok = is_zero_filled(value);
    std::cout << name << " zero_filled=" << (ok ? "true" : "false") << std::endl;
    return ok;
}

bool same_signature(const core::Layer &lhs, const core::Layer &rhs) {
    return lhs.opSignature() == rhs.opSignature();
}

bool same_params(const core::Layer &lhs, const core::Layer &rhs) {
    return lhs.params() == rhs.params();
}

bool print_layer_relation(const char *lhs_name,
                          const core::Layer &lhs,
                          const char *rhs_name,
                          const core::Layer &rhs) {
    const bool same_layer = (&lhs == &rhs);
    const bool shared_signature = same_signature(lhs, rhs);
    const bool shared_params = same_params(lhs, rhs);

    std::cout << lhs_name << "(id=" << lhs.id() << ")"
              << " vs "
              << rhs_name << "(id=" << rhs.id() << ")"
              << " same_layer=" << (same_layer ? "true" : "false")
              << " same_signature=" << (shared_signature ? "true" : "false")
              << " same_params=" << (shared_params ? "true" : "false")
              << std::endl;

    return (!same_layer) && shared_signature && shared_params;
}

void fill_random(void *data, DataType dtype, int n, uint32_t seed) {
    uint32_t st = seed;

    switch (dtype) {

    case DataType::FP32: {
        float* p = static_cast<float*>(data);
        for (int i = 0; i < n; ++i) {
            p[i] = frand01(st) - 0.5f; // [-0.5, 0.5)
        }
        break;
    }

    case DataType::FP16: {
        fp16_t* p = static_cast<fp16_t*>(data);
        for (int i = 0; i < n; ++i) {
            float v = frand01(st) - 0.5f;
            p[i] = float_to_fp16(v);
        }
        break;
    }

    case DataType::INT8: {
        int8_t* p = static_cast<int8_t*>(data);
        for (int i = 0; i < n; ++i) {
            // 映射到 [-127, 127]
            float v = frand01(st) - 0.5f;   // [-0.5, 0.5)
            int iv = static_cast<int>(v * 255.0f);
            if (iv > 127) iv = 127;
            if (iv < -127) iv = -127;
            p[i] = static_cast<int8_t>(iv);
        }
        break;
    }

    case DataType::INT32: {
        int32_t* p = static_cast<int32_t*>(data);
        for (int i = 0; i < n; ++i) {
            p[i] = static_cast<int32_t>(i);
        }
        break;
    }

    default:
        EXIT_ERROR("error Data_s.dtype");
    }
}

static void zero_buf(float* x, int n) {
  std::memset(x, 0, sizeof(float) * n);
}

static double max_abs_diff(const float* a, const float* b, int n) {
  double m = 0.0;
  for (int i = 0; i < n; i++) {
    double d = std::fabs((double)a[i] - (double)b[i]);
    if (d > m) m = d;
  }
  return m;
}

} // namespace end of Kernel 
