#include <generic/utils.h>

using Kernel::core::DataType;

namespace Kernel {

namespace sysutil {

unsigned int getCoreCount() {
    unsigned int cores = std::thread::hardware_concurrency();
    if (cores == 0) {
        // 返回 0 表示无法确定 可以选择抛异常或给个默认值
        std::cerr << "Warning: unable to determine hardware concurrency" << std::endl;
    }
    return cores;
}

} // namespace sysutil


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