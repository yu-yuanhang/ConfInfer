#ifndef __UTILS_H_CA__
#define __UTILS_H_CA__

#include <iostream>
#include <cstring> // for std::memcpy, std::memset 
#include <cmath> // for std::fabs
#include <core/Param.h>
#include <thread>

namespace Kernel {

template <typename T, size_t N>
void initArray(T (&arr)[N], const T& value) {
    for (size_t i = 0; i < N; ++i) {
        arr[i] = value;
    }
}

static inline uint32_t time_seed() {
    using namespace std::chrono;
    uint64_t t =
        duration_cast<nanoseconds>(
            high_resolution_clock::now().time_since_epoch()
        ).count();

    // 混一下高低位，避免低熵
    return static_cast<uint32_t>(t ^ (t >> 32));
}
static inline fp16_t float_to_fp16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));

    uint32_t sign = (x >> 16) & 0x8000;
    uint32_t mantissa = x & 0x007FFFFF;
    int exp = ((x >> 23) & 0xFF) - 127 + 15;

    if (exp <= 0) return (fp16_t)sign;
    if (exp >= 31) return (fp16_t)(sign | 0x7C00);

    return (fp16_t)(sign | (exp << 10) | (mantissa >> 13));
}
static inline float frand01(uint32_t& state) {
  // simple LCG
  state = state * 1664525u + 1013904223u;
  uint32_t x = (state >> 8) | 0x3F800000u; // 1.xxx
  float f;
  std::memcpy(&f, &x, sizeof(float));
  return f - 1.0f; // [0,1)
}

// 下面的这些功能函数后续需要 多态化实现
void fill_random(void *data, core::DataType dtype, int n, uint32_t seed);
static void zero_buf(float* x, int n);
static double max_abs_diff(const float* a, const float* b, int n);
unsigned int getCoreCount();

#define TIMESEED (time_seed())

} // namespace end of Kernel 
#endif