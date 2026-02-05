#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>

/*

********************************************
方式 A：OpenBLAS
********************************************
sudo apt install -y libopenblas-dev
g++ -O3 -march=native gemm_bench_a.cpp -o gemm_bench_a -lopenblas
运行（强制单核）：
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
taskset -c 0 ./gemm_bench_a 1024 5


********************************************
ARMPL（如果你用的是 Arm Performance Libraries）
********************************************
g++ -O3 -march=native gemm_bench_a.cpp -o gemm_bench -larmpl
 */

#if defined(__aarch64__) || defined(__ARM_NEON)
  #include <arm_neon.h>
  #define HAS_NEON 1
#else
  #define HAS_NEON 0
#endif

// 引入 CBLAS 接口 (OpenBLAS / ARMPL / BLIS 等通常都提供)
extern "C" {
#include <cblas.h>
}

static inline double now_seconds() {
  using clock = std::chrono::high_resolution_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

// 申请 64 字节对齐内存 (对 SIMD / cache 更友好)
static void* aligned_malloc_64(size_t bytes) {
#if defined(_MSC_VER)
  return _aligned_malloc(bytes, 64);
#else
  void* p = nullptr;
  if (posix_memalign(&p, 64, bytes) != 0) return nullptr;
  return p;
#endif
}
static void aligned_free_64(void* p) {
#if defined(_MSC_VER)
  _aligned_free(p);
#else
  free(p);
#endif
}

static inline float frand01(uint32_t& state) {
  // simple LCG
  state = state * 1664525u + 1013904223u;
  uint32_t x = (state >> 8) | 0x3F800000u; // 1.xxx
  float f;
  std::memcpy(&f, &x, sizeof(float));
  return f - 1.0f; // [0,1)
}

static void fill_random(float* x, int n, uint32_t seed) {
  uint32_t st = seed;
  for (int i = 0; i < n; i++) x[i] = frand01(st) - 0.5f; // [-0.5, 0.5)
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

// =====================================================
// 1) Naive GEMM (no SIMD)
// 数据布局：row-major (行主序)
// A: [N,N], B: [N,N], C: [N,N]
// =====================================================
static void gemm_naive(const float* A, const float* B, float* C, int N) {
  // i-j-k ordering (simple, but not cache-friendly)
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0.0f;
      const float* arow = A + i * N;
      for (int k = 0; k < N; k++) {
        sum += arow[k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

// =====================================================
// 2）NEON SIMD GEMM (单核 + SIMD)
// 计算：C = A * B
//
// 这里实现一个简单的 1x4 micro-kernel:
// 每次计算 C[i, j..j+3] 四个输出
//
// 注意 这个实现没有 packing / 多级 blocking
// 仍然比不过真正的 BLAS 但应该明显快于 naive
// =====================================================
static void gemm_neon_1x4(const float* A, const float* B, float* C, int N) {
#if HAS_NEON
  const int J_STEP = 4;

  for (int i = 0; i < N; i++) {
    const float* arow = A + i * N;

    int j = 0;
    for (; j + (J_STEP - 1) < N; j += J_STEP) {
      float32x4_t acc = vdupq_n_f32(0.0f);

      // k loop
      for (int k = 0; k < N; k++) {
        // load B[k, j..j+3]
        float32x4_t b = vld1q_f32(B + k * N + j);
        // broadcast A[i,k]
        float32x4_t a = vdupq_n_f32(arow[k]);
        // FMA: acc += a * b
#if defined(__aarch64__)
        acc = vfmaq_f32(acc, a, b);
#else
        acc = vmlaq_f32(acc, a, b);
#endif
      }

      vst1q_f32(C + i * N + j, acc);
    }

    // tail (scalar)
    for (; j < N; j++) {
      float sum = 0.0f;
      for (int k = 0; k < N; k++) {
        sum += arow[k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
#else
  // 回退 to naive if no NEON
  gemm_naive(A, B, C, N);
#endif
}

// =====================================================
// 3) 调用专用加速库 (BLAS) 实现 GEMM（单核）
// 计算：C = A * B
//
// 实际是否“单核”取决于你是否把 BLAS 线程数设为 1
// 用法 OPENBLAS_NUM_THREADS=1 / OMP_NUM_THREADS=1
// =====================================================
static void gemm_blas(const float* A, const float* B, float* C, int N) {
  // C := 1.0*A*B + 0.0*C
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              N, N, N,
              1.0f,
              A, N,
              B, N,
              0.0f,
              C, N);
}

using gemm_fn = void(*)(const float*, const float*, float*, int);
static double bench_one(gemm_fn fn, const float* A, const float* B, float* C,
                        int N, int iters, int warmup) {
  // 预热 : 让 cache / CPU 频率更稳定
  for (int w = 0; w < warmup; w++) {
    fn(A, B, C, N);
  }

  double best = 1e100;
  for (int t = 0; t < iters; t++) {
    double t0 = now_seconds();
    fn(A, B, C, N);
    double t1 = now_seconds();
    best = std::min(best, (t1 - t0));
  }
  return best;
}

static double gflops(double sec, int N) {
  // 2*N^3 FLOPs
  double flops = 2.0 * (double)N * (double)N * (double)N;
  return flops / sec / 1e9;
}

// 打印使用方法
static void usage(const char* prog) {
  std::printf("用法: %s [N] [iters] [warmup]\n", prog);
  std::printf("  N      : 矩阵大小 (默认 512) \n");
  std::printf("  iters  : 正式计时迭代次数 (默认 5) \n");
  std::printf("  warmup : 预热次数 (默认 2) \n\n");
  std::printf("推荐单核运行方式：\n");
  std::printf("  taskset -c 0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 %s 1024 5 2\n", prog);
}

int main(int argc, char** argv) {
  int N = 512;
  int iters = 5;
  int warmup = 2;

  if (argc >= 2) N = std::atoi(argv[1]);
  if (argc >= 3) iters = std::atoi(argv[2]);
  if (argc >= 4) warmup = std::atoi(argv[3]);

  if (N <= 0 || iters <= 0 || warmup < 0) {
    usage(argv[0]);
    return 1;
  }

  std::printf("GEMM Benchmark (single-core)\n");
  std::printf("N=%d, iters=%d, warmup=%d\n", N, iters, warmup);
  std::printf("NEON available: %s\n", HAS_NEON ? "YES" : "NO");
  std::printf("提示: 请使用 taskset 绑定单核，并设置 BLAS 线程=1\n\n");

  size_t bytes = (size_t)N * (size_t)N * sizeof(float);

  float* A = (float*)aligned_malloc_64(bytes);
  float* B = (float*)aligned_malloc_64(bytes);
  float* C1 = (float*)aligned_malloc_64(bytes);
  float* C2 = (float*)aligned_malloc_64(bytes);
  float* C3 = (float*)aligned_malloc_64(bytes);

  if (!A || !B || !C1 || !C2 || !C3) {
    std::fprintf(stderr, "Allocation failed\n");
    aligned_free_64(A);
    aligned_free_64(B);
    aligned_free_64(C1);
    aligned_free_64(C2);
    aligned_free_64(C3);
    return 1;
  }

  fill_random(A, N * N, 123);
  fill_random(B, N * N, 456);

  // 先用 BLAS 跑一次，作为 "参考正确结果"
  zero_buf(C3, N * N);
  gemm_blas(A, B, C3, N);

  // 1) naive
  zero_buf(C1, N * N);
  double t_naive = bench_one(gemm_naive, A, B, C1, N, iters, warmup);
  double diff_naive = max_abs_diff(C1, C3, N * N);

  // 2) neon
  zero_buf(C2, N * N);
  double t_neon = bench_one(gemm_neon_1x4, A, B, C2, N, iters, warmup);
  double diff_neon = max_abs_diff(C2, C3, N * N);

  // 3) blas (timed)
  zero_buf(C3, N * N);
  double t_blas = bench_one(gemm_blas, A, B, C3, N, iters, warmup);

  std::printf("Results (best of %d iters):\n", iters);
  std::printf("  naive : %8.4f ms | %8.2f GFLOPS | max_abs_diff vs BLAS = %.6g\n",
              t_naive * 1e3, gflops(t_naive, N), diff_naive);

  std::printf("  neon  : %8.4f ms | %8.2f GFLOPS | max_abs_diff vs BLAS = %.6g\n",
              t_neon * 1e3, gflops(t_neon, N), diff_neon);

  std::printf("  blas  : %8.4f ms | %8.2f GFLOPS | (reference)\n",
              t_blas * 1e3, gflops(t_blas, N));

  aligned_free_64(A);
  aligned_free_64(B);
  aligned_free_64(C1);
  aligned_free_64(C2);
  aligned_free_64(C3);

  return 0;
}
