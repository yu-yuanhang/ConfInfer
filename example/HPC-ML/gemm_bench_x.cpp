#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <iostream>

/*
用 OpenBLAS
    sudo apt install -y libopenblas-dev
编译 (推荐开 AVX2+FMA)
g++ -O3 -mavx2 -mfma gemm_bench_x.cpp -o gemm_bench_x -lopenblas

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
taskset -c 0 ./gemm_bench_x 1024 5 2

 */


// x86 SIMD 头文件 (包含 SSE/AVX/FMA)
#include <immintrin.h>

// 引入 CBLAS 接口（OpenBLAS / MKL / BLIS 等通常都提供）
extern "C" {
#include <cblas.h>
}

static inline double now_seconds() {
  using clock = std::chrono::high_resolution_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}
static void* aligned_malloc_64(size_t bytes) {
  void* p = nullptr;
  if (posix_memalign(&p, 64, bytes) != 0) return nullptr;
  return p;
}
static void aligned_free_64(void* p) {
  free(p);
}
static inline float frand01(uint32_t& state) {
  state = state * 1664525u + 1013904223u;
  uint32_t x = (state >> 8) | 0x3F800000u; // 构造 1.xxx 的浮点数
  float f;
  std::memcpy(&f, &x, sizeof(float));
  return f - 1.0f;
}
static void fill_random(float* x, int n, uint32_t seed) {
  uint32_t st = seed;
  for (int i = 0; i < n; i++) x[i] = frand01(st) - 0.5f;
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
// 1）朴素 GEMM：三重循环
// 计算：C = A * B
// 数据布局：row-major (行主序) 
// =====================================================
static void gemm_naive(const float* A, const float* B, float* C, int N) {
  for (int i = 0; i < N; i++) {
    const float* arow = A + i * N;
    for (int j = 0; j < N; j++) {
      float sum = 0.0f;
      for (int k = 0; k < N; k++) {
        sum += arow[k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

// =====================================================
// 2）单核 SIMD GEMM：优先 AVX2(1x8)，否则 SSE(1x4)
// 计算：C = A * B
//
// 说明：这是一个非常简化的 micro-kernel: 
// 每次计算 C[i, j..j+W-1]
// - AVX2: W=8
// - SSE : W=4
// =====================================================
static void gemm_simd(const float* A, const float* B, float* C, int N) {

#if defined(__AVX2__)
  // -----------------------------
  // AVX2 版本：一次处理 8 个 float
  // -----------------------------
  const int W = 8;

  for (int i = 0; i < N; i++) {
    const float* arow = A + i * N;

    int j = 0;
    for (; j + (W - 1) < N; j += W) {
      __m256 acc = _mm256_setzero_ps();

      for (int k = 0; k < N; k++) {
        // 读取 B[k, j..j+7]
        __m256 b = _mm256_loadu_ps(B + k * N + j);
        // 广播 A[i,k]
        __m256 a = _mm256_set1_ps(arow[k]);

#if defined(__FMA__)
        // acc += a*b（FMA 指令）
        acc = _mm256_fmadd_ps(a, b, acc);
#else
        acc = _mm256_add_ps(acc, _mm256_mul_ps(a, b));
#endif
      }

      // 写回 C[i, j..j+7]
      _mm256_storeu_ps(C + i * N + j, acc);
    }

    // 尾部不足 8 的部分用标量
    for (; j < N; j++) {
      float sum = 0.0f;
      for (int k = 0; k < N; k++) sum += arow[k] * B[k * N + j];
      C[i * N + j] = sum;
    }
  }

#else
  // -----------------------------
  // SSE 版本：一次处理 4 个 float
  // -----------------------------
  const int W = 4;

  for (int i = 0; i < N; i++) {
    const float* arow = A + i * N;

    int j = 0;
    for (; j + (W - 1) < N; j += W) {
      __m128 acc = _mm_setzero_ps();

      for (int k = 0; k < N; k++) {
        // 读取 B[k, j..j+3]
        __m128 b = _mm_loadu_ps(B + k * N + j);
        // 广播 A[i,k]
        __m128 a = _mm_set1_ps(arow[k]);
        // acc += a*b
        acc = _mm_add_ps(acc, _mm_mul_ps(a, b));
      }

      // 写回 C[i, j..j+3]
      _mm_storeu_ps(C + i * N + j, acc);
    }

    // 尾部不足 4 的部分用标量
    for (; j < N; j++) {
      float sum = 0.0f;
      for (int k = 0; k < N; k++) sum += arow[k] * B[k * N + j];
      C[i * N + j] = sum;
    }
  }
#endif
}

// =====================================================
// 3）调用专用加速库 (BLAS) 实现 GEMM (单核)
// 计算：C = A * B
// OPENBLAS_NUM_THREADS=1 / OMP_NUM_THREADS=1 / MKL_NUM_THREADS=1
// =====================================================
static void gemm_blas(const float* A, const float* B, float* C, int N) {
  // C := 1.0 * A * B + 0.0 * C
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              N, N, N,
              1.0f,
              A, N,
              B, N,
              0.0f,
              C, N);
}

using gemm_fn = void(*)(const float*, const float*, float*, int);

static double bench_one(gemm_fn fn,
                        const float* A, const float* B, float* C,
                        int N, int iters, int warmup) {
  // 预热：让 cache / CPU 频率更稳定
  for (int w = 0; w < warmup; w++) {
    fn(A, B, C, N);
  }

  // 正式计时：跑多次取最小值，减少系统干扰
  double best = 1e100;
  for (int t = 0; t < iters; t++) {
    double t0 = now_seconds();
    fn(A, B, C, N);
    double t1 = now_seconds();
    best = std::min(best, (t1 - t0));
  }
  return best;
}

// 计算 GFLOPS（浮点运算吞吐）
// GEMM 的 FLOPs = 2*N^3
static double gflops(double sec, int N) {
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
  std::printf("如果用 MKL: \n");
  std::printf("  taskset -c 0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 %s 1024 5 2\n", prog);
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

  std::printf("========================================\n");
  std::printf("x86 GEMM 性能对比实验（目标：单核）\n");
  std::printf("N=%d, iters=%d, warmup=%d\n", N, iters, warmup);

#if defined(__AVX2__)
  std::printf("SIMD 路径: AVX2\n");
#else
  std::printf("SIMD 路径: SSE (未启用 AVX2 编译) \n");
#endif

#if defined(__FMA__)
  std::printf("FMA 支持: 是\n");
#else
  std::printf("FMA 支持: 否\n");
#endif

  std::printf("请使用 taskset 绑定单核 并设置 BLAS 线程=1\n");
  std::printf("========================================\n\n");

  // 分配矩阵内存（A/B/C）
  size_t bytes = (size_t)N * (size_t)N * sizeof(float);

  float* A  = (float*)aligned_malloc_64(bytes);
  float* B  = (float*)aligned_malloc_64(bytes);
  float* C1 = (float*)aligned_malloc_64(bytes); // naive 输出
  float* C2 = (float*)aligned_malloc_64(bytes); // simd 输出
  float* C3 = (float*)aligned_malloc_64(bytes); // blas 输出（参考）

  if (!A || !B || !C1 || !C2 || !C3) {
    std::fprintf(stderr, "内存申请失败\n");
    aligned_free_64(A);
    aligned_free_64(B);
    aligned_free_64(C1);
    aligned_free_64(C2);
    aligned_free_64(C3);
    return 1;
  }

  // 初始化随机矩阵
  fill_random(A, N * N, 123);
  fill_random(B, N * N, 456);

  // 先用 BLAS 跑一次，作为 "参考正确结果"
  zero_buf(C3, N * N);
  gemm_blas(A, B, C3, N);

  // 1）朴素版本 benchmark
  zero_buf(C1, N * N);
  double t_naive = bench_one(gemm_naive, A, B, C1, N, iters, warmup);
  double diff_naive = max_abs_diff(C1, C3, N * N);

  // 2）SIMD 版本 benchmark
  zero_buf(C2, N * N);
  double t_simd = bench_one(gemm_simd, A, B, C2, N, iters, warmup);
  double diff_simd = max_abs_diff(C2, C3, N * N);

  // 3）BLAS 版本 benchmark（重新清零再测）
  zero_buf(C3, N * N);
  double t_blas = bench_one(gemm_blas, A, B, C3, N, iters, warmup);

  // 输出结果
  std::printf("结果（取 %d 次计时中的最短时间）：\n", iters);

  std::printf("  1) naive 朴素三重循环:\n");
  std::printf("     时间 = %8.4f ms | 吞吐 = %8.2f GFLOPS | 与 BLAS 最大误差 = %.6g\n",
              t_naive * 1e3, gflops(t_naive, N), diff_naive);

  std::printf("  2) simd  单核 + SIMD(AVX2/SSE):\n");
  std::printf("     时间 = %8.4f ms | 吞吐 = %8.2f GFLOPS | 与 BLAS 最大误差 = %.6g\n",
              t_simd * 1e3, gflops(t_simd, N), diff_simd);

  std::printf("  3) blas  专用加速库 cblas_sgemm:\n");
  std::printf("     时间 = %8.4f ms | 吞吐 = %8.2f GFLOPS | (参考结果)\n",
              t_blas * 1e3, gflops(t_blas, N));

  // 释放内存
  aligned_free_64(A);
  aligned_free_64(B);
  aligned_free_64(C1);
  aligned_free_64(C2);
  aligned_free_64(C3);

  return 0;
}
