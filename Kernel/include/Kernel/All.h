#ifndef __ALL_H_CA__
#define __ALL_H_CA__

#include <iostream>
#include <memory>
#include <logger/logger.h>
#include <cstddef>
#include <cstdlib>  // 包含 exit 函数的头文件
#include <random> 
#include <ctime>   // 包含时间函数的头文件
#include <chrono>
#include <unistd.h>

#include <cstdint>

#include <sys/time.h>   // struct timeval used in gettimeofday()
#include <generic/templateList.h>

#include <atomic>
#include <unordered_set>
#include <unordered_map>

namespace Kernel {

using std::cout;
using std::endl;
using std::vector;

// pytorch 中定义有 float32 和 float64 
// 目前只考虑默认的 float 为 32 位
// 项目只支持编译前预处理指定 float 类型
typedef float float32;
typedef uint16_t fp16_t;
#define FLOAT_SIZE (4)

#define INVALID_VALUE_U (0)
#define INVALID_VALUE (-1)

// 这里 int64_t 主要是针对 pytorch 的接口
// 本质上是没必要 而且对接 TA 那里还是要进行结构转换
// 索性把 结构转换移到 上层接口
#if INTPTR_MAX == INT64_MAX
    typedef int32_t INT;   // 64 位系统
#elif INTPTR_MAX == INT32_MAX
    typedef int32_t INT;   // 32 位系统
#else
    #error "Unsupported platform"
#endif
typedef float32     FLOAT;
typedef uint32_t    UINT;
typedef int8_t      BOOL;
typedef int8_t      FLAG;

#define INVALID_UINT_MAX   UINT32_MAX

#define MAX_CORES_NUM (16)    // 最大支持核心数 对应可能的 Net Num

} // namespace end of Kernel 
#endif
