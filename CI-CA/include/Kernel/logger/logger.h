#ifndef __LOGGER_H__
#define __LOGGER_H__

#ifndef LOGCPP_EXT
#define LOGCPP_EXT 0
#endif

#include <generic/Singleton.h>
#include <string>

#if LOGCPP_EXT
#include <log4cpp/Category.hh>
#endif // LOGCPP_EXT

using std::string;

namespace Kernel {
namespace SingleLog4 {

#if LOGCPP_EXT
#define PATH_TO_WDLOG "/home/yyh/2.Programs/2.workplace/optee_qemuv8/optee_doc-3.22.0/CI/ConfInfer/Kernel/Logger/logs/wd.log"

class logger {
public:
    enum Priority {
        // FATAL = 300,
        ERROR = 300,
        WARN,
        INFO,
        DEBUG
    };

	template <class... Args>
	void error(const char * msg, Args... args)
	{
		_cat.error(msg, args...);
	}

	template <class... Args>
	void warn(const char * msg, Args... args)
	{
		_cat.warn(msg, args...);
	}

	template <class... Args>
	void info(const char * msg, Args... args)
	{
		_cat.info(msg, args...);
	}

	template <class... Args>
	void debug(const char * msg, Args... args)
	{
		_cat.debug(msg, args...);
	}

	void error(const char * msg);
	void warn(const char * msg);
	void info(const char * msg);
	void debug(const char * msg);

// ===========================================================

    void setPriority(Priority pri);
public:
    logger();
    ~logger();
private:
    // static logger * _pInstance;
    log4cpp::Category & _cat;
	// static AutoRelease _ar;
};

//##__VA_ARGS__ 宏前面加上##的作用在于，当可变参数的个数为0时，
//这里的##起到把前面多余的","去掉的作用,否则会编译出错
/* #define LogError(msg) Mylogger::getInstance()->error(prefix(msg)) */
// 在宏定义中 ... 允许宏接受任意数量的参数
// #define LogError(msg, ...) logger::getInstance()->error(prefix(msg), ##__VA_ARGS__)
// #define LogWarn(msg, ...) logger::getInstance()->warn(prefix(msg), ##__VA_ARGS__)
// #define LogInfo(msg, ...) logger::getInstance()->info(prefix(msg), ##__VA_ARGS__)
// #define LogDebug(msg, ...) logger::getInstance()->debug(prefix(msg), ##__VA_ARGS__)

#define prefix_log(msg) string("[")\
	.append(__FILE__).append(":")\
	.append(__FUNCTION__).append(":")\
	.append(std::to_string(__LINE__)).append("] ")\
	.append(msg).c_str()

#define LogError(msg, ...) Singleton<SingleLog4::logger>::getInstance()->error(prefix_log(msg), ##__VA_ARGS__)
#define LogWarn(msg, ...) Singleton<SingleLog4::logger>::getInstance()->warn(prefix_log(msg), ##__VA_ARGS__)
#define LogInfo(msg, ...) Singleton<SingleLog4::logger>::getInstance()->info(prefix_log(msg), ##__VA_ARGS__)
#define LogDebug(msg, ...) Singleton<SingleLog4::logger>::getInstance()->debug(prefix_log(msg), ##__VA_ARGS__)
#else
#define prefix_log(msg)
#define LogError(msg, ...) 
#define LogWarn(msg, ...) 
#define LogInfo(msg, ...) 
#define LogDebug(msg, ...) 
#endif // LOGCPP_EXT

} // end of SingleLog4

// 1. 标准宏
// 这些宏通常是标准库或编译器提供的 供开发者使用来获取环境、版本、特性等信息 

// __cplusplus: 表示支持的 C++ 标准的版本号 例如: 
// 199711L (C++98)
// 201103L (C++11)
// 201402L (C++14)
// 201703L (C++17)
// 202002L (C++20)
// __FILE__: 表示当前文件的名称 (字符串形式) 
// __LINE__: 表示当前的行号 (整数) 
// __DATE__: 编译时的日期 (字符串形式) 
// __TIME__: 编译时的时间 (字符串形式) 
// __STDC__: 如果定义 表示编译器支持标准 C 
// 2. 与平台或编译器相关的宏
// 这些宏可以帮助开发者判断代码正在运行的环境或使用的编译器 以便编写跨平台代码 

// __GNUC__: 表示 GCC 的主版本号 
// __clang__: 如果定义 表示正在使用 Clang 编译器 
// _MSC_VER: 表示 Microsoft 编译器的版本号 
// __unix__、__APPLE__、_WIN32: 这些宏表示目标平台类型 分别表示 Unix 系统、苹果系统和 Windows 系统 
// 3. 特性检测宏
// 这些宏用于检查编译器是否支持特定的特性: 

// __has_include(<header>): 用于检查编译器是否支持特定的头文件 
// __has_cpp_attribute(attribute): 用于检查是否支持某个 C++ 属性 
// __has_builtin(builtin): 用于检查是否支持某个内建函数 
// 4. 预定义的处理器架构宏
// __x86_64__: 如果定义 表示目标平台为 64 位 x86 架构 
// __i386__: 如果定义 表示目标平台为 32 位 x86 架构 
// __arm__: 如果定义 表示目标平台为 ARM 架构 
// 5. 标准库相关的宏
// 这些宏与 C++ 标准库和标准 C 库有关 

// BUFSIZ: 在 C 标准库中定义的缓冲区大小 通常用于文件 I/O 
// NULL: 空指针的宏定义 通常在 C/C++ 中定义为 0 或 nullptr 
// EOF: 表示文件结束的宏 通常用于文件流操作 
// 6. 字节和类型相关的宏
// CHAR_BIT: 表示 char 类型的位数 通常为 8 位 
// SCHAR_MIN、SCHAR_MAX: 表示 signed char 的最小值和最大值 
// INT_MIN、INT_MAX: 表示 int 类型的最小值和最大值 
// SIZE_MAX: 表示 size_t 类型的最大值 
// 7. 线程相关的宏
// 如果你的编译环境支持多线程 以下宏可能会定义: 

// _REENTRANT: 表示可重入代码 多用于 POSIX 线程 (pthreads) 
// _OPENMP: 表示支持 OpenMP 并行编程 

// #if LOGCPP_EXT
#if true
#define ARGC_CHECK(argc, num) {\
    if((argc) != (num)) {\
        LogError("args error!");\
        return -1;\
    }\
}
#define ERROR_CHECK(ret, num, fmt, ...) {\
    if((ret) == (num)) {\
        LogError(fmt, ##__VA_ARGS__);\
        return -1;\
    }\
}
#define RET_ERROR_CHECK(ret, num, fmt, errorNum, ...) {\
    if((ret) == (num)) {\
        LogError(fmt, ##__VA_ARGS__);\
        return errorNum;\
    }\
}
#define EXIT_ERROR_CHECK_EQ(ret, num, fmt, ...) {\
    if((ret) == (num)) {\
        LogError(fmt, ##__VA_ARGS__);\
        exit(EXIT_FAILURE);\
    }\
}
#define EXIT_ERROR_CHECK_NE(ret, num, fmt, ...) {\
    if((ret) != (num)) {\
        LogError(fmt, ##__VA_ARGS__);\
        exit(EXIT_FAILURE);\
    }\
}
// 因为涉及到子线程和 TEE 资源的占用问题 所以这里的 exit 还需要进一步封装
#define EXIT_ERROR(fmt, ...) {\
    LogError(fmt, ##__VA_ARGS__);\
    exit(EXIT_FAILURE);\
}
#else
#define ARGC_CHECK(argc, num)
#define ERROR_CHECK(ret, num, fmt, ...)
#define RET_ERROR_CHECK(ret, num, fmt, errorNum, ...)
#define EXIT_ERROR_CHECK_EQ(ret, num, fmt, ...)
#define EXIT_ERROR_CHECK_NE(ret, num, fmt, ...)
#define EXIT_ERROR(fmt, ...)
#endif


} // namespace end of Kernel  

#endif

