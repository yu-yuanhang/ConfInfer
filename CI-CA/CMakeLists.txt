cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# 因为是交叉编译 这里没有办法直接通过系统环境来决定是否配置 arrch64 版本
# 这里只能用 build_aarch 文件名作为约定
get_filename_component(CONFINFER_BUILD_DIR_NAME "${CMAKE_BINARY_DIR}" NAME)
get_filename_component(CONFINFER_REPO_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(CONFINFER_DEFAULT_BUILDROOT_AARCH64 OFF)
if(CONFINFER_BUILD_DIR_NAME STREQUAL "build_arrch"
   OR CONFINFER_BUILD_DIR_NAME STREQUAL "build_aarch"
   OR CONFINFER_BUILD_DIR_NAME STREQUAL "build_aarch64")
    set(CONFINFER_DEFAULT_BUILDROOT_AARCH64 ON)
endif()

option(CONFINFER_USE_BUILDROOT_AARCH64
    "Configure Buildroot aarch64 cross-compilation defaults"
    ${CONFINFER_DEFAULT_BUILDROOT_AARCH64})

if(CONFINFER_USE_BUILDROOT_AARCH64)
    # 经典三件套 HOST_DIR / SYSROOT / BIN_DIR
    set(CONFINFER_BUILDROOT_HOST_DIR
        "${CONFINFER_REPO_ROOT}/out-br/host"
        CACHE PATH "Buildroot host directory for aarch64 cross build")
    set(CONFINFER_BUILDROOT_SYSROOT
        "${CONFINFER_BUILDROOT_HOST_DIR}/aarch64-buildroot-linux-gnu/sysroot"
        CACHE PATH "Buildroot sysroot for aarch64 cross build")
    set(CONFINFER_BUILDROOT_BIN_DIR
        "${CONFINFER_BUILDROOT_HOST_DIR}/bin"
        CACHE PATH "Buildroot toolchain bin directory for aarch64 cross build")

    # 目标系统指定成 Linux
    # 影响很多平台判断逻辑 后续查找系统库、系统头文件、平台特性时用哪套规则
    set(CMAKE_SYSTEM_NAME Linux CACHE STRING "Target system name" FORCE)
    # 意思是把目标 CPU 架构指定成 aarch64
    set(CMAKE_SYSTEM_PROCESSOR aarch64 CACHE STRING "Target system processor" FORCE)

    if(NOT DEFINED CMAKE_C_COMPILER)
        set(CMAKE_C_COMPILER
            "${CONFINFER_BUILDROOT_BIN_DIR}/aarch64-linux-gnu-gcc"
            CACHE FILEPATH "C compiler for Buildroot aarch64 cross build")
    endif()
    if(NOT DEFINED CMAKE_CXX_COMPILER)
        set(CMAKE_CXX_COMPILER
            "${CONFINFER_BUILDROOT_BIN_DIR}/aarch64-linux-gnu-g++"
            CACHE FILEPATH "CXX compiler for Buildroot aarch64 cross build")
    endif()
    if(NOT DEFINED CMAKE_SYSROOT)
        set(CMAKE_SYSROOT
            "${CONFINFER_BUILDROOT_SYSROOT}"
            CACHE PATH "Sysroot for Buildroot aarch64 cross build")
    endif()
    if(NOT DEFINED CMAKE_FIND_ROOT_PATH)
        set(CMAKE_FIND_ROOT_PATH
            "${CONFINFER_BUILDROOT_SYSROOT}"
            CACHE PATH "Find root path for Buildroot aarch64 cross build")
    endif()
endif()

project(ConfInfer LANGUAGES C CXX)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)

###########################################################################
# cmake -DLOGCPP_EXT=1 -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..
# cmake -DLOGCPP_EXT=1 -DRELEASE=0 -DENABLE_TEE_BRIDGE=1 ..
# cmake -DLOGCPP_EXT=1 -DRELEASE=0 -DENABLE_TEE_BRIDGE=1 -DTRUSTSPAN=1 ..

if(NOT DEFINED LOGCPP_EXT)
    set(LOGCPP_EXT 0 CACHE STRING "value of LOGCPP_EXT")
endif()
if(NOT DEFINED RELEASE)
    set(RELEASE 0 CACHE STRING "value of RELEASE")
endif()
if(NOT DEFINED ENABLE_TEE_BRIDGE)
    set(ENABLE_TEE_BRIDGE 0 CACHE STRING "value of ENABLE_TEE_BRIDGE")
endif()
if(NOT DEFINED TRUSTSPAN)
    set(TRUSTSPAN 0 CACHE STRING "value of TRUSTSPAN")
endif()
###########################################################################
message(STATUS "CMake source directory is: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "ConfInfer repo root is: ${CONFINFER_REPO_ROOT}")
message(STATUS "ConfInfer build dir name is: ${CONFINFER_BUILD_DIR_NAME}")
message(STATUS "CONFINFER_USE_BUILDROOT_AARCH64 = ${CONFINFER_USE_BUILDROOT_AARCH64}")
message(STATUS "CMAKE_CROSSCOMPILING = ${CMAKE_CROSSCOMPILING}")
message(STATUS "CMake system processor is: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "CMAKE_C_COMPILER = ${CMAKE_C_COMPILER}")
message(STATUS "CMAKE_CXX_COMPILER = ${CMAKE_CXX_COMPILER}")
message(STATUS "CMAKE_SYSROOT = ${CMAKE_SYSROOT}")
message(STATUS "TRUSTSPAN = ${TRUSTSPAN}")

set(CONFINFER_IS_AARCH64_BUILD FALSE)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|AARCH64|arm64|ARM64)$")
    set(CONFINFER_IS_AARCH64_BUILD TRUE)
endif()

set(CONFINFER_IS_AARCH64_CROSS_BUILD FALSE)
if(CONFINFER_IS_AARCH64_BUILD AND CMAKE_CROSSCOMPILING)
    set(CONFINFER_IS_AARCH64_CROSS_BUILD TRUE)
endif()

if(NOT DEFINED CONFINFER_DEPLOY_LIB_DIR)
    if(CONFINFER_IS_AARCH64_CROSS_BUILD)
        set(CONFINFER_DEPLOY_LIB_DIR
            "${CMAKE_CURRENT_SOURCE_DIR}/../../../out-br/target/usr/lib64"
            CACHE PATH "Optional deploy directory for the generated libConfInfer.so")
    else()
        set(CONFINFER_DEPLOY_LIB_DIR
            ""
            CACHE PATH "Optional deploy directory for the generated libConfInfer.so")
    endif()
endif()

file(GLOB MAIN_SOURCES_CPP CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/core/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/ops/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/backend/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu_ree/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu_ree/math/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu_ree_ref/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/Logger/*.cc"
)

file(GLOB MAIN_SOURCES_C CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu_ree/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu_ree/math/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu_ree_ref/*.c")

if(ENABLE_TEE_BRIDGE EQUAL 1)
    file(GLOB TEE_BRIDGE_SOURCES_CPP CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/bridges/*.cc")
    list(APPEND MAIN_SOURCES_CPP ${TEE_BRIDGE_SOURCES_CPP})
    list(APPEND MAIN_SOURCES_C
        "${CMAKE_CURRENT_SOURCE_DIR}/../CI-TA/host/confinfer_host.c")
endif()

message(STATUS "MAIN_SOURCES_CPP = ${MAIN_SOURCES_CPP}")
message(STATUS "MAIN_SOURCES_C = ${MAIN_SOURCES_C}")

add_library(${PROJECT_NAME} SHARED 
            ${MAIN_SOURCES_CPP}
            ${MAIN_SOURCES_C})

target_include_directories(${PROJECT_NAME}
                PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
                PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/Kernel
                PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/Kernel/core/ops
                PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../CI-TA/ta/include
                PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../log4cpp/build_aarch64/include
                PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../log4cpp/build_aarch64/include/threading
                PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../log4cpp/build/include
                PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../log4cpp/build/include/threading)

if(ENABLE_TEE_BRIDGE EQUAL 1)
    target_include_directories(${PROJECT_NAME}
                    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/Kernel/bridges
                    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../CI-TA/host/include
                    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../CI-TA/ta/include)
    if(CONFINFER_USE_BUILDROOT_AARCH64 AND CMAKE_SYSROOT)
        target_include_directories(${PROJECT_NAME}
                        PUBLIC ${CMAKE_SYSROOT}/usr/include)
    endif()
endif()

# 设置编译选项（跨平台兼容）
target_compile_options(${PROJECT_NAME} PRIVATE
    # 对所有源文件添加 -fPIC（位置无关代码）
    $<$<COMPILE_LANGUAGE:CXX>:-fPIC>
    $<$<COMPILE_LANGUAGE:C>:-fPIC>
)

target_link_libraries(${PROJECT_NAME} PRIVATE 
    ${CMAKE_DL_LIBS}  # 如果需要动态加载
    pthread          
    # Threads::Threads  # 如果使用了线程
)

if(ENABLE_TEE_BRIDGE EQUAL 1)
    target_link_libraries(${PROJECT_NAME} PRIVATE teec)
endif()

###########################################################################
if(LOGCPP_EXT EQUAL 1)  # LOGCPP_EXT
    find_library(LOG4CPP_LIB log4cpp PATHS 
        ${CMAKE_CURRENT_SOURCE_DIR}/../../log4cpp/build_aarch64/lib 
        ${CMAKE_CURRENT_SOURCE_DIR}/../../log4cpp/build/lib 
        # /another/path/to/library 
        REQUIRED) 

    # REQUIRED 参数会在库找不到时自动停止并报错 如果你需要自定义错误 可以手动判断
    if(NOT LOG4CPP_LIB)
        message(FATAL_ERROR "log4cpp library not found")
    endif()
    message(STATUS "LOG4CPP_LIB : ${LOG4CPP_LIB}")
    # 链接到某个外部库 (假设外部库名为 some_external_lib)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${LOG4CPP_LIB})
endif()
if(RELEASE EQUAL 1) # Release 模式配置
    # 明确指定优化级别（覆盖默认的 Debug 配置）
    # target_compile_options(${PROJECT_NAME} PRIVATE
    #     # $<$<NOT:$<CONFIG:Debug>>:-O3>  # Release 用 -O3
    #     $<$<NOT:$<CONFIG:Debug>>:-O0> 
    # )
    message(STATUS "Release 用 -O0")
    target_compile_options(${PROJECT_NAME} PRIVATE -O0)
    # 如果确实需要禁用优化（调试时）
    # target_compile_options(${PROJECT_NAME} PRIVATE $<$<CONFIG:Debug>:-O0>)
else()
    # Debug 模式默认 -O0（CMake 的 Debug 配置已自动处理）
    # target_compile_options(${PROJECT_NAME} PRIVATE
    #     $<$<CONFIG:Debug,RelWithDebInfo>:-g>
    #     $<$<CONFIG:Debug>:-O0>
    # )
    message(STATUS "Debug 配置用 -g -O0")
    target_compile_options(${PROJECT_NAME} PRIVATE -g -O0)
endif()

# add_definitions() 
target_compile_definitions(${PROJECT_NAME} PRIVATE LOGCPP_EXT=${LOGCPP_EXT})
target_compile_definitions(${PROJECT_NAME} PRIVATE RELEASE=${RELEASE})
target_compile_definitions(${PROJECT_NAME} PRIVATE ENABLE_TEE_BRIDGE=${ENABLE_TEE_BRIDGE})
target_compile_definitions(${PROJECT_NAME} PRIVATE TRUSTSPAN=${TRUSTSPAN})

if(CONFINFER_IS_AARCH64_CROSS_BUILD)
    if(CONFINFER_DEPLOY_LIB_DIR)
        get_filename_component(CONFINFER_DEPLOY_LIB_DIR_ABS
            "${CONFINFER_DEPLOY_LIB_DIR}" ABSOLUTE)
        if(IS_DIRECTORY "${CONFINFER_DEPLOY_LIB_DIR_ABS}")
            message(STATUS "ConfInfer deploy dir exists: ${CONFINFER_DEPLOY_LIB_DIR_ABS}")
            add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        $<TARGET_FILE:${PROJECT_NAME}>
                        "${CONFINFER_DEPLOY_LIB_DIR_ABS}/$<TARGET_FILE_NAME:${PROJECT_NAME}>"
                COMMENT "Copying libConfInfer.so to ${CONFINFER_DEPLOY_LIB_DIR_ABS}")
        else()
            message(STATUS
                "ConfInfer deploy dir not found, skip copy: ${CONFINFER_DEPLOY_LIB_DIR_ABS}")
        endif()
    else()
        message(STATUS "ConfInfer deploy dir is empty, skip copy for aarch64 build")
    endif()
else()
    message(STATUS "Non-aarch64-cross build detected, skip rootfs deploy")
endif()
