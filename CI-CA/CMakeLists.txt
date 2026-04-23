cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(ConfInfer LANGUAGES C CXX)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)      

###########################################################################
# cmake -DLOGCPP_EXT=1 -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..
# cmake -DLOGCPP_EXT=1 -DRELEASE=0 ..

if(NOT DEFINED LOGCPP_EXT)
    set(LOGCPP_EXT 0 CACHE STRING "value of LOGCPP_EXT")
endif()
if(NOT DEFINED RELEASE)
    set(RELEASE 0 CACHE STRING "value of RELEASE")
endif()
###########################################################################
message(STATUS "CMake source directory is: ${CMAKE_CURRENT_SOURCE_DIR}")

file(GLOB MAIN_SOURCES_CPP
    "${CMAKE_CURRENT_SOURCE_DIR}/core/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/ops/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/backend/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/Logger/*.cc"
)

file(GLOB MAIN_SOURCES_C
    "${CMAKE_CURRENT_SOURCE_DIR}/core/ops/cpu/*.c")

message(STATUS "MAIN_SOURCES_CPP = ${MAIN_SOURCES_CPP}")
message(STATUS "MAIN_SOURCES_C = ${MAIN_SOURCES_C}")

add_library(${PROJECT_NAME} SHARED 
            ${MAIN_SOURCES_CPP}
            ${MAIN_SOURCES_C})

target_include_directories(${PROJECT_NAME}
                PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
                PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/Kernel
                PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/Kernel/core/ops
                PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../log4cpp/build_aarch64/include
                PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../log4cpp/build_aarch64/include/threading
                PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../log4cpp/build/include
                PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../log4cpp/build/include/threading)

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
