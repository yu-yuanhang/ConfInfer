cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(demo LANGUAGES C CXX)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)      

file(GLOB MAIN_SOURCES_CPP
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cc"
)

file(GLOB MAIN_SOURCES_C
    "${CMAKE_CURRENT_SOURCE_DIR}/*.c"
)

message(STATUS "MAIN_SOURCES_CPP = ${MAIN_SOURCES_CPP}")
message(STATUS "MAIN_SOURCES_C = ${MAIN_SOURCES_C}")

add_executable(${PROJECT_NAME} ${MAIN_SOURCES_CPP} ${MAIN_SOURCES_C})

set(KERNEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../Kernel")

target_include_directories(${PROJECT_NAME}
                PUBLIC ${KERNEL_DIR}/include
                PUBLIC ${KERNEL_DIR}/include/Kernel
                PUBLIC ${KERNEL_DIR}/include/Kernel/core/ops
                PUBLIC ${KERNEL_DIR}/../../log4cpp/build_aarch64/include
                PUBLIC ${KERNEL_DIR}/../../log4cpp/build_aarch64/include/threading
                PUBLIC ${KERNEL_DIR}/../../log4cpp/build/include
                PUBLIC ${KERNEL_DIR}/../../log4cpp/build/include/threading)

find_library(LOG4CPP_LIB log4cpp PATHS 
    ${KERNEL_DIR}/../../log4cpp/build_aarch64/lib 
    ${KERNEL_DIR}/../../log4cpp/build/lib 
    # /another/path/to/library 
    REQUIRED) 
# REQUIRED 参数会在库找不到时自动停止并报错 如果你需要自定义错误 可以手动判断
if(NOT LOG4CPP_LIB)
    message(FATAL_ERROR "log4cpp library not found")
endif()
message(STATUS "LOG4CPP_LIB : ${LOG4CPP_LIB}")
# 链接到某个外部库 (假设外部库名为 some_external_lib)
target_link_libraries(${PROJECT_NAME} PRIVATE ${LOG4CPP_LIB})

find_library(TRUSTINFER_LIB TrustInfer PATHS 
    ${KERNEL_DIR}/build
    # /another/path/to/library 
    REQUIRED) 
# REQUIRED 参数会在库找不到时自动停止并报错 如果你需要自定义错误 可以手动判断
if(NOT TRUSTINFER_LIB)
    message(FATAL_ERROR "TrustInfer library not found")
endif()
message(STATUS "TRUSTINFER_LIB : ${TRUSTINFER_LIB}")
# 链接到某个外部库 (假设外部库名为 some_external_lib)
target_link_libraries(${PROJECT_NAME} PRIVATE ${TRUSTINFER_LIB})