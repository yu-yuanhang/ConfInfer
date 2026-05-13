cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(integration_demo LANGUAGES C CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

file(GLOB MAIN_SOURCES_CPP
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cc"
)

file(GLOB MAIN_SOURCES_C
    "${CMAKE_CURRENT_SOURCE_DIR}/*.c"
)

add_executable(${PROJECT_NAME} ${MAIN_SOURCES_CPP} ${MAIN_SOURCES_C})

set(KERNEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../CI-CA")

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
    REQUIRED)
if(NOT LOG4CPP_LIB)
    message(FATAL_ERROR "log4cpp library not found")
endif()
target_link_libraries(${PROJECT_NAME} PRIVATE ${LOG4CPP_LIB})

find_library(CONFINFER_LIB ConfInfer PATHS
    ${KERNEL_DIR}/build
    REQUIRED)
if(NOT CONFINFER_LIB)
    message(FATAL_ERROR "ConfInfer library not found")
endif()
target_link_libraries(${PROJECT_NAME} PRIVATE ${CONFINFER_LIB})
