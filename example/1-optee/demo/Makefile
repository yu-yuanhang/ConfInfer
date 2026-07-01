WORK_ROOT := $(abspath $(CURDIR)/../../../../..)
REPO_ROOT := $(WORK_ROOT)/optee_qemuv8/optee_doc-3.22.0
CI_CA_DIR := $(REPO_ROOT)/CI/ConfInfer/CI-CA
CI_TA_DIR := $(REPO_ROOT)/CI/ConfInfer/CI-TA
HOST_DIR := $(REPO_ROOT)/out-br/host
TARGET_ROOTFS_DIR := $(REPO_ROOT)/out-br/target
TARGET_APP_DIR := $(TARGET_ROOTFS_DIR)/root
TARGET_LIB_DIR := $(TARGET_ROOTFS_DIR)/usr/lib64
SYSROOT := $(HOST_DIR)/aarch64-buildroot-linux-gnu/sysroot
BIN_DIR := $(HOST_DIR)/bin

override CC := $(BIN_DIR)/aarch64-linux-gnu-gcc
override CXX := $(BIN_DIR)/aarch64-linux-gnu-g++
override AR := $(BIN_DIR)/aarch64-linux-gnu-ar
override RANLIB := $(BIN_DIR)/aarch64-linux-gnu-ranlib

TEEC_EXPORT := $(SYSROOT)/usr
ARTIFACT_DIR := $(CURDIR)
OUT_DIR := $(CURDIR)/out
OBJS := $(OUT_DIR)/main.o
BINARY := $(OUT_DIR)/confinfer_demo
DEPLOY_BINARY := $(TARGET_APP_DIR)/$(notdir $(BINARY))
CONFINFER_LIB := $(TARGET_LIB_DIR)/libConfInfer.so

CPPFLAGS += --sysroot=$(SYSROOT)
CPPFLAGS += -I$(CI_CA_DIR)/include
CPPFLAGS += -I$(CI_CA_DIR)/include/Kernel
CPPFLAGS += -I$(CI_CA_DIR)/include/Kernel/core/ops
CPPFLAGS += -I$(CI_TA_DIR)/host/include
CPPFLAGS += -I$(CI_TA_DIR)/ta/include
CPPFLAGS += -I$(TEEC_EXPORT)/include
CXXFLAGS += -Wall -Wextra -O2 -std=c++20

LDADD += $(CONFINFER_LIB)
LDADD += -L$(TEEC_EXPORT)/lib -lteec -ldl -lpthread -lm
# -Wl,xxx 表示 xxx 这个参数传给链接器 ld
# -rpath-link 是链接阶段使用的库搜索路径
LDFLAGS += -Wl,-rpath-link,$(TARGET_LIB_DIR)
LDFLAGS += -Wl,-rpath,/usr/lib64 -Wl,-rpath,/usr/lib

.PHONY: all clean info sync_ci_ta check_confinfer_lib

all: sync_ci_ta check_confinfer_lib $(BINARY)

sync_ci_ta:
	bash $(CI_TA_DIR)/scripts/sync_to_optee_examples.sh

check_confinfer_lib:
	test -f "$(CONFINFER_LIB)" || \
	( echo "Missing $(CONFINFER_LIB). Build and deploy the aarch64 ConfInfer library first."; exit 1 )

info:
	@echo "REPO_ROOT=$(REPO_ROOT)"
	@echo "CI_CA_DIR=$(CI_CA_DIR)"
	@echo "CI_TA_DIR=$(CI_TA_DIR)"
	@echo "HOST_DIR=$(HOST_DIR)"
	@echo "SYSROOT=$(SYSROOT)"
	@echo "CC=$(CC)"
	@echo "CXX=$(CXX)"
	@echo "TARGET_APP_DIR=$(TARGET_APP_DIR)"
	@echo "TARGET_LIB_DIR=$(TARGET_LIB_DIR)"
	@echo "CONFINFER_LIB=$(CONFINFER_LIB)"
	@echo "BINARY=$(BINARY)"
	@echo "DEPLOY_BINARY=$(DEPLOY_BINARY)"

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

$(ARTIFACT_DIR):
	mkdir -p $(ARTIFACT_DIR)

$(OUT_DIR)/main.o: main.cc | $(OUT_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BINARY): check_confinfer_lib $(OBJS) | $(OUT_DIR)
	$(CXX) $(LDFLAGS) -o $@ $(OBJS) $(LDADD)
	mkdir -p $(TARGET_APP_DIR)
	cp -f $@ $(DEPLOY_BINARY)


clean:
	rm -rf $(OUT_DIR) $(BINARY)
