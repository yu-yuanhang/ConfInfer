CC      ?= $(CROSS_COMPILE)gcc
LD      ?= $(CROSS_COMPILE)ld
AR      ?= $(CROSS_COMPILE)ar
NM      ?= $(CROSS_COMPILE)nm
OBJCOPY ?= $(CROSS_COMPILE)objcopy
OBJDUMP ?= $(CROSS_COMPILE)objdump
READELF ?= $(CROSS_COMPILE)readelf
RANLIB  ?= $(CROSS_COMPILE)ranlib

OUT_DIR := $(CURDIR)/out
LIB_OBJS := $(OUT_DIR)/confinfer_host.o

TEEC_EXPORT ?=

CFLAGS += -Wall -Wextra -O2 -fPIC
CFLAGS += -I../ta/include -I./include
ifneq ($(strip $(TEEC_EXPORT)),)
CFLAGS += -I$(TEEC_EXPORT)/include
LDADD += -L$(TEEC_EXPORT)/lib
endif
#Add/link other required libraries here
LDADD += -lteec

HOSTLIB = libconfinfer_host.a

.PHONY: all lib clean info
all: $(HOSTLIB)
lib: $(HOSTLIB)

info:
	@echo "CROSS_COMPILE=$(CROSS_COMPILE)"
	@echo "TEEC_EXPORT=$(TEEC_EXPORT)"
	@echo "CC=$(CC)"
	@echo "AR=$(AR)"
	@echo "RANLIB=$(RANLIB)"

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

$(HOSTLIB): $(LIB_OBJS)
	$(AR) rcs $@ $(LIB_OBJS)
	$(RANLIB) $@

.PHONY: clean
clean:
	rm -rf $(OUT_DIR) $(HOSTLIB)

$(OUT_DIR)/confinfer_host.o: confinfer_host.c ./include/confinfer_host.h | $(OUT_DIR)
	$(CC) $(CFLAGS) -c $< -o $@
