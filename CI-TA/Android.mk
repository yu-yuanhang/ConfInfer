###################### optee-conf-infer ######################
LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
LOCAL_CFLAGS += -DANDROID_BUILD
LOCAL_CFLAGS += -Wall

LOCAL_SRC_FILES += host/confinfer_host.c

LOCAL_C_INCLUDES := $(LOCAL_PATH)/ta/include
LOCAL_C_INCLUDES += $(LOCAL_PATH)/host/include

LOCAL_SHARED_LIBRARIES := libteec
LOCAL_MODULE := libconfinfer_host
LOCAL_VENDOR_MODULE := true
LOCAL_MODULE_TAGS := optional
include $(BUILD_STATIC_LIBRARY)

include $(LOCAL_PATH)/ta/Android.mk
