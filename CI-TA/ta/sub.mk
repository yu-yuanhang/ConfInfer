global-incdirs-y += include
global-incdirs-y += include/backend
srcs-y += conf_infer_ta.c
srcs-y += backend/confinfer_ta_backend.c
srcs-y += backend/confinfer_ta_backend_common.c
srcs-y += backend/confinfer_ta_math.c
srcs-y += backend/op_graph.c
srcs-y += backend/op_activation.c
srcs-y += backend/op_arithmetic.c
srcs-y += backend/op_convolution.c
srcs-y += backend/op_linear.c
srcs-y += backend/op_normalization.c
srcs-y += backend/op_pool.c
srcs-y += backend/op_reshape.c
srcs-y += confinfer_ta_commands.c
srcs-y += confinfer_ta_runtime.c

# To remove a certain compiler flag, add a line like this
#cflags-template_ta.c-y += -Wno-strict-prototypes
