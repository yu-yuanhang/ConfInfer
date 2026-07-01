#ifndef CONFINFER_TA_BACKEND_OPS_H
#define CONFINFER_TA_BACKEND_OPS_H

#include <confinfer_ta_backend.h>

TEE_Result ta_execute_conv2d_fp32(ta_layer_exec_ctx_t *ctx);
TEE_Result ta_execute_batchnorm2d_fp32(ta_layer_exec_ctx_t *ctx);
TEE_Result ta_execute_layernorm_fp32(ta_layer_exec_ctx_t *ctx);
TEE_Result ta_execute_groupnorm_fp32(ta_layer_exec_ctx_t *ctx);
TEE_Result ta_execute_graph_input(ta_layer_exec_ctx_t *ctx);
TEE_Result ta_execute_graph_output(ta_layer_exec_ctx_t *ctx);
TEE_Result ta_execute_relu_fp32(ta_layer_exec_ctx_t *ctx);
TEE_Result ta_execute_sigmoid_fp32(ta_layer_exec_ctx_t *ctx);
TEE_Result ta_execute_dropout_fp32(ta_layer_exec_ctx_t *ctx);
TEE_Result ta_execute_softmax_fp32(ta_layer_exec_ctx_t *ctx);
TEE_Result ta_execute_add_fp32(ta_layer_exec_ctx_t *ctx);
TEE_Result ta_execute_mul_fp32(ta_layer_exec_ctx_t *ctx);
TEE_Result ta_execute_bias_add_fp32(ta_layer_exec_ctx_t *ctx);
TEE_Result ta_execute_concat_fp32(ta_layer_exec_ctx_t *ctx);
TEE_Result ta_execute_flatten_default(ta_layer_exec_ctx_t *ctx);
TEE_Result ta_execute_matmul_fp32(ta_layer_exec_ctx_t *ctx);
TEE_Result ta_execute_linear_fp32(ta_layer_exec_ctx_t *ctx);
TEE_Result ta_execute_maxpool2d_fp32(ta_layer_exec_ctx_t *ctx);
TEE_Result ta_execute_avgpool2d_fp32(ta_layer_exec_ctx_t *ctx);
TEE_Result ta_execute_adaptive_avgpool2d_fp32(ta_layer_exec_ctx_t *ctx);
TEE_Result ta_execute_adaptive_maxpool2d_fp32(ta_layer_exec_ctx_t *ctx);

#endif
