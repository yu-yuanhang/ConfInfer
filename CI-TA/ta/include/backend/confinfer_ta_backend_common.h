#ifndef CONFINFER_TA_BACKEND_COMMON_H
#define CONFINFER_TA_BACKEND_COMMON_H

#include <backend/confinfer_ta_backend_ops.h>

TEE_Result ta_backend_require_layer_io(const ta_layer_exec_ctx_t *ctx,
                                       uint32_t input_count,
                                       uint32_t output_count);

ta_value_t *ta_backend_input(const ta_layer_exec_ctx_t *ctx, uint32_t index);
ta_value_t *ta_backend_output(const ta_layer_exec_ctx_t *ctx, uint32_t index);
ta_param_t *ta_backend_param_by_role(const ta_layer_exec_ctx_t *ctx, uint32_t role);

const void *ta_backend_attr(const ta_layer_exec_ctx_t *ctx, size_t min_size);

#endif
