#include <tee_internal_api.h>

#include <backend/confinfer_ta_backend_common.h>
#include <backend/confinfer_ta_backend_ops.h>

TEE_Result ta_execute_flatten_default(ta_layer_exec_ctx_t *ctx)
{
    ta_value_t *input = NULL;
    ta_value_t *output = NULL;

    if (TEE_SUCCESS != ta_backend_require_layer_io(ctx, 1, 1)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    input = ta_backend_input(ctx, 0);
    output = ta_backend_output(ctx, 0);
    if (!input || !output || !input->data.ptr || !output->data.ptr) {
        return TEE_ERROR_BAD_STATE;
    }
    if (input->data.byte_size != output->data.byte_size) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    // 当前协议未下发 start_dim / end_dim，先按 REE 默认 Flatten(1, -1) 语义，
    // 执行阶段只需要透传底层数据，shape 已在注册阶段由 REE 侧算好。
    TEE_MemMove(output->data.ptr, input->data.ptr, input->data.byte_size);
    return TEE_SUCCESS;
}
