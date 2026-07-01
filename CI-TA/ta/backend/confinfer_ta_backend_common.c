#include <tee_internal_api.h>

#include <backend/confinfer_ta_backend_common.h>

TEE_Result ta_backend_require_layer_io(const ta_layer_exec_ctx_t *ctx,
                                       uint32_t input_count,
                                       uint32_t output_count)
{
    if (!ctx || !ctx->model || !ctx->partition || !ctx->layer) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    if (ctx->layer->input_value_count != input_count ||
        ctx->layer->output_value_count != output_count) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    return TEE_SUCCESS;
}

ta_value_t *ta_backend_input(const ta_layer_exec_ctx_t *ctx, uint32_t index)
{
    if (!ctx || !ctx->partition || !ctx->layer ||
        index >= ctx->layer->input_value_count ||
        !ctx->layer->inputs) {
        return NULL;
    }
    return ctx->layer->inputs[index];
}

ta_value_t *ta_backend_output(const ta_layer_exec_ctx_t *ctx, uint32_t index)
{
    if (!ctx || !ctx->partition || !ctx->layer ||
        index >= ctx->layer->output_value_count ||
        !ctx->layer->outputs) {
        return NULL;
    }
    return ctx->layer->outputs[index];
}

ta_param_t *ta_backend_param_by_role(const ta_layer_exec_ctx_t *ctx, uint32_t role)
{
    uint32_t i = 0;

    if (!ctx || !ctx->model || !ctx->layer ||
        !ctx->layer->param_refs || !ctx->layer->params) {
        return NULL;
    }
    for (i = 0; i < ctx->layer->param_ref_count; ++i) {
        const ta_param_ref_t *ref = &ctx->layer->param_refs[i];
        if (ref->role != role) {
            continue;
        }
        return ctx->layer->params[i];
    }
    return NULL;
}

const void *ta_backend_attr(const ta_layer_exec_ctx_t *ctx, size_t min_size)
{
    if (!ctx || !ctx->layer) {
        return NULL;
    }
    if (min_size == 0) {
        return ctx->layer->attr.ptr;
    }
    if (!ctx->layer->attr.ptr || ctx->layer->attr.size < min_size) {
        return NULL;
    }
    return ctx->layer->attr.ptr;
}
