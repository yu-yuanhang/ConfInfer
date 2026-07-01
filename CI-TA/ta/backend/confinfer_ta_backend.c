#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>

#include <backend/confinfer_ta_backend_ops.h>

static TEE_Result default_execute_layer(const ta_backend_t *backend,
                                        ta_layer_exec_ctx_t *ctx)
{
    (void)backend;

    if (!ctx || !ctx->layer) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    switch (ctx->layer->layer_type) {
    case CONFINFER_LAYER_GRAPH_INPUT:
        return ta_execute_graph_input(ctx);
    case CONFINFER_LAYER_GRAPH_OUTPUT:
        return ta_execute_graph_output(ctx);
    case CONFINFER_LAYER_CONV2D:
        return ta_execute_conv2d_fp32(ctx);
    case CONFINFER_LAYER_MAXPOOL2D:
        return ta_execute_maxpool2d_fp32(ctx);
    case CONFINFER_LAYER_AVGPOOL2D:
        return ta_execute_avgpool2d_fp32(ctx);
    case CONFINFER_LAYER_ADAPTIVEAVGPOOL2D:
        return ta_execute_adaptive_avgpool2d_fp32(ctx);
    case CONFINFER_LAYER_ADAPTIVEMAXPOOL2D:
        return ta_execute_adaptive_maxpool2d_fp32(ctx);
    case CONFINFER_LAYER_BATCHNORM2D:
        return ta_execute_batchnorm2d_fp32(ctx);
    case CONFINFER_LAYER_LAYERNORM:
        return ta_execute_layernorm_fp32(ctx);
    case CONFINFER_LAYER_GROUPNORM:
        return ta_execute_groupnorm_fp32(ctx);
    case CONFINFER_LAYER_RELU:
        return ta_execute_relu_fp32(ctx);
    case CONFINFER_LAYER_SIGMOID:
        return ta_execute_sigmoid_fp32(ctx);
    case CONFINFER_LAYER_DROPOUT:
        return ta_execute_dropout_fp32(ctx);
    case CONFINFER_LAYER_SOFTMAX:
        return ta_execute_softmax_fp32(ctx);
    case CONFINFER_LAYER_FLATTEN:
        return ta_execute_flatten_default(ctx);
    case CONFINFER_LAYER_BIASADD:
        return ta_execute_bias_add_fp32(ctx);
    case CONFINFER_LAYER_ADD:
        return ta_execute_add_fp32(ctx);
    case CONFINFER_LAYER_MUL:
        return ta_execute_mul_fp32(ctx);
    case CONFINFER_LAYER_CONCAT:
        return ta_execute_concat_fp32(ctx);
    case CONFINFER_LAYER_MATMUL:
        return ta_execute_matmul_fp32(ctx);
    case CONFINFER_LAYER_LINEAR:
        return ta_execute_linear_fp32(ctx);
    default:
        EMSG("unsupported TA layer_type=%u layer_id=%u",
             ctx->layer->layer_type, ctx->layer->layer_id);
        return TEE_ERROR_NOT_SUPPORTED;
    }
}

static TEE_Result default_execute_partition(const ta_backend_t *backend,
                                            ta_model_t *model,
                                            ta_partition_t *partition)
{
    ta_layer_exec_ctx_t ctx;
    uint32_t i = 0;
    TEE_Result res = TEE_SUCCESS;

    if (!backend || !model || !partition) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    for (i = 0; i < partition->layer_count; ++i) {
        TEE_MemFill(&ctx, 0, sizeof(ctx));
        ctx.model = model;
        ctx.partition = partition;
        ctx.layer = &partition->layers[i];

        res = backend->execute_layer(backend, &ctx);
        if (res != TEE_SUCCESS) {
            EMSG("backend execute_layer failed: layer_id=%u layer_type=%u res=0x%x",
                 ctx.layer->layer_id, ctx.layer->layer_type, res);
            return res;
        }
    }

    return TEE_SUCCESS;
}

static const ta_backend_t g_default_backend = {
    .name = "ta_cpu_ref",
    .execute_partition = default_execute_partition,
    .execute_layer = default_execute_layer,
};

const ta_backend_t *ta_backend_default(void)
{
    return &g_default_backend;
}

TEE_Result ta_backend_execute_partition(const ta_backend_t *backend,
                                        ta_model_t *model,
                                        ta_partition_t *partition)
{
    if (!backend || !backend->execute_partition) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    return backend->execute_partition(backend, model, partition);
}

TEE_Result ta_backend_execute_layer(const ta_backend_t *backend,
                                    ta_layer_exec_ctx_t *ctx)
{
    if (!backend || !backend->execute_layer) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    return backend->execute_layer(backend, ctx);
}
