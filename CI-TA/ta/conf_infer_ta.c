/*
 * Copyright (c) 2016, Linaro Limited
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>

#include <conf_infer_ta.h>
#include <confinfer_ta_commands.h>
#include <confinfer_ta_runtime.h>

TEE_Result TA_CreateEntryPoint(void)
{
    ta_runtime_init();
    DMSG("TA_CreateEntryPoint");
    return TEE_SUCCESS;
}

void TA_DestroyEntryPoint(void)
{
    ta_runtime_deinit();
    DMSG("TA_DestroyEntryPoint");
}

TEE_Result TA_OpenSessionEntryPoint(uint32_t param_types,
                                    TEE_Param __maybe_unused params[4],
                                    void __maybe_unused **sess_ctx)
{
    const uint32_t exp_param_types =
        TEE_PARAM_TYPES(TEE_PARAM_TYPE_NONE,
                        TEE_PARAM_TYPE_NONE,
                        TEE_PARAM_TYPE_NONE,
                        TEE_PARAM_TYPE_NONE);

    if (param_types != exp_param_types) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    confinfer_ta_session_t *ctx = NULL;

    (void)&params;
    ctx = TEE_Malloc(sizeof(*ctx), TEE_MALLOC_FILL_ZERO);
    if (!ctx) {
        return TEE_ERROR_OUT_OF_MEMORY;
    }
    *sess_ctx = ctx;
    IMSG("ConfInfer TA session opened");
    return TEE_SUCCESS;
}

void TA_CloseSessionEntryPoint(void __maybe_unused *sess_ctx)
{
    confinfer_ta_session_t *ctx = (confinfer_ta_session_t *)sess_ctx;

    if (ctx) {
        if (ctx->prepare_image_upload.buffer) {
            TEE_Free(ctx->prepare_image_upload.buffer);
            ctx->prepare_image_upload.buffer = NULL;
        }
        if (ctx->exec_partition_upload.input_buffer) {
            TEE_Free(ctx->exec_partition_upload.input_buffer);
            ctx->exec_partition_upload.input_buffer = NULL;
        }
        if (ctx->exec_partition_upload.output_buffer) {
            TEE_Free(ctx->exec_partition_upload.output_buffer);
            ctx->exec_partition_upload.output_buffer = NULL;
        }
        TEE_Free(ctx);
    }
    IMSG("ConfInfer TA session closed");
}

TEE_Result TA_InvokeCommandEntryPoint(void __maybe_unused *sess_ctx,
                                      uint32_t cmd_id,
                                      uint32_t param_types,
                                      TEE_Param params[4])
{
    switch (cmd_id) {
    case TA_CONFINFER_CMD_PREPARE_MODEL_IMAGE:
        return confinfer_ta_prepare_model_image(sess_ctx, param_types, params);
    case TA_CONFINFER_CMD_PREPARE_MODEL_IMAGE_BEGIN:
        return confinfer_ta_prepare_model_image_begin(sess_ctx, param_types, params);
    case TA_CONFINFER_CMD_PREPARE_MODEL_IMAGE_CHUNK:
        return confinfer_ta_prepare_model_image_chunk(sess_ctx, param_types, params);
    case TA_CONFINFER_CMD_PREPARE_MODEL_IMAGE_END:
        return confinfer_ta_prepare_model_image_end(sess_ctx, param_types, params);
    case TA_CONFINFER_CMD_EXEC_PARTITION:
        return confinfer_ta_exec_partition(sess_ctx, param_types, params);
    case TA_CONFINFER_CMD_EXEC_PARTITION_BEGIN:
        return confinfer_ta_exec_partition_begin(sess_ctx, param_types, params);
    case TA_CONFINFER_CMD_EXEC_PARTITION_INPUT_CHUNK:
        return confinfer_ta_exec_partition_input_chunk(sess_ctx, param_types, params);
    case TA_CONFINFER_CMD_EXEC_PARTITION_RUN:
        return confinfer_ta_exec_partition_run(sess_ctx, param_types, params);
    case TA_CONFINFER_CMD_EXEC_PARTITION_OUTPUT_CHUNK:
        return confinfer_ta_exec_partition_output_chunk(sess_ctx, param_types, params);
    case TA_CONFINFER_CMD_EXEC_PARTITION_END:
        return confinfer_ta_exec_partition_end(sess_ctx, param_types, params);
    case TA_CONFINFER_CMD_UNLOAD_MODEL:
        return confinfer_ta_unload_model(sess_ctx, param_types, params);
    case TA_CONFINFER_CMD_INC_VALUE:
        return confinfer_ta_inc_value(sess_ctx, param_types, params);
    case TA_CONFINFER_CMD_DEC_VALUE:
        return confinfer_ta_dec_value(sess_ctx, param_types, params);
    default:
        return TEE_ERROR_BAD_PARAMETERS;
    }
}
