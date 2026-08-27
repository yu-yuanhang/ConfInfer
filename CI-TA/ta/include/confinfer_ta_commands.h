#ifndef CONFINFER_TA_COMMANDS_H
#define CONFINFER_TA_COMMANDS_H

#include <tee_internal_api.h>
#include <confinfer_protocol.h>

typedef struct {
    confinfer_model_id_t model_id;
    uint8_t *buffer;
    uint32_t total_size;
    uint32_t received_size;
} confinfer_prepare_image_upload_t;

typedef struct {
    confinfer_model_id_t model_id;
    confinfer_partition_id_t partition_id;
    uint32_t input_count;
    uint32_t output_count;
    uint32_t total_input_bytes;
    uint32_t total_output_bytes;
    uint8_t *input_buffer;
    uint8_t *output_buffer;
    uint32_t received_input_bytes;
    uint32_t produced_output_bytes;
    uint32_t run_completed;
} confinfer_exec_partition_upload_t;

typedef struct {
    confinfer_prepare_image_upload_t prepare_image_upload;
    confinfer_exec_partition_upload_t exec_partition_upload;
} confinfer_ta_session_t;

TEE_Result confinfer_ta_prepare_model_image(void *sess_ctx, uint32_t param_types, TEE_Param params[4]);
TEE_Result confinfer_ta_prepare_model_image_begin(void *sess_ctx, uint32_t param_types, TEE_Param params[4]);
TEE_Result confinfer_ta_prepare_model_image_chunk(void *sess_ctx, uint32_t param_types, TEE_Param params[4]);
TEE_Result confinfer_ta_prepare_model_image_end(void *sess_ctx, uint32_t param_types, TEE_Param params[4]);
TEE_Result confinfer_ta_exec_partition(void *sess_ctx, uint32_t param_types, TEE_Param params[4]);
TEE_Result confinfer_ta_exec_partition_begin(void *sess_ctx, uint32_t param_types, TEE_Param params[4]);
TEE_Result confinfer_ta_exec_partition_input_chunk(void *sess_ctx, uint32_t param_types, TEE_Param params[4]);
TEE_Result confinfer_ta_exec_partition_run(void *sess_ctx, uint32_t param_types, TEE_Param params[4]);
TEE_Result confinfer_ta_exec_partition_output_chunk(void *sess_ctx, uint32_t param_types, TEE_Param params[4]);
TEE_Result confinfer_ta_exec_partition_end(void *sess_ctx, uint32_t param_types, TEE_Param params[4]);
TEE_Result confinfer_ta_unload_model(void *sess_ctx, uint32_t param_types, TEE_Param params[4]);

TEE_Result confinfer_ta_inc_value(void *sess_ctx, uint32_t param_types, TEE_Param params[4]);
TEE_Result confinfer_ta_dec_value(void *sess_ctx, uint32_t param_types, TEE_Param params[4]);

#endif
