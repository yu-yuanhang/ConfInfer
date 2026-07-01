#ifndef CONFINFER_TA_COMMANDS_H
#define CONFINFER_TA_COMMANDS_H

#include <tee_internal_api.h>

TEE_Result confinfer_ta_register_model(uint32_t param_types, TEE_Param params[4]);
TEE_Result confinfer_ta_load_params(uint32_t param_types, TEE_Param params[4]);
TEE_Result confinfer_ta_register_partition(uint32_t param_types, TEE_Param params[4]);
TEE_Result confinfer_ta_exec_partition(uint32_t param_types, TEE_Param params[4]);
TEE_Result confinfer_ta_unload_model(uint32_t param_types, TEE_Param params[4]);

TEE_Result confinfer_ta_inc_value(uint32_t param_types, TEE_Param params[4]);
TEE_Result confinfer_ta_dec_value(uint32_t param_types, TEE_Param params[4]);

#endif
