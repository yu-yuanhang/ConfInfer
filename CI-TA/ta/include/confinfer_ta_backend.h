#ifndef CONFINFER_TA_BACKEND_H
#define CONFINFER_TA_BACKEND_H

#include <tee_internal_api.h>

#include <confinfer_ta_runtime.h>

typedef struct {
    ta_model_t *model;
    ta_partition_t *partition;
    ta_layer_t *layer;
} ta_layer_exec_ctx_t;

typedef struct ta_backend_s {
    const char *name;
    TEE_Result (*execute_partition)(const struct ta_backend_s *backend,
                                    ta_model_t *model,
                                    ta_partition_t *partition);
    TEE_Result (*execute_layer)(const struct ta_backend_s *backend,
                                ta_layer_exec_ctx_t *ctx);
} ta_backend_t;

const ta_backend_t *ta_backend_default(void);

TEE_Result ta_backend_execute_partition(const ta_backend_t *backend,
                                        ta_model_t *model,
                                        ta_partition_t *partition);

TEE_Result ta_backend_execute_layer(const ta_backend_t *backend,
                                    ta_layer_exec_ctx_t *ctx);

#endif
