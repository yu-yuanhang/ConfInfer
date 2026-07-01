#include <core/ProtoExecBridge.h>
#include <cstring>

namespace Kernel {
namespace core {

namespace {

std::vector<uint8_t> pack_value_blob(const std::vector<Value_t *>& values) {
    std::vector<uint8_t> blob;
    size_t total = 0;

    for (auto it = values.begin(); it != values.end(); ++it) {
        Value_t *value = *it;
        EXIT_ERROR_CHECK_EQ(nullptr, value, "Value_t is nullptr");
        EXIT_ERROR_CHECK_EQ(nullptr, value->data.ptr, "Value_t data.ptr is nullptr");
        total += value->data.shape.size * value->data.getTypeSize();
    }

    blob.resize(total);
    uint8_t *cursor = blob.data();
    for (auto it = values.begin(); it != values.end(); ++it) {
        Value_t *value = *it;
        const size_t byte_size = value->data.shape.size * value->data.getTypeSize();
        if (byte_size > 0) {
            std::memcpy(cursor, value->data.ptr, byte_size);
            cursor += byte_size;
        }
    }
    return blob;
}

void unpack_value_blob(const std::vector<Value_t *>& values, const std::vector<uint8_t>& blob) {
    const uint8_t *cursor = blob.data();
    size_t remain = blob.size();

    for (auto it = values.begin(); it != values.end(); ++it) {
        Value_t *value = *it;
        const size_t byte_size = value->data.shape.size * value->data.getTypeSize();
        EXIT_ERROR_CHECK_EQ(nullptr, value, "Value_t is nullptr");
        EXIT_ERROR_CHECK_EQ(nullptr, value->data.ptr, "Value_t data.ptr is nullptr");
        EXIT_ERROR_CHECK_EQ(true, byte_size > remain, "Output blob size is too small");
        if (byte_size > 0) {
            std::memcpy(value->data.ptr, cursor, byte_size);
            cursor += byte_size;
            remain -= byte_size;
        }
    }
    EXIT_ERROR_CHECK_NE(0u, static_cast<UINT>(remain), "Output blob has trailing bytes");
}

} // namespace

bool ProtoExecBridge::execute(const ExecUnit& unit, Executor *exec, ThreadCtx_t *ctx) {
    confinfer_partition_rsp_t rsp{};
    std::vector<uint8_t> input_blob;
    std::vector<uint8_t> output_blob;
    bool ok = false;

    (void)exec;
    (void)ctx;

    EXIT_ERROR_CHECK_EQ(nullptr, _runner, "ProtoExecBridge runner is nullptr");
    _last_proto = make_exec_unit_proto(unit);
    EXIT_ERROR_CHECK_EQ(CONFINFER_INVALID_MODEL_ID, _active_model_id,
                        "ProtoExecBridge active model_id is invalid");
    _last_proto.req.model_id = _active_model_id;
    input_blob = pack_value_blob(unit.inputs());
    output_blob.resize(_last_proto.data.req.total_output_bytes);

    ok = _runner(&_last_proto.req,
                 _last_proto.layers.empty() ? nullptr : _last_proto.layers.data(),
                 static_cast<UINT>(_last_proto.layers.size()),
                 _last_proto.layer_attrs.empty() ? nullptr : _last_proto.layer_attrs.data(),
                 static_cast<UINT>(_last_proto.layer_attrs.size()),
                 &_last_proto.data.req,
                 _last_proto.data.inputs.empty() ? nullptr : &_last_proto.data.inputs.front().desc,
                 static_cast<UINT>(_last_proto.data.inputs.size()),
                 _last_proto.data.outputs.empty() ? nullptr : &_last_proto.data.outputs.front().desc,
                 static_cast<UINT>(_last_proto.data.outputs.size()),
                 _last_proto.data.internals.empty() ? nullptr : &_last_proto.data.internals.front().desc,
                 static_cast<UINT>(_last_proto.data.internals.size()),
                 _last_proto.data.layer_ios.empty() ? nullptr : _last_proto.data.layer_ios.data(),
                 static_cast<UINT>(_last_proto.data.layer_ios.size()),
                 _last_proto.data.input_refs.empty() ? nullptr : _last_proto.data.input_refs.data(),
                 static_cast<UINT>(_last_proto.data.input_refs.size()),
                 _last_proto.data.output_refs.empty() ? nullptr : _last_proto.data.output_refs.data(),
                 static_cast<UINT>(_last_proto.data.output_refs.size()),
                 _last_proto.data.param_refs.empty() ? nullptr : _last_proto.data.param_refs.data(),
                 static_cast<UINT>(_last_proto.data.param_refs.size()),
                 input_blob.empty() ? nullptr : input_blob.data(),
                 static_cast<UINT>(input_blob.size()),
                 output_blob.empty() ? nullptr : output_blob.data(),
                 static_cast<UINT>(output_blob.size()),
                 &rsp,
                 _user_ctx);
    if (!ok) {
        return false;
    }

    EXIT_ERROR_CHECK_NE(CONFINFER_PROTOCOL_VERSION, rsp.version,
        "ProtoExecBridge response version mismatch");
    EXIT_ERROR_CHECK_NE(_last_proto.req.domain, rsp.domain,
        "ProtoExecBridge response domain mismatch");
    EXIT_ERROR_CHECK_NE(_last_proto.req.model_id, rsp.model_id,
        "ProtoExecBridge response model_id mismatch");
    EXIT_ERROR_CHECK_NE(_last_proto.req.partition_id, rsp.partition_id,
        "ProtoExecBridge response partition_id mismatch");
    EXIT_ERROR_CHECK_NE(CONFINFER_PART_OK, rsp.status,
        "ProtoExecBridge remote execution failed");
    EXIT_ERROR_CHECK_NE(_last_proto.req.layer_count, rsp.executed_layers,
        "ProtoExecBridge executed layer count mismatch");
    EXIT_ERROR_CHECK_NE(_last_proto.req.output_count, rsp.produced_outputs,
        "ProtoExecBridge produced output count mismatch");

    unpack_value_blob(unit.outputs(), output_blob);

    return true;
}

bool ProtoExecBridge::registerModel(const confinfer_model_desc_t& desc,
                                    confinfer_model_rsp_t *rsp) {
    confinfer_model_rsp_t local_rsp{};

    EXIT_ERROR_CHECK_EQ(nullptr, _register_model, "ProtoExecBridge register_model callback is nullptr");
    if (nullptr == rsp) {
        rsp = &local_rsp;
    }
    if (!_register_model(&desc, rsp, _user_ctx)) {
        return false;
    }
    EXIT_ERROR_CHECK_NE(CONFINFER_PROTOCOL_VERSION, rsp->version,
                        "ProtoExecBridge registerModel response version mismatch");
    EXIT_ERROR_CHECK_NE(desc.model_id, rsp->model_id,
                        "ProtoExecBridge registerModel response model_id mismatch");
    EXIT_ERROR_CHECK_NE(CONFINFER_MODEL_OK, rsp->status,
                        "ProtoExecBridge registerModel remote failed");
    _active_model_id = desc.model_id;
    return true;
}

bool ProtoExecBridge::loadParams(const confinfer_load_params_req_t& req,
                                 const confinfer_param_desc_t *param_descs,
                                 UINT param_count,
                                 const void *param_blob,
                                 UINT param_blob_size,
                                 confinfer_load_params_rsp_t *rsp) {
    confinfer_load_params_rsp_t local_rsp{};

    EXIT_ERROR_CHECK_EQ(nullptr, _load_params, "ProtoExecBridge load_params callback is nullptr");
    if (nullptr == rsp) {
        rsp = &local_rsp;
    }
    if (!_load_params(&req,
                      param_descs,
                      param_count,
                      param_blob,
                      param_blob_size,
                      rsp,
                      _user_ctx)) {
        return false;
    }
    EXIT_ERROR_CHECK_NE(CONFINFER_PROTOCOL_VERSION, rsp->version,
                        "ProtoExecBridge loadParams response version mismatch");
    EXIT_ERROR_CHECK_NE(req.model_id, rsp->model_id,
                        "ProtoExecBridge loadParams response model_id mismatch");
    EXIT_ERROR_CHECK_NE(CONFINFER_PARAM_OK, rsp->status,
                        "ProtoExecBridge loadParams remote failed");
    return true;
}

bool ProtoExecBridge::registerPartition(const ExecUnit& unit,
                                        confinfer_model_id_t model_id,
                                        confinfer_partition_rsp_t *rsp) {
    confinfer_partition_rsp_t local_rsp{};

    EXIT_ERROR_CHECK_EQ(nullptr, _register_partition,
                        "ProtoExecBridge register_partition callback is nullptr");
    _last_proto = make_exec_unit_proto(unit);
    _last_proto.req.model_id = model_id;

    if (nullptr == rsp) {
        rsp = &local_rsp;
    }
    if (!_register_partition(&_last_proto.req,
                             _last_proto.layers.empty() ? nullptr : _last_proto.layers.data(),
                             static_cast<UINT>(_last_proto.layers.size()),
                             _last_proto.layer_attrs.empty() ? nullptr : _last_proto.layer_attrs.data(),
                             static_cast<UINT>(_last_proto.layer_attrs.size()),
                             &_last_proto.data.req,
                             _last_proto.data.inputs.empty() ? nullptr : &_last_proto.data.inputs.front().desc,
                             static_cast<UINT>(_last_proto.data.inputs.size()),
                             _last_proto.data.outputs.empty() ? nullptr : &_last_proto.data.outputs.front().desc,
                             static_cast<UINT>(_last_proto.data.outputs.size()),
                             _last_proto.data.internals.empty() ? nullptr : &_last_proto.data.internals.front().desc,
                             static_cast<UINT>(_last_proto.data.internals.size()),
                             _last_proto.data.layer_ios.empty() ? nullptr : _last_proto.data.layer_ios.data(),
                             static_cast<UINT>(_last_proto.data.layer_ios.size()),
                             _last_proto.data.input_refs.empty() ? nullptr : _last_proto.data.input_refs.data(),
                             static_cast<UINT>(_last_proto.data.input_refs.size()),
                             _last_proto.data.output_refs.empty() ? nullptr : _last_proto.data.output_refs.data(),
                             static_cast<UINT>(_last_proto.data.output_refs.size()),
                             _last_proto.data.param_refs.empty() ? nullptr : _last_proto.data.param_refs.data(),
                             static_cast<UINT>(_last_proto.data.param_refs.size()),
                             rsp,
                             _user_ctx)) {
        return false;
    }
    EXIT_ERROR_CHECK_NE(CONFINFER_PROTOCOL_VERSION, rsp->version,
                        "ProtoExecBridge registerPartition response version mismatch");
    EXIT_ERROR_CHECK_NE(_last_proto.req.model_id, rsp->model_id,
                        "ProtoExecBridge registerPartition response model_id mismatch");
    EXIT_ERROR_CHECK_NE(_last_proto.req.partition_id, rsp->partition_id,
                        "ProtoExecBridge registerPartition response partition_id mismatch");
    EXIT_ERROR_CHECK_NE(_last_proto.req.domain, rsp->domain,
                        "ProtoExecBridge registerPartition response domain mismatch");
    EXIT_ERROR_CHECK_NE(CONFINFER_PART_OK, rsp->status,
                        "ProtoExecBridge registerPartition remote failed");
    return true;
}

bool ProtoExecBridge::unloadModel(const confinfer_unload_model_req_t& req,
                                  confinfer_unload_model_rsp_t *rsp) {
    confinfer_unload_model_rsp_t local_rsp{};

    EXIT_ERROR_CHECK_EQ(nullptr, _unload_model,
                        "ProtoExecBridge unload_model callback is nullptr");
    if (nullptr == rsp) {
        rsp = &local_rsp;
    }
    if (!_unload_model(&req, rsp, _user_ctx)) {
        return false;
    }
    EXIT_ERROR_CHECK_NE(CONFINFER_PROTOCOL_VERSION, rsp->version,
                        "ProtoExecBridge unloadModel response version mismatch");
    EXIT_ERROR_CHECK_NE(req.model_id, rsp->model_id,
                        "ProtoExecBridge unloadModel response model_id mismatch");
    EXIT_ERROR_CHECK_NE(CONFINFER_UNLOAD_MODEL_OK, rsp->status,
                        "ProtoExecBridge unloadModel remote failed");
    _active_model_id = CONFINFER_INVALID_MODEL_ID;
    return true;
}

} // namespace core
} // namespace Kernel
