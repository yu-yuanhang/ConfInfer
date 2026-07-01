# CI-TA

## 结构定位

`CI-TA` 是 ConfInfer 的 TEE 相关源码主目录。

当前唯一源码修改入口固定为：

`/home/yyh/2.Programs/2.workplace/optee_qemuv8/optee_doc-3.22.0/CI/ConfInfer/CI-TA`

这里是唯一的源码修改入口，后续所有代码修改都应先落在这里。

当前按两种构建环境组织：

1. 本地独立交叉编译
   - 位置：当前目录
   - 目标：先在主机上编 `host` 静态库、后续再补本地实验构建链
2. OP-TEE example 工程编译
   - 位置：`/home/yyh/2.Programs/2.workplace/optee_qemuv8/optee_doc-3.22.0/optee_examples/CI-TA`
   - 目标：接入 OP-TEE 自己的 example 构建流程

## 目录说明

- `host/`
  - REE 侧 TEE client 代码
  - 产出 `libconfinfer_host.a`
- `ta/`
  - TEE 侧 TA 源码
  - `ta/include/confinfer_protocol.h` 是当前 CA/TA 共享协议头
- `toolchains/`
  - 本地独立交叉编译所需工具链说明和环境脚本
- `scripts/`
  - 构建脚本和同步脚本

## 当前原则

- 这里不保留无关模板文件
- 这里不保留 Android / CMake 兼容文件
- 这里尽量不保留生成产物
- 这里是源码主仓，`optee_examples/CI-TA` 不直接手工修改，由脚本同步

## 当前协议层状态

- 已新增 `TA_CONFINFER_CMD_EXEC_PARTITION`
- 已定义 `confinfer_partition_req_t`
- 已定义 `confinfer_layer_desc_t`
- 已定义 `confinfer_partition_rsp_t`
- host 侧已可通过 memref 发送一个 execution unit 的基础描述
- TA 侧当前只做协议校验和回包，不做真实算子执行
- bridge 安装与运行时装配逻辑已经移回 `CI-CA`，这里仅保留通信与协议

## 常用操作

构建 host 静态库：

```bash
bash CI/ConfInfer/CI-TA/scripts/build_hostlib.sh
```

同步源码到 `optee_examples/CI-TA`：

```bash
bash CI/ConfInfer/CI-TA/scripts/sync_to_optee_examples.sh
```

仅检查同步差异：

```bash
bash CI/ConfInfer/CI-TA/scripts/sync_to_optee_examples.sh --check
```
