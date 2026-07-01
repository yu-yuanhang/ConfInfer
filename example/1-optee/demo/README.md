# 1-optee Demo

## 目标

这个目录用于生成一个最小 REE 侧测试程序：

- 构建 `CI-CA` 的 aarch64 版本 `libConfInfer.so`
- 通过 `ENABLE_TEE_BRIDGE=1` 自动挂载 TEE bridge
- 生成可放入 rootfs 的可执行文件

当前测试程序会做一个最小 8 层网络测试：

1. 前 4 层保持在 `REE`
2. 后 4 层标记为 `TEE`
3. `network.prepare()` 后检查分区结果是否为 `REE + TEE`
4. `network.run()` 时走 `CI-CA -> TEE bridge -> CI-TA protocol`
5. 当前重点是验证通信链路和执行分区，不验证真实 TEE 算子数值

## 构建

在仓库根目录执行：

```bash
make -C CI/ConfInfer/example/1-optee/demo
```

构建产物：

- `.codex_build/1-optee-demo/confinfer_demo`

## 依赖

- Buildroot aarch64 toolchain
- `libteec.so`
- `/tmp/confinfer_ci_ca_build_optee/libConfInfer.so`

## 说明

- `make` 时会先执行 `CI/ConfInfer/CI-TA/scripts/sync_to_optee_examples.sh`
- `make` 时会自动在 `/tmp/confinfer_ci_ca_build_optee` 下交叉编译 `ENABLE_TEE_BRIDGE=1` 的 `libConfInfer.so`
- 由于 `CI/ConfInfer/example/1-optee/demo` 在当前环境下映射到外部源码目录，实际编译产物放在仓库根目录的 `.codex_build/1-optee-demo/`
