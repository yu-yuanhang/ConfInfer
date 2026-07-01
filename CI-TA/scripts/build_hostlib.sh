#!/usr/bin/env bash

set -eu
# -e：命令出错就退出
# -u：用了未定义变量就报错退出

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
CI_TA_DIR="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"

. "$CI_TA_DIR/toolchains/buildroot-aarch64-env.sh" >/dev/null

make -C "$CI_TA_DIR/host" \
    clean lib info \
    CROSS_COMPILE="$CONFINFER_CI_TA_CROSS_COMPILE" \
    CC="$CONFINFER_CI_TA_CC" \
    AR="$CONFINFER_CI_TA_AR" \
    RANLIB="$CONFINFER_CI_TA_RANLIB" \
    TEEC_EXPORT="$CONFINFER_CI_TA_TEEC_EXPORT"
