#!/usr/bin/env bash

set -eu

SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)"
WORK_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/../../../.." && pwd)"

find_repo_root_from() {
    local start="$1"
    local dir=""

    dir="$(CDPATH= cd -- "$start" && pwd)"
    while [ "$dir" != "/" ]; do
        if [ -d "$dir/out-br/host/bin" ]; then
            printf '%s\n' "$dir"
            return 0
        fi
        dir="$(dirname "$dir")"
    done
    return 1
}

resolve_repo_root() {
    local root=""
    local default_repo_root=""

    default_repo_root="$WORK_ROOT/FVP/FVP-3.22.0"
    if [ -n "${CONFINFER_CI_TA_REPO_ROOT:-}" ]; then
        root="$(CDPATH= cd -- "$CONFINFER_CI_TA_REPO_ROOT" && pwd)"
    elif root="$(find_repo_root_from "$PWD")"; then
        :
    elif [ -d "$default_repo_root/out-br/host/bin" ]; then
        root="$default_repo_root"
    elif [ -d "$WORK_ROOT/out-br/host/bin" ]; then
        root="$WORK_ROOT"
    else
        echo "Unable to locate Buildroot host toolchain. Set CONFINFER_CI_TA_REPO_ROOT explicitly." >&2
        exit 1
    fi

    printf '%s\n' "$root"
}

REPO_ROOT="$(resolve_repo_root)"
HOST_DIR="$REPO_ROOT/out-br/host"
SYSROOT="$HOST_DIR/aarch64-buildroot-linux-gnu/sysroot"
BIN_DIR="$HOST_DIR/bin"

export CONFINFER_CI_TA_REPO_ROOT="$REPO_ROOT"
export CONFINFER_CI_TA_HOST_DIR="$HOST_DIR"
export CONFINFER_CI_TA_SYSROOT="$SYSROOT"
export CONFINFER_CI_TA_CROSS_COMPILE="$BIN_DIR/aarch64-linux-gnu-"
export CONFINFER_CI_TA_CC="$BIN_DIR/aarch64-linux-gnu-gcc"
export CONFINFER_CI_TA_CXX="$BIN_DIR/aarch64-linux-gnu-g++"
export CONFINFER_CI_TA_AR="$BIN_DIR/aarch64-linux-gnu-ar"
export CONFINFER_CI_TA_RANLIB="$BIN_DIR/aarch64-linux-gnu-ranlib"
export CONFINFER_CI_TA_STRIP="$BIN_DIR/aarch64-linux-gnu-strip"
export CONFINFER_CI_TA_TEEC_INCLUDE="$SYSROOT/usr/include"
export CONFINFER_CI_TA_TEEC_LIBDIR="$SYSROOT/usr/lib"
export CONFINFER_CI_TA_TEEC_EXPORT="$SYSROOT/usr"

echo "CONFINFER_CI_TA_HOST_DIR=$CONFINFER_CI_TA_HOST_DIR"
echo "CONFINFER_CI_TA_SYSROOT=$CONFINFER_CI_TA_SYSROOT"
echo "CONFINFER_CI_TA_CROSS_COMPILE=$CONFINFER_CI_TA_CROSS_COMPILE"
echo "CONFINFER_CI_TA_CC=$CONFINFER_CI_TA_CC"
echo "CONFINFER_CI_TA_CXX=$CONFINFER_CI_TA_CXX"
echo "CONFINFER_CI_TA_AR=$CONFINFER_CI_TA_AR"
echo "CONFINFER_CI_TA_RANLIB=$CONFINFER_CI_TA_RANLIB"
echo "CONFINFER_CI_TA_STRIP=$CONFINFER_CI_TA_STRIP"
echo "CONFINFER_CI_TA_TEEC_INCLUDE=$CONFINFER_CI_TA_TEEC_INCLUDE"
echo "CONFINFER_CI_TA_TEEC_LIBDIR=$CONFINFER_CI_TA_TEEC_LIBDIR"
echo "CONFINFER_CI_TA_TEEC_EXPORT=$CONFINFER_CI_TA_TEEC_EXPORT"
