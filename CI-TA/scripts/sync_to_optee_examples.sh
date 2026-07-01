#!/usr/bin/env bash

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
CI_TA_DIR="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(CDPATH= cd -- "$CI_TA_DIR/../../.." && pwd)"
TARGET_DIR="$REPO_ROOT/optee_examples/CI-TA"

usage() {
    cat <<'EOF'
Usage:
  bash CI/ConfInfer/CI-TA/scripts/sync_to_optee_examples.sh [--check]

Description:
  Source of truth:  CI/ConfInfer/CI-TA
  Sync target:      optee_examples/CI-TA

Options:
  --check   only print the source/target paths and compare the tracked sync set
EOF
}

CHECK_ONLY=0
if [ $# -gt 1 ]; then
    usage
    exit 1
fi
if [ $# -eq 1 ]; then
    case "$1" in
        --check)
            CHECK_ONLY=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage
            exit 1
            ;;
    esac
fi

TOP_FILES=(
    "README.md"
    "Makefile"
    "Android.mk"
    "CMakeLists.txt"
)

SYNC_DIRS=(
    "host"
    "ta"
)

echo "CI-TA source dir : $CI_TA_DIR"
echo "Sync target dir  : $TARGET_DIR"

sync_one() {
    local rel="$1"
    local src="$CI_TA_DIR/$rel"
    local dst="$TARGET_DIR/$rel"

    if [ ! -f "$src" ]; then
        echo "Missing source file: $src" >&2
        exit 1
    fi

    if [ "$CHECK_ONLY" -eq 1 ]; then
        if [ ! -f "$dst" ]; then
            echo "[missing] $rel"
            return
        fi
        if cmp -s "$src" "$dst"; then
            echo "[same]    $rel"
        else
            echo "[diff]    $rel"
        fi
        return
    fi

    mkdir -p "$(dirname "$dst")"
    cp -f "$src" "$dst"
    echo "[sync]    $rel"
}

collect_dir_files() {
    local dir="$1"
    (
        cd "$CI_TA_DIR/$dir"
        find . -type f ! -path './build/*' ! -path './out/*' | sort
    ) | sed "s#^\./#$dir/#"
}

collect_all_files() {
    local rel
    for rel in "${TOP_FILES[@]}"; do
        printf '%s\n' "$rel"
    done
    local dir
    for dir in "${SYNC_DIRS[@]}"; do
        collect_dir_files "$dir"
    done
}

report_stale_files() {
    local expected_list actual_list rel
    expected_list="$(mktemp)"
    actual_list="$(mktemp)"

    trap 'rm -f "$expected_list" "$actual_list"' RETURN

    collect_all_files | sort > "$expected_list"
    (
        cd "$TARGET_DIR"
        find . -type f | sort
    ) | sed 's#^\./##' > "$actual_list"

    while IFS= read -r rel; do
        [ -n "$rel" ] || continue
        if ! grep -Fxq "$rel" "$expected_list"; then
            echo "[stale]   $rel"
        fi
    done < "$actual_list"
}

for rel in "${TOP_FILES[@]}"; do
    sync_one "$rel"
done

for dir in "${SYNC_DIRS[@]}"; do
    while IFS= read -r rel; do
        [ -n "$rel" ] || continue
        sync_one "$rel"
    done <<EOF
$(collect_dir_files "$dir")
EOF
done

report_stale_files

if [ "$CHECK_ONLY" -eq 1 ]; then
    echo "CI-TA sync check completed."
else
    echo "CI-TA source sync completed."
fi
