import argparse
import json
import os
from collections import OrderedDict
from typing import Any, Dict

import torch

# python ConfInfer/tools/export_model_params.py \
#   --model-file ConfInfer/example/3-experimental/mobilenetv1/mobilenetv1_cifar100_best.pt \
#   --output-dir ConfInfer/example/3-experimental/mobilenetv1/export/params

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export PyTorch model parameters/buffers to ConfInfer binary exchange format."
    )
    parser.add_argument(
        "--model-file",
        required=True,
        help="Path to a PyTorch model file, such as TorchScript .pt or state_dict .pt/.pth",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write params.json and *.bin files",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64", "int64", "int32", "int8", "uint8"),
        default="float32",
        help="Export dtype",
    )
    parser.add_argument(
        "--strip-prefix",
        action="append",
        default=["module."],
        help="State-dict key prefix to strip. Can be passed multiple times.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Allow exporting an empty state_dict without error",
    )
    return parser.parse_args()


def normalize_state_dict_keys(state_dict: "OrderedDict[str, torch.Tensor]",
                              strip_prefixes: list[str]) -> "OrderedDict[str, torch.Tensor]":
    normalized = OrderedDict()
    for key, value in state_dict.items():
        new_key = key
        for prefix in strip_prefixes:
            if prefix and new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        normalized[new_key] = value
    return normalized


def is_state_dict_like(obj: Any) -> bool:
    if not isinstance(obj, dict) or not obj:
        return False
    first_value = next(iter(obj.values()))
    return torch.is_tensor(first_value)


def extract_state_dict(model_file: str) -> tuple[str, "OrderedDict[str, torch.Tensor]"]:
    try:
        module = torch.jit.load(model_file, map_location="cpu")
        return "torchscript", OrderedDict(module.state_dict())
    except Exception:
        pass

    checkpoint = torch.load(model_file, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, torch.nn.Module):
        return "nn.Module", OrderedDict(checkpoint.state_dict())

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            return "checkpoint.state_dict", OrderedDict(checkpoint["state_dict"])
        if "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
            return "checkpoint.model_state_dict", OrderedDict(checkpoint["model_state_dict"])
        if is_state_dict_like(checkpoint):
            return "plain_state_dict", OrderedDict(checkpoint)

    raise TypeError(f"Unsupported checkpoint format: {type(checkpoint)}")


def convert_param_dtype(param: torch.Tensor, dtype_name: str) -> torch.Tensor:
    mapping = {
        "float32": torch.float32,
        "float64": torch.float64,
        "int64": torch.int64,
        "int32": torch.int32,
        "int8": torch.int8,
        "uint8": torch.uint8,
    }
    return param.detach().cpu().contiguous().to(mapping[dtype_name])


def key_to_filename(key: str) -> str:
    safe = key.replace("/", "__")
    return f"{safe}.bin"


def export_state_dict(state_dict: "OrderedDict[str, torch.Tensor]",
                      output_dir: str,
                      export_dtype: str,
                      model_file: str,
                      source_format: str) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)

    meta: Dict[str, Any] = {
        "format_version": 1,
        "model_file": os.path.abspath(model_file),
        "source_format": source_format,
        "export_dtype": export_dtype,
        "param_count": 0,
        "params": {},
    }

    for key, param in state_dict.items():
        converted = convert_param_dtype(param, export_dtype)
        file_name = key_to_filename(key)
        file_path = os.path.join(output_dir, file_name)
        converted.numpy().tofile(file_path)

        meta["params"][key] = {
            "shape": list(converted.shape),
            "dtype": export_dtype,
            "file": file_name,
            "numel": int(converted.numel()),
            "bytes": int(converted.numel() * converted.element_size()),
        }
        meta["param_count"] += 1

    meta_path = os.path.join(output_dir, "params.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return meta


def main() -> int:
    args = parse_args()

    source_format, state_dict = extract_state_dict(args.model_file)
    state_dict = normalize_state_dict_keys(state_dict, args.strip_prefix)

    if not state_dict and not args.allow_empty:
        raise RuntimeError("state_dict is empty, refusing to export")

    meta = export_state_dict(
        state_dict=state_dict,
        output_dir=args.output_dir,
        export_dtype=args.dtype,
        model_file=args.model_file,
        source_format=source_format,
    )

    print(f"source_format: {source_format}")
    print(f"model_file: {os.path.abspath(args.model_file)}")
    print(f"output_dir: {os.path.abspath(args.output_dir)}")
    print(f"param_count: {meta['param_count']}")
    print("params.json written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
