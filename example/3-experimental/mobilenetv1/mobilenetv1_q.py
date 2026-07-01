import argparse
import os
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader, Subset

from eval_mobilenetv1_cifar100 import CIFAR100Dataset, evaluate
from mobilenetv1 import MobileNetV1


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Evaluate MobileNetV1 CIFAR-100 float model and PTQ INT8 model."
    )
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(script_dir, "mobilenetv1_cifar100_best.pt"),
        help="Path to MobileNetV1 checkpoint (.pt/.pth or TorchScript).",
    )
    parser.add_argument(
        "--data-root",
        default=os.path.abspath(
            os.path.join(script_dir, "..", "dataset", "cifar100", "cifar-100-python")
        ),
        help="Path to CIFAR-100 python-format dataset directory.",
    )
    parser.add_argument(    # 用哪个数据集划分来做最终精度评估
        "--test-split",
        choices=("train", "test"),
        default="test",
        help="Dataset split used for accuracy evaluation.",
    )
    parser.add_argument(    # PTQ 校准使用哪个数据集划分
        "--calibration-split",
        choices=("train", "test"),
        default="train",
        help="Representative split used for PTQ calibration.",
    )
    parser.add_argument(    # 用于 PTQ 校准的样本数量
        "--calibration-samples",
        type=int,
        default=1024,
        help="Number of samples used for PTQ calibration. 0 means full calibration split.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--calibration-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(    # FP32 模型评估使用的设备
        "--float-device",
        default="cpu",
        # default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device used to evaluate the float model.",
    )
    parser.add_argument(    # PyTorch INT8 量化后端
        "--quant-backend",
        choices=("fbgemm", "qnnpack"),
        default="fbgemm",
        help="Quantized backend for PTQ.",
    )
    parser.add_argument(    # CIFAR-100 normalization 要和训练保持一致
        "--disable-normalize",
        action="store_true",
        help="Disable CIFAR-100 normalization.",
    )
    return parser.parse_args()


def extract_state_dict(checkpoint_path):
    try: # 先尝试加载为 TorchScript 模型
        scripted = torch.jit.load(checkpoint_path, map_location="cpu")
        return "torchscript", OrderedDict(scripted.state_dict())
    except Exception:
        pass

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, torch.nn.Module):
        return "nn.Module", OrderedDict(checkpoint.state_dict())
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return "checkpoint.state_dict", OrderedDict(checkpoint["state_dict"])
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return "checkpoint.model_state_dict", OrderedDict(checkpoint["model_state_dict"])
    if isinstance(checkpoint, dict):
        return "plain_state_dict", OrderedDict(checkpoint)

    raise TypeError(f"Unsupported checkpoint format: {type(checkpoint)}")


# 清理模型参数名字，去掉多 GPU 训练留下的 module. 前缀
def normalize_state_dict(state_dict):
    cleaned = OrderedDict()
    for key, value in state_dict.items():
        cleaned_key = key[7:] if key.startswith("module.") else key
        cleaned[cleaned_key] = value
    return cleaned


def load_float_model(checkpoint_path):
    source_format, state_dict = extract_state_dict(checkpoint_path)
    state_dict = normalize_state_dict(state_dict)

    # 加载原始的模型 也就是 FP32 模型
    model = MobileNetV1(num_classes=100, width_multiplier=1.0, dropout_rate=0.2)
    model.load_state_dict(state_dict)
    model.eval()
    return model, source_format


def make_loader(dataset, batch_size, num_workers, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )


def build_calibration_dataset(root, split, normalize, calibration_samples):
    dataset = CIFAR100Dataset(root=root, split=split, normalize=normalize)
    if calibration_samples > 0:
        calibration_samples = min(calibration_samples, len(dataset))
        dataset = Subset(dataset, range(calibration_samples))
    return dataset


def calibrate_model(prepared_model, calibration_loader):
    prepared_model.eval()
    with torch.no_grad():
        for images, _ in calibration_loader:
            prepared_model(images)


def quantize_model_fx(float_model, calibration_loader, backend):
    if not hasattr(torch, "ao") or not hasattr(torch.ao, "quantization"):
        raise RuntimeError("torch.ao.quantization is not available in this PyTorch build")

    try:
        from torch.ao.quantization import get_default_qconfig_mapping
        from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
    except Exception as exc:
        raise RuntimeError("FX graph mode quantization API is unavailable") from exc

    if backend not in torch.backends.quantized.supported_engines:
        raise RuntimeError(
            f"Unsupported quantized backend: {backend}. "
            f"supported={torch.backends.quantized.supported_engines}"
        )

    torch.backends.quantized.engine = backend

    model_cpu = load_model_copy_to_cpu(float_model)
    example_inputs = (torch.randn(1, 3, 32, 32),)
    qconfig_mapping = get_default_qconfig_mapping(backend)
    prepared = prepare_fx(model_cpu, qconfig_mapping, example_inputs=example_inputs)
    calibrate_model(prepared, calibration_loader)
    quantized = convert_fx(prepared)
    quantized.eval()
    return quantized


def load_model_copy_to_cpu(model):
    model_cpu = MobileNetV1(num_classes=100, width_multiplier=1.0, dropout_rate=0.2)
    model_cpu.load_state_dict(model.state_dict())
    model_cpu.eval()
    return model_cpu


def print_result_block(title, top1, top5, total, device_name):
    print(title)
    print(f"device: {device_name}")
    print(f"samples: {total}")
    print(f"top1: {top1:.4f}%")
    print(f"top5: {top5:.4f}%")


def main():
    args = parse_args()
    normalize = not args.disable_normalize

    test_dataset = CIFAR100Dataset(
        root=args.data_root,
        split=args.test_split,
        normalize=normalize,
    )
    calibration_dataset = build_calibration_dataset(
        root=args.data_root,
        split=args.calibration_split,
        normalize=normalize,
        calibration_samples=args.calibration_samples,
    )

    test_loader_float = make_loader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    test_loader_quant = make_loader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    calibration_loader = make_loader(
        calibration_dataset,
        batch_size=args.calibration_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    float_model, source_format = load_float_model(args.checkpoint)
    float_device = torch.device(args.float_device)
    float_model = float_model.to(float_device)

    float_top1, float_top5, total = evaluate(float_model, test_loader_float, float_device)

    quant_model = quantize_model_fx(float_model, calibration_loader, args.quant_backend)
    quant_top1, quant_top5, quant_total = evaluate(
        quant_model,
        test_loader_quant,
        torch.device("cpu"),
    )

    print(f"checkpoint: {args.checkpoint}")
    print(f"checkpoint_format: {source_format}")
    print(f"test_split: {args.test_split}")
    print(f"calibration_split: {args.calibration_split}")
    print(f"calibration_samples: {len(calibration_dataset)}")
    print(f"quant_backend: {args.quant_backend}")
    print_result_block("float_model", float_top1, float_top5, total, str(float_device))
    print_result_block("quantized_model", quant_top1, quant_top5, quant_total, "cpu")
    print("accuracy_delta")
    print(f"top1_delta: {quant_top1 - float_top1:.4f}%")
    print(f"top5_delta: {quant_top5 - float_top5:.4f}%")


if __name__ == "__main__":
    main()
