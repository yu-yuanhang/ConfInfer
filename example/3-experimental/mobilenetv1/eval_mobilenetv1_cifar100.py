import argparse
import os
import pickle

import torch
from torch.utils.data import DataLoader, Dataset

from mobilenetv1 import MobileNetV1


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


class CIFAR100Dataset(Dataset):
    def __init__(self, root, split="train", normalize=True):
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")

        path = os.path.join(root, split)
        with open(path, "rb") as f:
            payload = pickle.load(f, encoding="latin1")

        self.images = payload["data"]
        self.labels = payload["fine_labels"]
        self.normalize = normalize
        self.mean = torch.tensor(CIFAR100_MEAN, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(CIFAR100_STD, dtype=torch.float32).view(3, 1, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = torch.tensor(self.images[index], dtype=torch.float32).view(3, 32, 32) / 255.0
        if self.normalize:
            image = (image - self.mean) / self.std
        label = int(self.labels[index])
        return image, label


def load_model(checkpoint_path, device):
    try:
        model = torch.jit.load(checkpoint_path, map_location=device)
        model.eval()
        print("load mode: torchscript")
        return model
    except Exception:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = MobileNetV1(num_classes=100, width_multiplier=1.0, dropout_rate=0.2).to(device)

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        cleaned_state_dict = {}
        for key, value in state_dict.items():
            cleaned_key = key[7:] if key.startswith("module.") else key
            cleaned_state_dict[cleaned_key] = value

        model.load_state_dict(cleaned_state_dict)
        model.eval()
        print("load mode: state_dict")
        return model


def evaluate(model, loader, device):
    total = 0
    correct1 = 0
    correct5 = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            _, pred1 = logits.topk(1, dim=1)
            _, pred5 = logits.topk(5, dim=1)

            total += labels.size(0)
            correct1 += pred1.squeeze(1).eq(labels).sum().item()
            correct5 += pred5.eq(labels.view(-1, 1)).any(dim=1).sum().item()

    top1 = 100.0 * correct1 / total
    top5 = 100.0 * correct5 / total
    return top1, top5, total


def print_logits(model, dataset, device, sample_index, split):
    if sample_index < 0 or sample_index >= len(dataset):
        raise ValueError(f"sample_index out of range: {sample_index}")

    image, label = dataset[sample_index]
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image).detach().cpu().view(-1)

    logits_text = ", ".join(f"{value.item():.9g}" for value in logits)

    print("logits ok")
    print(f"split: {split}")
    print(f"sample_index: {sample_index}")
    print(f"label: {label}")
    print(f"logits: [{logits_text}]")


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.path.join(script_dir, "mobilenetv1_cifar100_best.pt"))
    parser.add_argument("--data-root", default=os.path.abspath(os.path.join(script_dir, "..", "dataset", "cifar100", "cifar-100-python")))
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--disable-normalize", action="store_true")
    parser.add_argument("--print-logits", action="store_true")
    parser.add_argument("--sample-index", type=int, default=0)
    return parser.parse_args()

# python eval_mobilenetv1_cifar100.py --print-logits --sample-index 0
def main():
    args = parse_args()
    device = torch.device(args.device)

    dataset = CIFAR100Dataset(
        root=args.data_root,
        split=args.split,
        normalize=not args.disable_normalize,
    )
    model = load_model(args.checkpoint, device).to(device)

    if args.print_logits:
        print(f"checkpoint: {args.checkpoint}")
        print(f"split: {args.split}")
        print_logits(model, dataset, device, args.sample_index, args.split)
        return

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    top1, top5, total = evaluate(model, loader, device)

    print(f"checkpoint: {args.checkpoint}")
    print(f"split: {args.split}")
    print(f"samples: {total}")
    print(f"top1: {top1:.4f}%")
    print(f"top5: {top5:.4f}%")


if __name__ == "__main__":
    main()
