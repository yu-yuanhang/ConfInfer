import torch
import torch.nn as nn


class MobileNetV1(nn.Module):
    """
    CIFAR-friendly MobileNetV1

    设计目标：
    1. 保留 MobileNetV1 的 depthwise separable conv 风格
    2. 保留 self.ops / flag / ratio 划分等接口，服务后续窃取攻击实验
    3. 针对 CIFAR-100 (32x32) 做更稳的结构适配
    4. 降低尾部过重带来的过拟合风险
    """

    def __init__(self, num_classes=100, width_multiplier=1.0, dropout_rate=0.2):
        super(MobileNetV1, self).__init__()

        self.num_classes = num_classes
        self.width_multiplier = width_multiplier
        self.dropout_rate = dropout_rate

        self.ops = nn.ModuleList()
        self.op_names = []
        self.op_types = []

        in_channels = 3

        # ================= Backbone =================
        # CIFAR 输入 32x32，因此 stem 不下采样
        # 整体采用 4 次 stage，末端控制在 512 通道，不再额外加宽尾部

        # Stem: 32x32 -> 32x32
        out_channels = self._make_divisible(32 * width_multiplier)
        self._add_conv_bn_act(in_channels, out_channels, stride=1, block_id="stem")
        in_channels = out_channels

        # Stage 1: 32x32 -> 32x32
        out_channels = self._make_divisible(64 * width_multiplier)
        self._add_depthwise_separable_conv(in_channels, out_channels, stride=1, block_id="1_0")
        in_channels = out_channels

        # Stage 2: 32x32 -> 16x16
        out_channels = self._make_divisible(128 * width_multiplier)
        self._add_depthwise_separable_conv(in_channels, out_channels, stride=2, block_id="2_0")
        self._add_depthwise_separable_conv(out_channels, out_channels, stride=1, block_id="2_1")
        in_channels = out_channels

        # Stage 3: 16x16 -> 8x8
        out_channels = self._make_divisible(256 * width_multiplier)
        self._add_depthwise_separable_conv(in_channels, out_channels, stride=2, block_id="3_0")
        self._add_depthwise_separable_conv(out_channels, out_channels, stride=1, block_id="3_1")
        self._add_depthwise_separable_conv(out_channels, out_channels, stride=1, block_id="3_2")
        in_channels = out_channels

        # Stage 4: 8x8 -> 4x4
        out_channels = self._make_divisible(512 * width_multiplier)
        self._add_depthwise_separable_conv(in_channels, out_channels, stride=2, block_id="4_0")
        self._add_depthwise_separable_conv(out_channels, out_channels, stride=1, block_id="4_1")
        self._add_depthwise_separable_conv(out_channels, out_channels, stride=1, block_id="4_2")
        self._add_depthwise_separable_conv(out_channels, out_channels, stride=1, block_id="4_3")
        in_channels = out_channels

        # ================= Head =================
        self._add_op("avgpool", nn.AdaptiveAvgPool2d(1), "avgpool")
        self._add_op("flatten", nn.Flatten(1), "flatten")
        self._add_op("dropout", nn.Dropout(self.dropout_rate), "dropout")
        self._add_op("classifier", nn.Linear(in_channels, num_classes), "linear")

        self._initialize_weights()

    # ==================== 构建辅助函数 ====================

    def _make_divisible(self, value, divisor=8):
        value = int(value + divisor / 2) // divisor * divisor
        return max(divisor, value)

    def _add_op(self, name, module, op_type):
        module.flag = None
        self.ops.append(module)
        self.op_names.append(name)
        self.op_types.append(op_type)

    def _add_conv_bn_act(self, in_channels, out_channels, stride, block_id):
        self._add_op(
            f"conv_{block_id}",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            "conv",
        )
        self._add_op(f"bn_{block_id}", nn.BatchNorm2d(out_channels), "bn")
        self._add_op(f"act_{block_id}", nn.ReLU(inplace=True), "relu")

    def _add_depthwise_separable_conv(self, in_channels, out_channels, stride, block_id):
        # depthwise
        self._add_op(
            f"dw_conv_{block_id}",
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            "depthwise_conv",
        )
        self._add_op(f"dw_bn_{block_id}", nn.BatchNorm2d(in_channels), "bn")
        self._add_op(f"dw_act_{block_id}", nn.ReLU(inplace=True), "relu")

        # pointwise
        self._add_op(
            f"pw_conv_{block_id}",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            "pointwise_conv",
        )
        self._add_op(f"pw_bn_{block_id}", nn.BatchNorm2d(out_channels), "bn")
        self._add_op(f"pw_act_{block_id}", nn.ReLU(inplace=True), "relu")

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    # ==================== 前向传播 ====================

    def forward(self, x):
        for op in self.ops:
            x = op(x)
        return x

    # ==================== 信息查看接口 ====================

    def get_op_info(self):
        info = []
        for i, (name, op_type, op) in enumerate(zip(self.op_names, self.op_types, self.ops)):
            params = sum(p.numel() for p in op.parameters())
            flagged = getattr(op, "flag", None)
            info.append(
                {
                    "index": i,
                    "name": name,
                    "type": op_type,
                    "params": params,
                    "flag": flagged,
                }
            )
        return info

    def print_op_summary(self):
        total_params = 0
        print("=" * 100)
        print(f"{'idx':<5}{'name':<24}{'type':<20}{'params':<15}{'flag':<10}")
        print("=" * 100)
        for i, (name, op_type, op) in enumerate(zip(self.op_names, self.op_types, self.ops)):
            params = sum(p.numel() for p in op.parameters())
            total_params += params
            print(f"{i:<5}{name:<24}{op_type:<20}{params:<15}{str(op.flag):<10}")
        print("=" * 100)
        print(f"Total params: {total_params}")
        print("=" * 100)

    # ==================== Flag 管理接口 ====================

    def set_flags_by_ratio(self, ratio, protect_from="front", flag_value=True):
        """按参数量比例设置连续的前部/后部保护，并尽量贴近目标比例。"""
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("ratio 必须在 [0, 1] 范围内")

        for op in self.ops:
            op.flag = None

        op_params = []
        for i, op in enumerate(self.ops):
            params = sum(p.numel() for p in op.parameters())
            op_params.append((i, params))

        total_params = sum(p for _, p in op_params)
        if total_params == 0 or ratio == 0.0:
            return 0.0, []

        target_params = total_params * ratio

        if protect_from == "front":
            ordered_ops = op_params
        elif protect_from == "back":
            ordered_ops = list(reversed(op_params))
        else:
            raise ValueError("protect_from 必须是 'front' 或 'back'")

        accumulated = 0
        best_cut = 0
        best_error = abs(target_params)

        for cut, (_, params) in enumerate(ordered_ops, start=1):
            accumulated += params
            current_error = abs(accumulated - target_params)
            if current_error < best_error:
                best_error = current_error
                best_cut = cut

        protected_indices = [idx for idx, _ in ordered_ops[:best_cut]]
        for idx in protected_indices:
            self.ops[idx].flag = flag_value

        protected_params = sum(params for _, params in ordered_ops[:best_cut])
        actual_ratio = protected_params / total_params
        return actual_ratio, protected_indices

    def set_flags_by_indices(self, indices, flag_value=True, reset=True):
        if reset:
            for op in self.ops:
                op.flag = None
        for idx in indices:
            if idx < 0 or idx >= len(self.ops):
                raise IndexError(f"索引越界: {idx}")
            self.ops[idx].flag = flag_value

    def set_flags_by_range(self, start_idx, end_idx, flag_value=True, reset=True):
        if start_idx > end_idx:
            raise ValueError("区间非法")
        if start_idx < 0 or end_idx >= len(self.ops):
            raise IndexError("区间越界")
        if reset:
            for op in self.ops:
                op.flag = None
        for idx in range(start_idx, end_idx + 1):
            self.ops[idx].flag = flag_value

    def get_flagged_indices(self):
        return [i for i, op in enumerate(self.ops) if op.flag is not None]

    def get_whitebox_indices(self):
        return [i for i, op in enumerate(self.ops) if op.flag is None]

    def count_parameters_by_flag(self):
        flag_none = 0
        flag_true = 0
        for op in self.ops:
            params = sum(p.numel() for p in op.parameters())
            if op.flag is None:
                flag_none += params
            else:
                flag_true += params
        return flag_none, flag_true, flag_none + flag_true

    def get_whitebox_state_dict(self):
        state_dict = {}
        for i, op in enumerate(self.ops):
            if op.flag is None:
                for name, param in op.named_parameters():
                    state_dict[f"ops.{i}.{name}"] = param.data.clone()
        return state_dict

    def get_blackbox_state_dict(self):
        state_dict = {}
        for i, op in enumerate(self.ops):
            if op.flag is not None:
                for name, param in op.named_parameters():
                    state_dict[f"ops.{i}.{name}"] = param.data.clone()
        return state_dict

    def init_substitute_from_target(self, substitute_model):
        """从当前模型（目标模型）提取白盒权重初始化替代模型。"""
        whitebox_indices = substitute_model.get_whitebox_indices()
        state_dict = {}

        for idx in whitebox_indices:
            target_op = self.ops[idx]
            for name, param in target_op.named_parameters():
                state_dict[f"ops.{idx}.{name}"] = param.data.clone()

        substitute_model.load_state_dict(state_dict, strict=False)
        return substitute_model

    def get_whitebox_param_count(self):
        count = 0
        for op in self.ops:
            if op.flag is None:
                count += sum(p.numel() for p in op.parameters())
        return count

    def get_blackbox_param_count(self):
        count = 0
        for op in self.ops:
            if op.flag is not None:
                count += sum(p.numel() for p in op.parameters())
        return count


if __name__ == "__main__":
    model = MobileNetV1(num_classes=100, width_multiplier=1.0, dropout_rate=0.2)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print("Output shape:", y.shape)
    model.print_op_summary()

    actual_ratio, protected = model.set_flags_by_ratio(0.3, protect_from="back", flag_value=True)
    print(f"Protected ratio: {actual_ratio:.4f}")
    print(f"Protected indices: {protected}")
    print("Whitebox params / Blackbox params / Total params:", model.count_parameters_by_flag())
