#ifndef __MOBILENETV1_RUNTIME_EXAMPLE_H__
#define __MOBILENETV1_RUNTIME_EXAMPLE_H__

#include <core/Network.h>

#include "mobilenetv1_model.h"

// 把模型对象、Graph、Network 放在一起，只是为了固定生命周期顺序。
// model 持有具体 Layer / Value，graph 引用 model 的输入输出，network 再引用 graph。
class MobileNetV1Runtime {
public:
    MobileNetV1Runtime()
        : model(),
          bindings(model.build_param_bindings()),
          graph(
              { Kernel::core::GraphInputSlot("input", model.graph_input()) },
              { Kernel::core::GraphOutputSlot("output", model.output_layer().output()) }
          ),
          network(graph, Kernel::core::RUNTIME) {}

    MobileNetV1Model model;
    Kernel::core::ParamBindingTable bindings;
    Kernel::core::Graph graph;
    Kernel::core::Network network;
};

#endif
