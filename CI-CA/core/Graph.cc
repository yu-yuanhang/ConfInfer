#include <core/Graph.h>
#include <core/BoundaryLayer.h>

namespace Kernel {
namespace core {

// 匿名命名空间 这里也没有必要写到通用工具函数中
namespace {
bool has_input(const std::vector<GraphInputSlot>& inputs, Value_t* v) {
    for (auto it = inputs.begin(); it != inputs.end(); ++it) {
        if (it->value == v) {
            return true;
        }
    }
    return false;
}

bool has_output_name_dup(const std::vector<GraphOutputSlot>& outputs) {
    std::unordered_set<std::string> names;
    names.reserve(outputs.size());
    for (auto it = outputs.begin(); it != outputs.end(); ++it) {
        if (!names.insert(it->name).second) {
            return true;
        }
    }
    return false;
}
} // namespace

Graph::Graph():
    _sig(),
    _inputBoundary(nullptr), _outputBoundary(nullptr),
    _layersNum(INVALID_VALUE_U),
    _layers(), _confLayers(),
    _execOrder() {}

Graph::Graph(const GraphSignature& sig):
    Graph() {
    // 这里的拷贝构造用的是默认的 
    // Data 数据本质上是 浅拷贝
    _sig = sig;
    build();
}

Graph::Graph(std::initializer_list<GraphInputSlot> inputs,
             std::initializer_list<GraphOutputSlot> outputs):
    Graph() {
    for (auto it = inputs.begin(); it != inputs.end(); ++it) {
        _sig.inputs.push_back(*it);
    }
    for (auto it = outputs.begin(); it != outputs.end(); ++it) {
        _sig.outputs.push_back(*it);
    }
    build();
}

void Graph::addInput(const std::string& name, Value_t& value) {
    _sig.inputs.emplace_back(name, value);
}

void Graph::addOutput(const std::string& name, Value_t& value) {
    _sig.outputs.emplace_back(name, value);
}

// 至此之前 Layers 之间的依赖关系已经存在了
void Graph::build() {
    // 这里是防止重复构建
    EXIT_ERROR_CHECK_NE(nullptr, _inputBoundary, "Graph has already been built");
    EXIT_ERROR_CHECK_NE(nullptr, _outputBoundary, "Graph has already been built");

    // 正式插入 boundary layer 之前 先校验并收集图结构
    checkSig();     // 检查 GraphSignature 本身是否合法
    collectOuts();  // 从 graph outputs 反向 DFS 把所有能到达这些 outputs 的 layer 收集进 _layers
    checkInputs();

    // 创建输入输出的边界层
    std::vector<Value_t*> input_vals;
    input_vals.reserve(_sig.inputs.size());
    for (auto it = _sig.inputs.begin(); it != _sig.inputs.end(); ++it) {
        input_vals.push_back(it->value);
    }
    _inputBoundary = new GraphInputLayer(input_vals);

    std::vector<Value_t*> output_vals;
    output_vals.reserve(_sig.outputs.size());
    for (auto it = _sig.outputs.begin(); it != _sig.outputs.end(); ++it) {
        output_vals.push_back(it->value);
    }
    _outputBoundary = new GraphOutputLayer(output_vals);

    wireIns();
    _layers.insert(_inputBoundary);
    _layers.insert(_outputBoundary);
    rebuildLinks();

    _layersNum = static_cast<UINT>(_layers.size());

    // 从输入边界层开始做拓扑排序 生成 _execOrder
    buildExecutionOrder(_inputBoundary);
}

Graph::~Graph() {
    if (_inputBoundary) {
        std::unordered_map<Value_t*, Value_t*> remap;
        remap.reserve(_sig.inputs.size());

        for (UINT i = 0; i < _sig.inputs.size(); ++i) {
            remap[&_inputBoundary->output(i)] = _sig.inputs[i].value;
        }

        for (auto layer_it = _layers.begin(); layer_it != _layers.end(); ++layer_it) {
            Layer* l = *layer_it;
            if (l == _inputBoundary || l == _outputBoundary) {
                continue;
            }
            for (auto in_it = l->_inputs.begin(); in_it != l->_inputs.end(); ++in_it) {
                auto it = remap.find(*in_it);
                if (it != remap.end()) {
                    *in_it = it->second;
                }
            }
        }
    }

    _layers.erase(_inputBoundary);
    _layers.erase(_outputBoundary);
    if (!_layers.empty()) {
        rebuildLinks();
    }
    _execOrder.clear();
    _layersNum = static_cast<UINT>(_layers.size());

    delete _inputBoundary;
    _inputBoundary = nullptr;
    delete _outputBoundary;
    _outputBoundary = nullptr;
}

void Graph::checkSig() {
    EXIT_ERROR_CHECK_EQ(true, _sig.outputs.empty(), "Graph outputs must not be empty");
    EXIT_ERROR_CHECK_EQ(true, has_output_name_dup(_sig.outputs), "Graph output names must be unique");

    for (auto it = _sig.outputs.begin(); it != _sig.outputs.end(); ++it) {
        EXIT_ERROR_CHECK_EQ(nullptr, it->value, "Graph output value is nullptr");
        // 这里存在一种特殊的情况 Value_t 同时是 input output 
        // 这时 output 就不存在 producer 的情况 
        // 但是当前也用不到 就之后遇到了再说吧
        EXIT_ERROR_CHECK_EQ(nullptr, it->value->producer, "Graph output must have a producer layer");
    }

    for (auto it = _sig.inputs.begin(); it != _sig.inputs.end(); ++it) {
        EXIT_ERROR_CHECK_EQ(nullptr, it->value, "Graph input value is nullptr");
    }
}

void Graph::collectOuts() {
    std::unordered_set<Layer*> visited;
    for (auto it = _sig.outputs.begin(); it != _sig.outputs.end(); ++it) {
        dfs_collect(it->value->producer, visited);
    }
}

void Graph::checkInputs() {
    // 主要检查输入不完整的情况 
    // graph 是否还存在 其他没有被 GraphSignature 记录的输入 
    for (auto layer_it = _layers.begin(); layer_it != _layers.end(); ++layer_it) {
        Layer* l = *layer_it;
        for (auto value_it = l->_inputs.begin(); value_it != l->_inputs.end(); ++value_it) {
            Value_t* v = *value_it;
            if (nullptr == v->producer) {
                EXIT_ERROR_CHECK_EQ(false, has_input(_sig.inputs, v),
                    "Found external input Value not declared in GraphSignature");
            }
        }
    }

    for (auto it = _sig.outputs.begin(); it != _sig.outputs.end(); ++it) {
        EXIT_ERROR_CHECK_EQ(
            false,
            _layers.count(it->value->producer) > 0,
            "Graph output producer is not in graph"
        );
    }

    // 检查输入 unused 的情况
    for (auto it = _sig.inputs.begin(); it != _sig.inputs.end(); ++it) {
        bool used = false;
        for (auto layer_it = _layers.begin(); layer_it != _layers.end(); ++layer_it) {
            Layer* l = *layer_it;
            for (auto value_it = l->_inputs.begin(); value_it != l->_inputs.end(); ++value_it) {
                if (*value_it == it->value) {
                    used = true; break;
                }
            }
            if (used) { break; }
        }
        EXIT_ERROR_CHECK_EQ(false, used, "Declared graph input is not used");
    }
}

// 绑定输入数据 依赖关系
void Graph::wireIns() {
    std::unordered_map<Value_t*, Value_t*> remap;
    remap.reserve(_sig.inputs.size());
    for (UINT i = 0; i < _sig.inputs.size(); ++i) {
        remap[_sig.inputs[i].value] = &_inputBoundary->output(i);
    }

    for (auto layer_it = _layers.begin(); layer_it != _layers.end(); ++layer_it) {
        Layer* l = *layer_it;
        for (auto in_it = l->_inputs.begin(); in_it != l->_inputs.end(); ++in_it) {
            auto it = remap.find(*in_it);
            if (it != remap.end()) {
                *in_it = it->second;
            }
        }
    }
}

void Graph::rebuildLinks() {
    for (auto layer_it = _layers.begin(); layer_it != _layers.end(); ++layer_it) {
        Layer* l = *layer_it;
        while (!l->_inputsL.empty()) {
            l->_inputsL.erase_front();
        }
        while (!l->_outputsL.empty()) {
            l->_outputsL.erase_front();
        }
        l->_inputsLNum = 0;
        l->_outputsLNum = 0;
    }

    for (auto it = _sig.inputs.begin(); it != _sig.inputs.end(); ++it) {
        if (it->value) {
            it->value->consumers.clear();
        }
    }
    for (auto layer_it = _layers.begin(); layer_it != _layers.end(); ++layer_it) {
        Layer* l = *layer_it;
        for (auto out_it = l->_outputs.begin(); out_it != l->_outputs.end(); ++out_it) {
            (*out_it)->consumers.clear();
        }
    }

    for (auto layer_it = _layers.begin(); layer_it != _layers.end(); ++layer_it) {
        Layer* l = *layer_it;
        for (auto value_it = l->_inputs.begin(); value_it != l->_inputs.end(); ++value_it) {
            Value_t* v = *value_it;
            if (!v) {
                continue;
            }
            if (v->producer) {
                if (!l->_inputsL.contains(v->producer)) {
                    l->_inputsL.push_back(v->producer);
                    ++l->_inputsLNum;
                }
                if (!v->producer->_outputsL.contains(l)) {
                    v->producer->_outputsL.push_back(l);
                    ++v->producer->_outputsLNum;
                }
            }
            v->consumers.push_back(l);
        }
    }
}

Layer *Graph::operator[](UINT id) {
    if (id >= _execOrder.size()) { return nullptr; }
    return _execOrder[id];
}

std::vector<Layer *> Graph::prevs(const Layer *layer) const {
    EXIT_ERROR_CHECK_EQ(nullptr, layer, "Layer is nullptr");

    std::vector<Layer *> result;
    result.reserve(layer->_inputsLNum);
    Layer *mutable_layer = const_cast<Layer *>(layer);
    for (auto it = mutable_layer->_inputsL.begin(); it != mutable_layer->_inputsL.end(); ++it) {
        result.push_back(*it);
    }
    return result;
}

std::vector<Layer *> Graph::nexts(const Layer *layer) const {
    EXIT_ERROR_CHECK_EQ(nullptr, layer, "Layer is nullptr");

    std::vector<Layer *> result;
    result.reserve(layer->_outputsLNum);
    Layer *mutable_layer = const_cast<Layer *>(layer);
    for (auto it = mutable_layer->_outputsL.begin(); it != mutable_layer->_outputsL.end(); ++it) {
        result.push_back(*it);
    }
    return result;
}

std::vector<Value_t *> Graph::ins(const Layer *layer) const {
    EXIT_ERROR_CHECK_EQ(nullptr, layer, "Layer is nullptr");

    std::vector<Value_t *> result;
    result.reserve(layer->_inputs.size());
    for (auto it = layer->_inputs.begin(); it != layer->_inputs.end(); ++it) {
        result.push_back(*it);
    }
    return result;
}

std::vector<Value_t *> Graph::outs(const Layer *layer) const {
    EXIT_ERROR_CHECK_EQ(nullptr, layer, "Layer is nullptr");

    std::vector<Value_t *> result;
    result.reserve(layer->_outputs.size());
    for (auto it = layer->_outputs.begin(); it != layer->_outputs.end(); ++it) {
        result.push_back(it->get());
    }
    return result;
}

bool Graph::isGraphInputValue(const Value_t *value) const {
    if (nullptr == value) {
        return false;
    }
    for (auto it = _sig.inputs.begin(); it != _sig.inputs.end(); ++it) {
        if (it->value == value) {
            return true;
        }
    }
    if (_inputBoundary) {
        for (UINT i = 0; i < _inputBoundary->outputNum(); ++i) {
            if (&_inputBoundary->output(i) == value) {
                return true;
            }
        }
    }
    return false;
}

bool Graph::isGraphOutputValue(const Value_t *value) const {
    if (nullptr == value) {
        return false;
    }
    for (auto it = _sig.outputs.begin(); it != _sig.outputs.end(); ++it) {
        if (it->value == value) {
            return true;
        }
    }
    if (_outputBoundary) {
        for (UINT i = 0; i < _outputBoundary->inputNum(); ++i) {
            if (&_outputBoundary->input(i) == value) {
                return true;
            }
        }
    }
    return false;
}

bool Graph::valueCrossesDomain(const Value_t *value) const {
    if (nullptr == value) {
        return false;
    }

    const Layer *producer = value->producer;
    if (nullptr == producer) {
        return false;
    }

    const ExecutionDomain prod_domain = producer->execDomain();
    for (auto it = value->consumers.begin(); it != value->consumers.end(); ++it) {
        const Layer *consumer = *it;
        if (nullptr == consumer) {
            continue;
        }
        if (consumer->execDomain() != prod_domain) {
            return true;
        }
    }
    return false;
}

void Graph::buildExecutionOrder(Layer *inputL) {
    EXIT_ERROR_CHECK_EQ(nullptr, inputL, "Input layer is null");
    _execOrder.clear();

    std::unordered_map<Layer *, UINT> indegree;
    indegree.reserve(_layers.size());
    for (auto it = _layers.begin(); it != _layers.end(); ++it) {
        Layer *l = *it;
        indegree[l] = l->_inputsLNum;
    }

    EXIT_ERROR_CHECK_NE(0, indegree[inputL], "Input layer indegree is not zero");

    List<Layer, false> readyQ;
    readyQ.push_back(inputL);

    while (!readyQ.empty()) {
        Layer *cur = readyQ.pop_front();
        _execOrder.push_back(cur);

        for (auto it = cur->_outputsL.begin(); it != cur->_outputsL.end(); ++it) {
            Layer *next = *it;
            auto degIt = indegree.find(next);
            EXIT_ERROR_CHECK_EQ(degIt, indegree.end(), "Broken graph: output layer not in graph");

            UINT &deg = degIt->second;
            deg--;
            if (0 == deg) {
                readyQ.push_back(next);
                indegree.erase(next);
            }
        }
    }

    EXIT_ERROR_CHECK_NE(_execOrder.size(), _layers.size(), "Graph has cycle or unreachable layers");
}

UINT Graph::WorkspaceSize() {
    if (_execOrder.empty()) return 0;

    UINT wss = 0;
    for (auto it = _execOrder.begin(); it != _execOrder.end(); ++it) {
        if ((*it)->_workspaceSize > wss) wss = (*it)->_workspaceSize;
    }
    return wss;
}

void Graph::dfs_collect(Layer *cur, std::unordered_set<Layer *> &visited) {
    if (!cur || visited.count(cur)) return;
    visited.insert(cur);

    for (auto it = cur->_inputsL.begin(); it != cur->_inputsL.end(); ++it) {
        dfs_collect(*it, visited);
    }
    _layers.insert(cur);
    if (cur->isInTEE()) _confLayers.insert(cur);
}

} // namespace core
} // namespace Kernel
