#include <core/Graph.h>

namespace Kernel {
namespace core {

// 利用 Layer* _output; (graph 中的逻辑上输出节点作为输入)
// 逆向遍历构建图
Graph::Graph(Layer &layer):
    _layersNum(INVALID_VALUE_U),
    _layers(), _confLayers(),
    _inputL(nullptr), _outputL(&layer),
    _execOrder()
{
    EXIT_ERROR_CHECK_NE(0, layer._outputsL.size(),
                        "The input must be a logical output node");
    
    // DFS 从输出节点逆向收集所有 Layer
    std::unordered_set<Layer*> visited;
    dfs_collect(_outputL, visited);

    _layersNum = static_cast<UINT>(_layers.size());
    _inputL = find_input_node();

    buildExecutionOrder(_inputL);   // 这里检测了是否为 单入口 / 单出口
}

Graph::~Graph() {

}

Layer *Graph::operator[](UINT id) {
    if (id >= _execOrder.size()) { return nullptr; }
    return _execOrder[id];
}


void Graph::buildExecutionOrder(Layer *inputL) {
    EXIT_ERROR_CHECK_EQ(nullptr, inputL, "Input layer is null");
    _execOrder.clear();

    // 用于辅助检查 是否为 单出口
    // 因为 dfs_collect() 中是从 out 节点开始基于入度向前 深度优先遍历 计算图
    // 因此 _layers map 中的节点 总入度 <= 总出度 下面需要检查两个方面
    // 1. _layers 是否存在多个 入度为 0 的 layer (导致遍历不全)
    // 2. _layers 的 Layer 是否存在不合法的 出度 (出度所指的 Layer 不在 _Layers 中)

    // 构建 shadow 入度表
    std::unordered_map<Layer *, UINT> indegree;
    indegree.reserve(_layers.size());
    for (auto it = _layers.begin(); it != _layers.end(); ++it) {
        Layer *l = *it;
        indegree[l] = l->_inputsLNum;
    }

    // 这里基于 Graph 的设计逻辑 每个计算图 有且仅有一个入度为 0 的 inputL
    EXIT_ERROR_CHECK_NE(0, indegree[_inputL], 
        "Input layer indegree is not zero");
    List<Layer, false> readyQ;
    readyQ.push_back(_inputL);

    while (!readyQ.empty()) {
        Layer *cur = readyQ.pop_front();
        _execOrder.push_back(cur);

        for (auto it = cur->_outputsL.begin(); it != cur->_outputsL.end(); ++it) {
            Layer *next = *it;
            auto degIt = indegree.find(next);
            // 2. _layers 的 Layer 是否存在不合法的 出度 (出度所指的 Layer 不在 _Layers 中)
            EXIT_ERROR_CHECK_EQ(degIt, indegree.end(), 
            "Broken graph: output layer not in graph");

            UINT &deg = degIt->second;
            // EXIT_ERROR_CHECK_EQ(0, deg, 
            //     "Broken graph: indegree underflow");

            deg--;
            if (0 == deg) {
                readyQ.push_back(next);
                indegree.erase(next);
            }
        }
    }
    // 1. _layers 是否存在多个 入度为 0 的 layer (导致遍历不全)
    EXIT_ERROR_CHECK_NE(_execOrder.size(), _layers.size(),
            "Graph has cycle or unreachable layers");
    return;
}
UINT Graph::WorkspaceSize() {
    if (_execOrder.empty()) return 0;

    UINT wss = 0;
    for (auto it = _execOrder.begin(); it != _execOrder.end(); ++it) {
        if ((*it)->_workspaceSize > wss) wss = (*it)->_workspaceSize;
    }
    return wss;
}
bool Graph::splittable(UINT num) {
    for (auto it = _execOrder.begin(); it != _execOrder.end(); ++it) {
        if (!(*it)->sliceable(num)) return false;
    }
    return true;
}

// 从某个输出节点开始 深度优先遍历计算图
// 没有检查 单入口 和 单出口
void Graph::dfs_collect(Layer *cur, std::unordered_set<Layer *> &visited)
{   
    // cur 为空 或是 cur 节点已经被访问过
    if (!cur || visited.count(cur)) return;
    visited.insert(cur);

    // 遍历 inputs（逆向拓扑）
    for (auto it = cur->_inputsL.begin(); it != cur->_inputsL.end(); ++it) {
        dfs_collect(*it, visited);
    }
    // 收集自己
    _layers.insert(cur);
    if (cur->isInTEE()) _confLayers.insert(cur);
}

// =================== 找输入节点 ===================
Layer* Graph::find_input_node()
{
    Layer *input = nullptr;
    BOOL flag = false;
    for(std::unordered_set<Layer *>::iterator it = _layers.begin();
        it != _layers.end(); ++it) {
        if ((*it)->_inputsL.empty()) {
            // 因为 dfs_collect() 中是从 out 节点开始向前 深度优先遍历 计算图
            // 因此可能存在多个 入口节点
            // 这个后面 buildExecutionOrder() 中会进一步检测
            EXIT_ERROR_CHECK_EQ(true, flag, "Graph has multiple input nodes");
            flag = true;
            input = *it;
        }
    }
    EXIT_ERROR_CHECK_EQ(false, flag, "Graph has no input node");
    return input;
}


} // namespace end of core
} // namespace end of Kernel 