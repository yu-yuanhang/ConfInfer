#ifndef __GRAPH_H_CA__
#define __GRAPH_H_CA__

#include <core/Layer.h>

namespace Kernel {
namespace core {

// struct RuntimeContext {
//     Param* kv_cache;
//     uint32_t seq_len;
// };

// Graph 不负责 "边"
// 拓扑关系在 Layer 之间 而 Graph 只负责 "容器 + 顺序视图"
class Graph {
public:
    Graph() = delete;
    Graph(Layer &layer);
    ~Graph();

    Layer *operator[](UINT id);

    // 拓扑构建
    // void add_layer(Layer* layer);
    // 子图切分（语义切分 / pipeline）
    // Graph* split(uint32_t begin, uint32_t end);

    // 拓扑分析 用于构建执行顺序
    void buildExecutionOrder(Layer *inputL);
    UINT WorkspaceSize();
    bool splittable(UINT num = 0);
    std::vector<LayerSlice *> &getLayerSlices(UINT sliceId, UINT sliceNum);

private:
    void dfs_collect(Layer *cur, std::unordered_set<Layer *> &visited);
    Layer* find_input_node();
public:
    UINT _layersNum;

    // Graph 中用于保存所有 Layer 的一个 "线性视图 / 管理数组"
    // 注意: 拓扑结构信息 Graph 并不存储 而是通过 Layer 内部的输入/输出引用 来表示
    // 按类型 _layers 包含 _confLayers
    // unordered_set (Hash 无序 无重复)
    std::unordered_set<Layer *> _layers;
    std::unordered_set<Layer *> _confLayers;

    // 逻辑视角上 每一个 Graph 都至少包含 _input / _output
    // 逻辑上的 一个输入节点 和 一个输出节点
    // 这两个节点需要调用者主动构建 并通过 layer.getGraph() 获得 Graph
    Layer *_inputL;   // 虚拟节点
    Layer *_outputL;  // 虚拟节点

    // Graph 视角需要记录下执行顺序
    std::vector<Layer *> _execOrder;

};

} // namespace end of core
} // namespace end of Kernel 

#endif