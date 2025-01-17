#ifndef YASTAR_HPP
#define YASTAR_HPP

#include <vector>
#include <queue>
#include <unordered_map>
#include <functional>
#include "ytensor.hpp"



// @brief 旨在使用空间换速度的A*算法实现 
class AStar {
public:
    AStar() = default;
    AStar(int width, int height, u_char* mapData);
    
    // 设置映射（每个像素代表x米）
    void setMapping(float _scaleMeterPerPixel);

    // 设置地图
    void setMap(int width, int height, u_char *mapData);

    // 设置代价地图，weight 为代价权重，默认为1
    void setCostMap(int width, int height, float* _costmapData, float weight=1.0f);

    // 设置代价地图
    void setCostMap(int width, int height, u_char *_costmapData, float weight=1.0f);

    // 获取代价地图数据，不可以保存为图片
    YTensor<float,2> getCostMap();

    // 获取代价地图，可以保存为图片
    YTensor<u_char,2> getCostMapImage();

    // 设置行走步长,需要小于Speed的值
    void setStride(float _strideMeter);

    // 设置邻居节点个数（推荐为大于2的质数，实测3已经能够达到比较可以的效果）
    void setNeighourCount(int _neighourCount);

    // 设置速度(m/s),需要大于Stride的值
    void setSpeed(float _speed);

    // 设置是否采用传统A*算法，默认为false。传统A*算法至少需要4个邻居节点。
    void setTraditional(bool _traditional);

    // @brief 初始化代价地图
    // @param _costWeight 代价权重
    // @param _funcInflateRadius 障碍物函数影响半径（米）
    // @param _decayFunction 代价衰减函数，传入最近障碍物的距离（米），返回代价。函数不需要考虑代价权重
    void initCostMap(float _costWeight = 1.0f, float _funcInflateRadius = 1.f, std::function<float(float)> _decayFunction = [](float x){ return 1 / x; });

    // @brief SIMD初始化代价地图(avx2)
    // @param _costWeight 代价权重
    // @param _funcInflateRange 障碍物函数影响边长（方形，像素，需要8的倍数）
    // @param _decayFunction 代价衰减函数，传入最近障碍物的距离（米），返回代价。函数不需要考虑代价权重
    void initCostMapFast(float _costWeight = 1.0f, float _funcInflateRadius = 1.f, std::function<float(float)> _decayFunction = [](float x){ return 1 / x; });

    // @brief 搜索路径
    // @param start 起点
    // @param end 终点
    // @return 路径 （返回空数组表示无解）
    std::vector<std::pair<float, float>> search(std::pair<float, float> start, std::pair<float, float> end);

    // 重置地图，每次搜索前都需要调用一次，不过其实search函数里面有检查的，会自动重置。
    void reset();

    // @brief 压缩路径，将路径中的冗余点删除。
    // @param path 路径
    // @param threshold 阈值，小于该值的点会被删除，单位可以为米，与传入路径的scale相同。
    std::vector<std::pair<float, float>> simplifyPath(std::vector<std::pair<float, float>>& _path, float threshold=0.1f);

    // @brief 压缩路径，按照距离间隔压缩。
    std::vector<std::pair<float, float>> chunkPath(std::vector<std::pair<float, float>>& _path, float threshold=0.1f);

    // @brief 获取路径长度
    // @param _path 路径
    // @return 路径长度
    float getLength(std::vector<std::pair<float, float>>& _path)const;
protected:
    // 节点结构体
    struct Node{
        float cost, estim; // 到节点前的代价，到终点的启发代价
        int parent;        // 父节点索引
        float speedx, speedy; // 速度（*就是每一步的速度偏移量，按照像素计算*）  不需要添加angle，因为都是两次求解，没必要浪费空间。
        bool closed;// 是不是闭集
        inline float getCostTotal() { return cost + estim; }// 我有一个主意，把speedx和speedy也加入到代价中，因为实际情况是我希望车车越快越好。
        explicit Node(float _cost = std::numeric_limits<float>::infinity(), float _estim = std::numeric_limits<float>::infinity(), int _parent = -1, float _speedx = 0, float _speedy = 0):
            cost(_cost), estim(_estim), parent(_parent), speedx(_speedx), speedy(_speedy),closed(false) {}
        Node& operator=(const Node& other){
            cost = other.cost;
            estim = other.estim;
            parent = other.parent;
            speedx = other.speedx;
            speedy = other.speedy;
            closed = other.closed;
            return *this;
        }
    };

    // 比较节点的优先级（cost total）
    struct CompareNode{
        const YTensor<Node,2> &cpMap;
        // const float estimWeight;// 预计代价权重

        CompareNode(const YTensor<Node,2>& _cpMap, std::_Placeholder<1> _= std::placeholders::_1):
        cpMap(_cpMap)
        {}

        bool operator()(size_t& index, size_t& index2){
            return cpMap.data[index].getCostTotal() > cpMap.data[index2].getCostTotal();
        }
    };

    YTensor<u_char,2> originMap;
    YTensor<float,2> costMap;
    YTensor<Node,2> nodeMap;
    float mapping;// 映射比例，每个像素代表多少米
    float stride;// 行走步长(米)
    float speed;// 行走速度(米/秒)
    int neighourCount; // 邻居节点个数
    // float estimWeight;// 预计代价权重
    float costWeight;// 代价权重
    bool reseted;// 是否重置过
    bool traditional;// 是否使用传统A*算法
    
};

#endif // YASTAR_HPP