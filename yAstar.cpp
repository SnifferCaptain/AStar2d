#include "yAstar.hpp"
#include<algorithm>
#include<cmath>
#include<execution>
#include<thread>
#include<future>
#include<ranges>

inline float quickSqrt(float x){
    // // c++版本
    // int i = *reinterpret_cast<int*>(&x);
    // i = 0x1fbd1df5 + (i >> 1);
    // float y = *reinterpret_cast<float *>(&i);
    // y = 0.5f * y + 0.5f * x / y; // n1  402.729/401.593
    // // y = 0.5f * y + 0.5f * x / y; // n2 not needed   401.593/401.593

    // // c语言版本
    // int i = *(int*)&x;
    // i = 0x1fbd1df5 + (i >> 1);
    // float y = *(float*)&i;
    // y = 0.5f * y + 0.5f * x / y; // n1

    // return y;

    // sqrtf拥有底层硬件支持，所以上面的实现还快。
    return sqrtf(x);
}

inline float distanceFromLine(float l0x,float l0y,float l1x,float l1y,float x,float y){
    float dx = l1x-l0x, dy = l1y-l0y;              // 线段的向量
    float dx2 = x - l0x, dy2 = y - l0y;            // 线段起点到点的向量
    float dot = dx * dx2 + dy * dy2;               // =|a||b|cosθ
    float d1 = dot / quickSqrt(dx * dx + dy * dy); // a-
    float d12 = d1 * d1;                           // a-^2
    return quickSqrt(dx2 * dx2 + dy2 * dy2 - d12);
}

std::vector<std::pair<float, float>> simplifyPathDP(std::vector<std::pair<float, float>>& path,int index0, int index1, float& threshold){
    // DP 简化路径
    if(index1-index0<=2){
        std::vector<std::pair<float, float>> oppath;
        oppath.emplace_back(path[index1]);
        return oppath;
    }
    float maxDistance = 0;
    int maxIndex = index0;
    std::vector<float> distances(index1-index0-1);// 计算每个点到直线的距离
    std::transform(std::execution::par_unseq, path.begin()+index0+1, path.begin()+index1, distances.begin(), [&](auto& p){
        return distanceFromLine(path[index0].first, path[index0].second, path[index1].first, path[index1].second, p.first, p.second);
    });
    maxIndex = std::distance(distances.begin(), std::max_element(distances.begin(), distances.end()))+index0+1;
    maxDistance = distances[maxIndex-index0-1];
    if(maxDistance>threshold){
        std::future<std::vector<std::pair<float, float>>> f0 = std::async(std::launch::async, simplifyPathDP, std::ref(path), index0, maxIndex, std::ref(threshold));
        std::future<std::vector<std::pair<float, float>>> f1 = std::async(std::launch::async, simplifyPathDP, std::ref(path), maxIndex, index1, std::ref(threshold));
        std::vector<std::pair<float, float>> path0 = f0.get();
        std::vector<std::pair<float, float>> path1 = f1.get();
        path0.insert(path0.end(), path1.begin(), path1.end());
        return path0;
    }else{
        std::vector<std::pair<float, float>> oppath;
        oppath.emplace_back(path[index1]);
        return oppath;
    }
}

std::vector<std::pair<float, float>> AStar::simplifyPath(std::vector<std::pair<float, float>>& path, float threshold){
    // DP 简化路径
    int index0=0, index1=path.size()-1;
    auto noStart = simplifyPathDP(path, index0, index1, threshold);
    noStart.insert(noStart.begin(), path.front());
    return noStart;
}



float AStar::getLength(std::vector<std::pair<float, float>>& _path) const {
    return std::transform_reduce(std::execution::par_unseq, _path.begin(), _path.end()-1,_path.begin()+1,0.f,std::plus<>(),[](auto& a, auto& b){
        return quickSqrt((a.first-b.first)*(a.first-b.first)+(a.second-b.second)*(a.second-b.second));
    });
}

////////////////////////////// AStar //////////////////////////////

AStar::AStar(int width, int height, u_char* mapData){
    setMap(width, height, mapData);
    setMapping(1.0f);
    setStride(1.0f);
    setNeighourCount(8);
    reseted = false;
    costWeight = 1.0f;
    speed=0.f;
    traditional = false;
}

void AStar::setMapping(float _scaleMeterPerPixel){
    mapping = _scaleMeterPerPixel;
}

void AStar::setMap(int width, int height, u_char *mapData){
    originMap = YTensor<u_char,2>(height, width);
    nodeMap= YTensor<Node,2>(height, width);
    reset();
    reseted = true;
    std::copy(mapData, mapData+width*height, originMap.data);
}

void AStar::setStride(float _strideMeter){
    stride = _strideMeter;
}

void AStar::setNeighourCount(int _neighourCount){
    neighourCount = _neighourCount;
}

void AStar::setSpeed(float _speed){
    speed = _speed;
}

void AStar::setTraditional(bool _traditional){
    traditional = _traditional;
}

void AStar::setCostMap(int width, int height, float* _costmapData,float weight){
    costMap = YTensor<float,2>(height, width);
    // std::copy(_costmapData, _costmapData+width*height, costMap.data);
    std::transform(std::execution::par_unseq, _costmapData, _costmapData+width*height, costMap.data, [weight](auto& x){ return x*weight; });
}

void AStar::setCostMap(int width, int height, u_char *_costmapData, float weight){
    costMap = YTensor<float, 2>(height, width);
    for(int a=0;a<costMap.size();a++){
        float basic = _costmapData[a];
        if(basic>=255){
            basic=std::numeric_limits<float>::infinity();
        }
        else if(basic<=0){
            basic=0.1f;
        }
        costMap.atData(a) = basic * weight;
    }
}

YTensor<float,2> AStar::getCostMap(){
    return costMap;
}

YTensor<u_char,2> AStar::getCostMapImage(){
    YTensor<u_char,2> costMapImage(costMap.shape());
    for(size_t a=0; a<costMapImage.size();a++){
        float scale = costMap.atData(a);
        scale=std::clamp(scale, 0.f, 255.f);
        costMapImage.atData(a) = static_cast<u_char>(scale);
    }
    return costMapImage;
}

void AStar::initCostMap(float _costWeight, float _funcInflateRadius, std::function<float(float)> _decayFunction){
    // 创建mask
    auto mapsize = originMap.size();
    auto mapshape = originMap.shape();
    int r = _funcInflateRadius / mapping; // 膨胀半径
    std::vector<std::pair<std::pair<int,int>,std::pair<int,float>>> biasMask;// xy bias val
    for (int a = -r; a <= r; a++){
        for (int b = -r; b <= r; b++){
            float d = std::hypotf(a, b) * mapping;
            d = _decayFunction(d);// d是权重
            int bias=a*mapshape[1]+b;
            if(d>1.f){
                d *= _costWeight;
                biasMask.emplace_back(std::make_pair(b, a), std::make_pair(bias, d));
            }
        }
    }
    costMap = YTensor<float, 2>(originMap.shape()).fill(1.f * _costWeight);
    size_t cur=0u;
    for(int a=0;a<mapshape[0];a++){
        for(int b=0;b<mapshape[1];b++){
            if(originMap.atData(cur)==0){
                //mask
                std::for_each(std::execution::seq, biasMask.begin(), biasMask.end(), [&](std::pair<std::pair<int, int>, std::pair<int,float>>& mask0){
                    int x=mask0.first.first+b, y=mask0.first.second+a;
                    if(x<0 || x>=mapshape[1] || y<0 || y>=mapshape[0])return;
                    auto& target=costMap.atData(mask0.second.first+cur);
                    target=std::max(target,mask0.second.second);
                });
                costMap.atData(cur)=std::numeric_limits<float>::infinity();
            }
            cur++;
        }
    }
}

void AStar::initCostMapFast(float _costWeight, float _funcInflateRadius, std::function<float(float)> _decayFunction){
    int r = _funcInflateRadius / mapping; // 膨胀半径
    YTensor<float, 2> temp(originMap.shape());
    // 创建一个相当于迭代器的东西，遍历所有的，空白的点。
    std::vector<std::pair<std::pair<int,int>,std::pair<int,float>>> biasMask;// xy bias val
    for (int a = -r; a <= r; a++){
        for (int b = -r; b <= r; b++){
            float d = std::hypotf(a, b) * mapping;
            d = _decayFunction(d);// d是权重
            int bias=a*originMap.shape(1)+b;
            if(d>1.f){
                d *= _costWeight;
                biasMask.emplace_back(std::make_pair(b, a), std::make_pair(bias, d));
            }
        }
    }
    std::sort(biasMask.begin(), biasMask.end(), [](auto& a, auto& b){
        return a.second.second>b.second.second;
    });
    // 现在，biasMask是按照权重从大到小排序的。
    auto indexView = std::views::iota(0, static_cast<int>(originMap.size()));
    std::transform(std::execution::par_unseq, indexView.begin(), indexView.end(), temp.data, [&](int i){
        if(originMap.atData(i) == 0)return std::numeric_limits<float>::infinity();
        int x = i % originMap.shape(1);
        int y = i / originMap.shape(1);
        float mindist = std::numeric_limits<float>::infinity();
        for(auto& mask0:biasMask){
            int nx=mask0.first.first+x, ny=mask0.first.second+y;
            if(nx<0 || nx>=originMap.shape(1) || ny<0 || ny>=originMap.shape(0))continue;
            auto& target= originMap.atData(mask0.second.first+i);
            if(target==0){
                // 碰到了障碍物，直接结算！
                return mask0.second.second;
            }
        }
        return _costWeight;
    });
    costMap = temp;
}

std::vector<std::pair<float,float>> AStar::search(std::pair<float, float> start, std::pair<float, float> end){
    if(!reseted){
        reset();
    }
    // 创建最大堆(index存储)
    std::priority_queue<size_t, std::vector<size_t>, CompareNode> openList(CompareNode(nodeMap, std::placeholders::_1));
    int startx = start.first/mapping, starty = start.second/mapping;// 起点
    int endx = end.first/mapping, endy = end.second/mapping;// 终点
    int traditionalNeighourCount = std::clamp(neighourCount, 4, 8);
    float mappedStride = stride/mapping;// 映射后的步长
    float mappedSpeed = speed/mapping;// 映射后的最大速度
    float mappedTogether=mappedStride+mappedSpeed;
    std::vector<float> angles(neighourCount);
    for(int i=0;i<neighourCount;i++){
        angles[i] = i*2*M_PI/neighourCount;
    }
    Node startNode(0, std::hypotf(startx-endx,starty-endy), -1);
    nodeMap[starty][startx] = startNode;
    openList.push(starty*nodeMap.shape(1)+startx);
    while(!openList.empty()){
        size_t index = openList.top();
        openList.pop();
        int y = index/nodeMap.shape(1), x = index%nodeMap.shape(1);
        if(x==endx && y==endy){
            // 找到终点
            std::vector<std::pair<float, float>> path;
            while(index!=-1){
                path.push_back(std::make_pair(x*mapping, y*mapping));
                index = nodeMap.atData(index).parent;
                if(index==-1){
                    break;
                }
                x = index%nodeMap.shape(1);
                y = index/nodeMap.shape(1);
            }
            std::reverse(path.begin(), path.end());
            return path;
        }
        nodeMap.atData(index).closed = true;// 标记为已关闭
        if(traditional || nodeMap.atData(index).estim<mappedTogether){
            // 传统A*算法 or 临近终点
            static constexpr int neighour[8][2] = {{-1,0},{1,0},{0,-1},{0,1},{-1,-1},{-1,1},{1,-1},{1,1}};
            for (int i = 0; i < traditionalNeighourCount; i++){
                int nx=x+neighour[i][1], ny=y+neighour[i][0];
                size_t nindex=ny*nodeMap.shape(1)+nx;
                if(nx>=0 && nx<nodeMap.shape(1) && ny>=0 && ny<nodeMap.shape(0)){
                    if(costMap.atData(nindex)<std::numeric_limits<float>::infinity()){
                        auto& nd=nodeMap.atData(nindex);
                        if(nd.closed)continue;
                        float newCost = nodeMap.atData(index).cost + costMap.atData(nindex) * (1.f + static_cast<int>(i/4)*0.414f);// 分支优化最终版本！
                        if(newCost<nd.cost){
                            nd.cost = newCost;
                            nd.estim = quickSqrt((nx - endx) * (nx - endx) + (ny - endy) * (ny - endy));
                            // nd.estim = std::hypotf(nx - endx, ny - endy);// 变慢了！！！
                            nd.parent = index;
                            openList.push(nindex);
                        }
                    }
                }
            }
        }// 初始blast算法没有任何改进
        else{
            // 考虑车车动量的A*算法，更慢但是更快（指实际行走）
            float forx=nodeMap.atData(index).speedx, fory=nodeMap.atData(index).speedy;
            float angle0 = std::atan2(fory, forx);// 速度方向角度
            // constexpr float angle0 = 0.f;// 轨迹质量严重下降，如果关闭这个
            forx += x;
            fory += y; // 现在表示速度方向后x的位置
            for(int i=0;i<neighourCount;i++){
                float angle = angles[i] + angle0; // 邻居角度（已考虑方向）
                int nx = forx + mappedStride * quickSqrt(1 / (1 + std::tan(angle) * std::tan(angle))) * std::cos(angle);
                int ny = fory + mappedStride * quickSqrt(1 / (1 + std::tan(angle) * std::tan(angle))) * std::sin(angle); // 求解的最终邻居位置，且距离代价理应相等
                size_t nindex = ny*nodeMap.shape(1)+nx;
                if(nx>=0 && nx<nodeMap.shape(1) && ny>=0 && ny<nodeMap.shape(0)){
                    if(costMap.atData(nindex)<std::numeric_limits<float>::infinity()){
                        // 能走
                        auto& nd = nodeMap.atData(nindex);
                        if(nd.closed)continue;
                        float newCost = nodeMap.atData(index).cost + stride * costMap.atData(nindex); // costWeight 在初始化处已经乘过了
                        if(newCost< nd.cost){
                            // 不管是不是open都可以更新。
                            nd.cost = newCost;
                            nd.estim = quickSqrt((nx - endx) * (nx - endx) + (ny - endy) * (ny - endy)) * mapping;
                            nd.parent = index;
                            nd.speedx = std::clamp(static_cast<float>(nx - x), -mappedSpeed, mappedSpeed);
                            nd.speedy = std::clamp(static_cast<float>(ny - y), -mappedSpeed, mappedSpeed);
                            openList.push(nindex);
                        }
                    }
                }
            }
        }

    }
    // 未找到路径
    std::cout<<"No path found!"<<std::endl;
    return std::vector<std::pair<float, float>>();
}

void AStar::reset(){
    Node zeroNode;
    std::fill(nodeMap.data, nodeMap.data+nodeMap.size(), zeroNode);
    reseted = true;
}
