#pragma once
#include <vector>
#include<cstddef>
#include <utility>
#include <iostream>

template <typename T=float, int dim=1>
class YTensor
{
public:
    using iterater = T*;
    T *data;
    int *dimensions;
    bool parent;

    ~YTensor();
    YTensor();
    YTensor(std::vector<int> dims);
    template <typename... Args>
    YTensor(Args...);
    YTensor(std::initializer_list<int> list);
    YTensor<T, dim - 1> operator[](int index);
    YTensor<T, dim> &operator=(const YTensor& other);
    YTensor<T, dim> clone()const;
    YTensor<T, dim>& fill(T value);
    YTensor<T, dim>& setAll(std::function<T(T&)> func);
    YTensor<T, dim> operator*(const YTensor &other);
    inline size_t size() const;
    std::vector<int> shape() const;
    inline int shape(int atDim) const;
    inline size_t shapeSize() const;
    inline size_t dimSize(int atDim) const;
    inline std::vector<size_t> dimSizes() const;
    template<typename... Args> inline T &at(Args... args); 
    inline T &at(std::vector<int> &pos);
    inline T &at(int pos[]);
    inline T &atData(int dataPos);
    template<typename... Args> inline size_t toIndex(Args... args);
    inline size_t toIndex(std::vector<int> &pos);
    inline size_t toIndex(int pos[]);


    template<typename _T,int _D> friend std::ostream &operator<<(std::ostream &os, const YTensor<_T, _D> &tensor);
private:
};

template <typename T>
class YTensor<T, 1>
{
public:
    using iterator = T*;
    T *data;
    int *dimensions;
    bool parent;
    ~YTensor();
    YTensor();
    YTensor(int dim0);
    T &operator[](int index);
    size_t size() const;
    template <typename _T> friend std::ostream &operator<<(std::ostream &os, const YTensor<_T,1> &tensor);
};


// realize

#define _CRT_SECURE_NO_WARNINGS
#include <algorithm>
#include <map>
#include <deque>
#include <cassert>
#include <execution>
#include <iostream>
#include <cstdarg>

template <typename T, int dim>
YTensor<T, dim>::~YTensor()
{
    if (parent)
    {
        if (data != nullptr)
        {
            delete[] data;
            data = nullptr;
        }
        if (dimensions != nullptr){
            delete[] dimensions;
            dimensions = nullptr;
        }
    }
}

template <typename T, int dim>
YTensor<T, dim>::YTensor()
{
    dimensions = new int[dim];
    std::fill(dimensions, dimensions + dim, 1);
    data = nullptr;
    parent = true;
}

template <typename T, int dim>
YTensor<T, dim>::YTensor(std::vector<int> dims)
{
    dimensions = new int[dims.size()]; // std::fill
    std::copy(dims.begin(), dims.end(), dimensions);
    parent = true;
    data = new T[size()];
}

template <typename T, int dim>
template <typename... Args>
YTensor<T, dim>::YTensor(Args... args)
{
    static_assert(sizeof...(args) == dim, "Number of arguments must match the dimension");
    dimensions = new int[dim];
    // auto seq = std::make_index_sequence<sizeof...(args)>();
    int a = 0;
    ((dimensions[a++] = args), ...);
    data = new T[size()];
    parent = true;
}

template <typename T, int dim>
YTensor<T, dim>::YTensor(std::initializer_list<int> list)
{
    dimensions = new int[list.size()];
    std::copy(list.begin(), list.end(), dimensions);
    data = new T[size()];
    parent = true;
}

template <typename T, int dim>
YTensor<T, dim - 1> YTensor<T, dim>::operator[](int index)
{
    index = (index % dimensions[0] + dimensions[0]) % dimensions[0];
    YTensor<T, dim - 1> op;
    delete[] op.dimensions;
    op.dimensions = this->dimensions + 1;
    op.data = this->data + op.size() * index;
    op.parent = false;
    return op;
}

template <typename T, int dim>
YTensor<T, dim> &YTensor<T, dim>::operator=(const YTensor<T, dim> &other)
{
    if (dim != other.shapeSize())
    {
        throw std::invalid_argument("YTensor shape size does not match");
    }
    if (parent)
    {
        if (data != nullptr)
        {
            delete[] data;
        }
    }
    std::copy(other.dimensions, other.dimensions + dim, dimensions);
    parent = true;
    this->data = new T[size()];
    std::copy(other.data, other.data + other.size(), data);
    return *this;
}

template<typename T,int dim>
YTensor<T, dim> YTensor<T, dim>::clone()const{
    YTensor<T,dim> op(this->shape());
    std::copy(std::execution::par_unseq ,data,data+size(),op.data);
    return op;
}

template<typename T,int dim>
YTensor<T,dim>& YTensor<T,dim>::fill(T value){
    std::fill(std::execution::par_unseq,data,data+size(),value);
    return *this;
}

template<typename T,int dim>
YTensor<T,dim>& YTensor<T,dim>::setAll(std::function<T(T&)> func){
    std::transform(std::execution::par_unseq,data,data+size(),data,func);
    return *this;
}

template <typename T, int dim>
YTensor<T, dim> YTensor<T, dim>::operator*(const YTensor &other)
{
    if (dim != other.shapeSize())
    {
        throw std::invalid_argument("Dimensions must match");
    }
    YTensor<T, dim> op(this->shape());
    op.parent = false;
    std::transform(std::execution::par_unseq, this->data, this->data + size(), other.data, op.data, std::multiplies<T>());
    return op;
}

template <typename T, int dim>
inline size_t YTensor<T, dim>::size() const
{
    size_t op = 1;
    int a = 0;
    for (; a < dim; a++)
    {
        op *= dimensions[a];
    }
    return op;
}

template <typename T, int dim>
std::vector<int> YTensor<T, dim>::shape() const
{
    std::vector<int> op(dim);
    for (int a = 0; a < dim; a++)
    {
        op[a] = dimensions[a];
    }
    return op;
}

template <typename T, int dim>
int YTensor<T, dim>::shape(int atDim) const
{
    atDim = (atDim % dim + dim) % dim;
    return dimensions[atDim];
}

template <typename T, int dim>
template <typename... Args>
T &YTensor<T, dim>::at(Args... args)
{
    if (dim != sizeof...(args))
    {
        throw std::invalid_argument("Number of arguments must match the dimension");
    }
    size_t index = 0;
    int a = 0;
    ((index += args * dimSize(a++)), ...);
    return data[index];
}

template <typename T, int dim>
T &YTensor<T, dim>::at(std::vector<int> &posLoc)
{
    size_t pos = 0;
    int a = 0;
    for (; a < dim; a++)
    {
        pos += posLoc[a] * dimSize(a);
    }
    return *(data + pos);
}

template <typename T, int dim>
T &YTensor<T, dim>::at(int posLoc[])
{
    size_t pos = 0;
    int a = 0;
    for (; a < dim; a++)
    {
        pos += posLoc[a] * dimSize(a);
    }
    return *(data + pos);
}

template<typename T, int dim>
T& YTensor<T,dim>::atData(int atData){
    return *(data+atData);
}

template <typename T, int dim>
size_t YTensor<T, dim>::dimSize(int atDim) const
{
    size_t op = 1;
    for (int a = atDim + 1; a < dim; a++)
    {
        op *= dimensions[a];
    }
    return op;
}

template <typename T, int dim>
std::vector<size_t> YTensor<T, dim>::dimSizes() const
{
    std::vector<size_t> op;
    for (int a = 0; a < dim; a++)
    {
        op.emplace_back(dimSize(a));
    }
    return op;
}

template <typename T, int dim>
size_t YTensor<T, dim>::shapeSize() const
{
    return dim;
}

template <typename T, int dim>
template<typename... Args>
size_t YTensor<T, dim>::toIndex(Args... args){
    if (dim != sizeof...(args))
    {
        throw std::invalid_argument("Number of arguments must match the dimension");
    }
    size_t index = 0;
    std::vector<size_t> sizes=dimSizes();
    int a = 0;
    ((index += args * sizes[a++]), ...);
    return index;
}

template <typename T, int dim>
size_t YTensor<T, dim>::toIndex(std::vector<int> &pos){
    size_t index = 0;
    std::vector<size_t> sizes=dimSizes();
    int a = 0;
    for (; a < dim; a++)    {
        index += pos[a] * sizes[a];
    }
    return index;
}

template <typename T, int dim>
size_t YTensor<T, dim>::toIndex(int pos[]){
    size_t index = 0;
    std::vector<size_t> sizes=dimSizes();
    int a = 0;
    for (; a < dim; a++){
        index += pos[a] * sizes[a];
    }
    return index;
}

template <typename T, int dim>
std::ostream &operator<<(std::ostream &out, const YTensor<T, dim> &tensor)
{
    out << "[YTensor]:<" << typeid(T).name() << ">" << std::endl;
    out << "[itemSize]: " << tensor.size() << std::endl;
    out << "[byteSize]: " << tensor.size() * sizeof(T) << std::endl;
    out << "[shape]: [";
    std::vector<int> dims = tensor.shape();
    for (int a = 0; a < static_cast<int> (dims.size() - 1); a++)
    {
        out << dims[a] << ", ";
    }
    out << dims[static_cast<int> (dims.size()) - 1] << "]" << std::endl;
    out << "[data]:" << std::endl;
    for (int a = 0; a < tensor.size(); a++)
    {
        for (int b = 0; b < static_cast<int>(dims.size()) - 3; b++)
        {
            if (a % tensor.dimSize(b) == 0)
            {
                out << "[";
            }
        }
        for (int b = static_cast<int>(dims.size()) - 3; b < static_cast<int>(dims.size()) - 1; b++)
        {
            if(b<0)continue;
            if (a % tensor.dimSize(b) == 0)
            {
                out << "[";
            }
        }
        out << tensor.data[a] << " ";
        for (int b = 0; b < static_cast<int> (dims.size() - 3); b++)
        {
            if (a % tensor.dimSize(b) == tensor.dimSize(b) - 1)
            {
                out << "]";
            }
        }
        for (int b = static_cast<int> (dims.size()) - 3; b < static_cast<int>(dims.size()) - 1; b++)
        {
            if(b<0)continue;
            if (a % tensor.dimSize(b) == tensor.dimSize(b) - 1)
            {
                out << "]" << std::endl;
            }
        }
    }
    return out;
}

//========================dim==1========================

template <typename T>
T &YTensor<T, 1>::operator[](int index)
{
    return *(data + index);
}

template <typename T>
size_t YTensor<T, 1>::size() const
{
    return dimensions[0];
}

template <typename T>
YTensor<T, 1>::~YTensor()
{
    if (parent)
    {
        if (data != nullptr)
        {
            delete[] data;
            data = nullptr;
        }
        if (dimensions != nullptr){
            delete[] dimensions;
            dimensions = nullptr;
        }
    }
}

template <typename T>
YTensor<T, 1>::YTensor()
{
    dimensions = new int[1];
    dimensions[0] = 1;
    data = nullptr;
    parent = true;
}

template <typename T>
YTensor<T, 1>::YTensor(int dim0)
{
    dimensions = new int[1];
    dimensions[0] = dim0;
    data = new T[dim0];
    parent = true;
}

template <typename T>
std::ostream &operator<<(std::ostream &out, const YTensor<T, 1> &tensor)
{
    out << "[YTensor]:<" << typeid(T).name() << ">" << std::endl;
    out << "[itemSize]: " << tensor.size() << std::endl;
    out << "[byteSize]: " << tensor.size() * sizeof(T) << std::endl;
    out << "[shape]: [";
    std::vector<int> dims = tensor.shape();
    for (int a = 0; a < dims.size() - 1; a++)
    {
        out << a << ", " << dims[a];
    }
    out << dims[dims.size() - 1] << "]" << std::endl;
    out << "[data]:" << std::endl;
    for (int a = 0; a < tensor.size(); a++)
    {
        for (int b = 0; b < dims.size() - 3; b++)
        {
            if (a % dims[b] == 0)
            {
                out << "[";
            }
        }
        for (int b = dims.size() - 3; b < dims.size() - 1; b++)
        {
            if (a % dims[b] == 0)
            {
                out << "[";
            }
        }
        out << tensor.data[a] << " ";
        for (int b = 0; b < dims.size() - 3; b++)
        {
            if (a % dims[b] == dims[b] - 1)
            {
                out << "]";
            }
        }
        for (int b = dims.size() - 3; b < dims.size() - 1; b++)
        {
            if (a % dims[b] == dims[b] - 1)
            {
                out << "]" << std::endl;
            }
        }
    }
    return out;
}

// int main(){
//     YTensor<float, 3> a(8, 8, 8);
//     for(int i=0;i<a.size();i++){
//         *(a.data+i)=i;
//     }
//     a[0][0][0]=10;
//     std::cout<<a.size()<<std::endl;
//     a=a*a;
//     for(int i=0;i<a.shape()[0];i++){
//         for(int j=0;j<a.shape()[1];j++){
//             for(int k=0;k<a.shape()[2];k++){
//                 std::cout<<a[i][j][k]<<" ";
//             }
//             std::cout<<std::endl;
//         }
//         std::cout<<std::endl;
//     }
//     std::cout<<a.at(7,3,4)<<std::endl;
//     std::cout<<a<<std::endl;

//     YTensor<float, 1> b(8);
//     int pin1=b.size();
//     for(int i=0;i<b.size();i++){
//         *(b.data+i)=i;
//     }
//     b[0]=8;
//     for(int i=0;i<b.size();i++){
//         std::cout<<b[i]<<" ";
//     }

//     return 0;
// }
