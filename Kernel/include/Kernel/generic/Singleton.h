#ifndef __SINGLETON_H__
#define __SINGLETON_H__

#include <iostream>

namespace Kernel {

using std::cout;
using std::cin;
using std::endl;

#if 0
class Singleton
{
public:
    static Point *getInstance(int ix, int iy)
    {
		if(nullptr == _pInstance) 
        {
			_pInstance = new Point(ix, iy);
			_ar;// 为了在模板参数推导时创建ar对象
		}
		return _pInstance;

    }
};
#endif

template <class T>
class Singleton {
public:
    template <class... Args>
    static T* getInstance(Args... args) {
        if(nullptr == _pInstance) {
            _pInstance = new T(args...);// 在类中创建堆对象
            _ar;
        }
        return _pInstance;
    }
#if 0
    static void destory() {
        if(nullptr != _pInstance) {
            delete _pInstance;
            _pInstance = nullptr;
        }
    }
#endif
private:
    class AutoRelease {
    public:
        AutoRelease() {
            cout << "AutoRelease()" << endl;
        }
        ~AutoRelease() {
            cout << "~AutoRelease()" << endl;
            if(_pInstance) {
                delete _pInstance;
                _pInstance = nullptr;
            }
        }
    };

private:
    // 私有成员包括成员变量和成员函数 它们只能在类的内部被访问 不能在类的外部或者派生类中被访问
    Singleton() {// 构造函数设置私有权限
        cout << "Singleton()" << endl;    
    }

    ~Singleton() {// 析构函数设置为私有 使delete函数error
        cout << "~Singleton()" << endl;
    }
    
private:
    static T * _pInstance;
    static AutoRelease _ar;
};

// 当你在模板类中定义静态数据成员时 每个模板实例都会有自己的静态数据成员 
// 这些数据成员不会在不同的模板实例之间共享
template <class T>
T * Singleton<T>::_pInstance = nullptr;
template <class T>
typename Singleton<T>::AutoRelease Singleton<T>::_ar; // typename表名是一个类型

} // namespace end of Kernel 
#endif