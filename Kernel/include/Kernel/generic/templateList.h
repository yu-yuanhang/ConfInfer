#ifndef __TEMPLATELIST_H__
#define __TEMPLATELIST_H__

#include <iostream>
#include <memory>

using std::cout;
using std::cin;
using std::endl;
using std::unique_ptr;


namespace Kernel {

struct FILECloser    {
    void operator()(FILE *fp) {
        if (fp) {
            cout << "fclose()" << endl;
            fclose(fp);
        }
    }
};

template <typename T, bool Owns = false>
class List;

// 这里是一个链表的类模板 
// 默认情况下 : 这里节点的析构函数不会影响 _data 所指对象的生命周期
template <typename T, bool Owns = false>
class Node {
friend class List<T, Owns>;
private:
    Node(T* data = nullptr): 
        _next(nullptr), _pre(nullptr), _data(data), _owns(Owns) {}
    ~Node() {
        if (_owns) { delete _data; }
        _next = nullptr;
        _pre = nullptr;
    }

public:
    // 原则上 开发者调用不到这里的接口函数
    void print() const { if (_data) std::cout << *_data << std::endl; }
    T* getData() { return _data; }
    void setOwns(bool owns) { _owns = owns; }

private:
    Node    *_next;
    Node    *_pre;
    T       *_data;
    bool    _owns;
};

// 这里如果 T 是自定义的类型需要重定义输出运算符
template <typename T, bool Owns>
class List {
// ==================================
public:
    class iterator {
        public:
            iterator(Node<T, Owns> *ptr):_ptr(ptr) {}
            iterator(const iterator &rhs):_ptr(rhs._ptr) {}
            ~iterator() {}

            T *operator*() const { return (_ptr->_data); }
            T *operator->() const { return (_ptr->_data); }

            iterator &operator++() {
                _ptr = _ptr->_next;
                return *this;
            }
            iterator &operator--() {
                _ptr = _ptr->_pre;
                return *this;
            }

            iterator& operator=(const iterator& rhs) {
                if (this != &rhs) { _ptr = rhs._ptr; }
                return *this;  
            }
            iterator operator++(int) {
                iterator tmp = *this;
                _ptr = _ptr->_next;     
                return tmp;             
            }
            iterator operator--(int) {
                iterator tmp = *this;
                _ptr = _ptr->_pre;
                return tmp;
            }

            bool operator==(const iterator &rhs) const { return _ptr == rhs._ptr; } 
            bool operator!=(const iterator &rhs) const { return _ptr != rhs._ptr; }

        private:
            Node<T, Owns> *_ptr;
    };
    iterator begin() { return iterator(_head->_next); }
    iterator end() { return iterator(_tail); }
// ==================================
public:
    List(): _head(new Node<T, Owns>()), _tail(new Node<T, Owns>()), 
            _ptr(nullptr), _size(0) {
        _head->_next = _tail;
        _tail->_pre = _head;
    }   
    ~List() {
        _ptr = _head->_next;
        while (_ptr != _tail) {
            _head->_next = _ptr->_next;
            _ptr->_next->_pre = _head;
            delete _ptr;
            _ptr = nullptr;
            _ptr = _head->_next;
            _size--;
        }
        delete _head;
        _head = nullptr;
        _ptr = nullptr;
        delete _tail;
        _tail = nullptr;
    }
    size_t size() const { return _size; }
    bool empty() const { return 0 == _size ? true : false; }

    bool contains(T* data) const {
        Node<T, Owns> *cur = _head->_next;
        while (cur != _tail) {
            if (cur->getData() == data) {
                return true;
            }
            cur = cur->_next;
        }
        return false;
    }

    void push_front(T *data) {
        Node<T, Owns> *ptr = new Node<T, Owns>(data);
        ptr->_next = _head->_next;
        ptr->_pre = _head;
        ptr->_next->_pre = ptr;
        _head->_next = ptr;
        _size++;
    }
    void push_back(T *data) {
        Node<T, Owns> *ptr = new Node<T, Owns>(data);
        ptr->_next = _tail;
        ptr->_pre = _tail->_pre;
        ptr->_pre->_next = ptr;
        _tail->_pre = ptr;
        _size++;
    }
    void erase_front() {
        Node<T, Owns> *ptr = _head->_next;
        if (ptr == _tail) return;
        _head->_next = ptr->_next;
        ptr->_next->_pre = _head;
        delete ptr;
        ptr = nullptr;
        _size--;
    }
    void erase_back() {
        Node<T, Owns> *ptr = _tail->_pre;
        if (ptr == _head) return;
        _tail->_pre = ptr->_pre;
        ptr->_pre->_next = _tail;
        delete ptr;
        ptr = nullptr;
        _size--;
    }

    T *pop_front() {
        Node<T, Owns> *ptr = _head->_next;
        if (ptr == _tail) return nullptr;
        _ptr->setOwns(false);
        erase_front();
        return _ptr->getData();
    }
    T *pop_back() {
        Node<T, Owns> *ptr = _tail->_pre;
        if (ptr == _head) return nullptr;
        _ptr->setOwns(false);
        erase_back();
        return _ptr->getData();
    }

    // 调用之前 reset_ptr 重置 _ptr
    // 不过 get_ptr / reset_ptr 并不建议用 上面有迭代器可以代替
    Node<T, Owns> *get_ptr() {
        if (_ptr->_next != _tail) {
            _ptr = _ptr->_next;
        } else {
            std::cout << "_ptr->_next == _tail" << std::endl;
            return nullptr;
        }
        return _ptr;    
    }
    void reset_ptr() { _ptr = _head; }

    void display() const {}
    void print() const {
        cout << "List print : size == " << _size << endl;
        Node<T, Owns> *ptmp = _head->_next;
        for (int i = 0; i < _size; ++i) {
            cout << "!!!!!!!!!!! idx == 1 !!!!!!!!!!!" << endl;
            ptmp->print();
        }
    }
private:
    // _head 与 _tail 为空 作为边界
    Node<T, Owns> *_head;
    Node<T, Owns> *_tail;
    // _ptr 在类内部仅仅 析构函数和 reset_ptr / get_ptr 可以使用
    Node<T, Owns> *_ptr;
    size_t _size;
};

template <typename T>
bool has_duplicate_address(T* /*unused*/) {
    return false;
}

template <typename T, typename... Rest>
bool has_duplicate_addr(T *first, Rest *...rest) {
    // 折叠表达式  判断 first 是否等于后面的任何一个 
    if (((first == rest) || ...)) {
        return true;
    }
    // 后面的是否彼此重复
    return has_duplicate_address(rest...);
}

} // namespace end of Kernel 
#endif