#pragma once

#include "tensor_common.h"


namespace tensors {

template<typename T, size_t D> class tensor_slice;
template<typename T, size_t D> class tensor_subslice;

namespace _details {

template<typename T, size_t D> class tensor_iterator;

struct size_and_step {
    size_t size;
    ptrdiff_t step;
};


template<typename T, size_t D> class tensor_base {
private:
    const size_and_step *const _shape;
    T *_ptr;

public:
    using iterator = tensor_iterator<T, D - 1>;
    using const_iterator = tensor_iterator<const T, D - 1>;

    tensor_base(const size_and_step *shape, T *ptr) :
        _shape(shape), _ptr(ptr)
    {}

protected:
    const size_and_step* shape() const {
        return _shape;
    }

    T* ptr() const {
        return _ptr;
    }

private:
    T*& mutable_ptr() {
        return _ptr;
    }

    template<
        typename OTHER_T, size_t OTHER_D
    > friend class tensor_iterator;

public:
    tensor_subslice<T, D> forward() {
        return tensor_subslice<T, D>(_shape, _ptr);
    }

    tensor_subslice<const T, D> forward() const {
        return tensor_subslice<const T, D>(_shape, _ptr);
    }

    std::array<size_t, D> sizes() const {
        std::array<size_t, D> res;
        for (size_t i = 0 ; i < D; ++i) {
            res[i] = _shape[i].size;
        }
        return res;
    }

    size_t size(size_t i) const {
        return _shape[i].size;
    }

protected:
    size_t step(size_t i) const {
        return _shape[i].step;
    }
};


template<typename T, size_t D> class tensor_iterable : public tensor_base<T, D> {
protected:
    using tensor_base<T, D>::ptr;
    using tensor_base<T, D>::shape;
public:
    using tensor_base<T, D>::size;

public:
    using iterator = tensor_iterator<T, D - 1>;
    using const_iterator = tensor_iterator<const T, D - 1>;

    tensor_iterable(const size_and_step *shape_, T *ptr_) :
        tensor_base<T, D>(shape_, ptr_)
    {}

    iterator begin() {
        return iterator(shape() + 1, ptr());
    }

    iterator end() {
        return begin() += size(0);
    }

    const_iterator begin() const {
        return const_iterator(shape() + 1, ptr());
    }

    const_iterator end() const {
        return begin() += size(0);
    }

    typename iterator::value_type operator[](size_t i) {
        return *(begin() += i);
    }

    typename const_iterator::value_type operator[](size_t i) const {
        return *(begin() += i);
    }
};


template<typename T> class tensor_iterable<T, 0> : public tensor_base<T, 0> {
protected:
    using tensor_base<T, 0>::ptr;

public:
    tensor_iterable(const size_and_step *shape_, T *ptr_) :
        tensor_base<T, 0>(shape_, ptr_)
    {}

    const T& get() const {
        return *ptr();
    }

    T& get() {
        return *ptr();
    }
};

}  // namespace _details
}  // namespace tensors
