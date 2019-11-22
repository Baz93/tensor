#pragma once

#include "tensor_subslice.h"


namespace tensors {
namespace _details {

template<typename T, size_t D> class tensor_iterator :
    private tensor_subslice<T, D>,
    public std::iterator<
        std::random_access_iterator_tag,
        tensor_subslice<T, D>
    >
{
private:
    using tensor_base<T, D>::mutable_ptr;
    using tensor_base<T, D>::step;

    size_t _index;

    template<
        typename OTHER_T, size_t OTHER_D
    > friend class tensor_iterable;

    tensor_iterator(const size_and_step *shape_, T *ptr_, size_t index = 0) :
        tensor_subslice<T, D>(shape_, ptr_),
        _index(index)
    {}

public:
    using value_type = typename std::iterator<
        std::random_access_iterator_tag,
        tensor_subslice<T, D>
    >::value_type;

private:
    value_type* arrow() {
        return static_cast<value_type*>(this);
    }

    const value_type* arrow() const {
        return static_cast<const value_type*>(this);
    }

    size_t step() const {
        return step(size_t(-1));
    }

public:
    value_type* operator->() {
        return arrow();
    }

    const value_type* operator->() const {
        return arrow();
    }

    value_type& operator*() {
        return *arrow();
    }

    const value_type& operator*() const {
        return *arrow();
    }

    tensor_iterator& operator++() {
        tensor_subslice<T, D>::mutable_ptr() += step();
        _index++;
        return *this;
    }

    tensor_iterator& operator--() {
        tensor_subslice<T, D>::mutable_ptr() -= step();
        _index--;
        return *this;
    }

    tensor_iterator& operator+=(ptrdiff_t n) {
        tensor_subslice<T, D>::mutable_ptr() += n * step();
        _index += n;
        return *this;
    }

    tensor_iterator& operator-=(ptrdiff_t n) {
        tensor_subslice<T, D>::mutable_ptr() -= n * step(size_t(-1));
        _index -= n;
        return *this;
    }

    tensor_iterator operator++(int) {
        auto copy = *this;
        ++*this;
        return copy;
    }

    tensor_iterator operator--(int) {
        auto copy = *this;
        --*this;
        return copy;
    }

    tensor_iterator operator+(ptrdiff_t n) {
        auto copy = *this;
        return copy += n;
    }

    tensor_iterator& operator-(ptrdiff_t n) {
        auto copy = *this;
        return copy -= n;
    }

    bool operator<(const tensor_iterator &other) {
        return _index < other._index;
    }

    bool operator>(const tensor_iterator &other) {
        return _index > other._index;
    }

    bool operator==(const tensor_iterator &other) {
        return _index == other._index;
    }

    bool operator!=(const tensor_iterator &other) {
        return _index != other._index;
    }
};

}  // namespace _details
}  // namespace tensors
