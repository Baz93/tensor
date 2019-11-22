#pragma once

#include "tensor_subslice.h"


template<typename T, size_t D> class tensor_iterator :
    private tensor_subslice<T, D>,
    public std::iterator<
        std::random_access_iterator_tag,
        tensor_subslice<T, D>
    >
{
protected:
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
        tensor_subslice<T, D>::_ptr += step();
        _index++;
        return *this;
    }

    tensor_iterator& operator--() {
        tensor_subslice<T, D>::_ptr -= step();
        _index--;
        return *this;
    }

    tensor_iterator& operator+=(ptrdiff_t n) {
        tensor_subslice<T, D>::_ptr += n * step();
        _index += n;
        return *this;
    }

    tensor_iterator& operator-=(ptrdiff_t n) {
        tensor_subslice<T, D>::_ptr -= n * step(size_t(-1));
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


template<typename T, size_t D> class tensor_slice : public tensor_subslice<T, D> {
private:
    const std::array<size_and_step, D> _shape;

protected:
    template<
        typename OTHER_OP, size_t OTHER_D
    > friend class tensor_subslice;

    tensor_slice(const std::array<size_and_step, D> &shape, T *ptr_ = nullptr) :
        tensor_subslice<T, D>(_shape.data(), ptr_),
        _shape(shape)
    {}

    std::array<size_t, D> shape() const {
        return _shape;
    }

public:
    tensor_slice& operator=(const tensor_slice &other) {
        return operator=<T, D>(other);
    }

    template<typename OTHER_T, size_t OTHER_D> tensor_slice &operator=(
        const tensor_subslice<OTHER_T, OTHER_D> &other
    ) {
        element_wise_apply([](T &lhs, const OTHER_T &rhs) {
            lhs = rhs;
        }, *this, other);
        return *this;
    }
};


template<size_t D> std::array<size_and_step, D> default_shape(const std::array<size_t, D> &sizes) {
    std::array<size_and_step, D> result;
    ptrdiff_t product = 1;
    for (size_t i = D; i-- > 0; ) {
        result[i] = {sizes[i], product};
        product *= sizes[i];
    }
    return result;
}

template<size_t D> size_t product(const std::array<size_t, D> &sizes) {
    size_t product = 1;
    for (size_t i = D; i-- > 0; ) {
        product *= sizes[i];
    }
    return product;
}


template<size_t D> struct scan_sizes_impl {
    template<typename A> static void process(size_t *pos, const A &a) {
        *pos = a.size();
        for (const auto &b : a) {
            scan_sizes_impl<D - 1>::process(pos + 1, b);
            break;
        }
    }
};

template<> struct scan_sizes_impl<0> {
    template<typename A> static void process(size_t *pos, const A &a) {
        UNUSED(pos);
        UNUSED(a);
    }
};

template<size_t D, typename A> std::array<size_t, D> scan_sizes(const A &a) {
    std::array<size_t, D> result;
    scan_sizes_impl<D>::process(result.data(), a);
    return result;
}


template<typename T, size_t D> struct planarize_impl {
    template<typename A> static void process(std::vector<T> &result, const A &a) {
        for (const auto &b : a) {
            planarize_impl<T, D - 1>::process(result, b);
        }
    }
};

template<typename T> struct planarize_impl<T, 0> {
    template<typename A> static void process(std::vector<T> &result, const A &a) {
        result.push_back(a);
    }
};

template<typename T, size_t D, typename A> std::vector<T> planarize(const A &a) {
    std::vector<T> result;
    planarize_impl<T, D>::process(result, a);
    return result;
}


template<typename T> class tensor_container : private std::vector<T> {
public:
    tensor_container(size_t n, const T &val) :
        std::vector<T>(n, val)
    {}

    tensor_container(std::initializer_list<T> a) :
        std::vector<T>(a)
    {}

    explicit tensor_container(std::vector<T> &&a) :
        std::vector<T>(std::move(a))
    {}

    T* data() {
        return std::vector<T>::data();
    }
};


template<> class tensor_container<bool> : private std::vector<char> {
public:
    tensor_container(size_t n, const bool &val) :
        std::vector<char>(n, val)
    {}

    tensor_container(std::initializer_list<bool> a) :
        std::vector<char>(a.begin(), a.end())
    {}

    explicit tensor_container(std::vector<bool> &&a) :
        std::vector<char>(a.begin(), a.end())
    {}

    bool* data() {
        return reinterpret_cast<bool *>(std::vector<char>::data());
    }
};


template<typename T, size_t D> class tensor : public tensor_slice<T, D> {
protected:
    using tensor_base<T, D>::_ptr;

private:
    tensor_container<T> _data;

public:
    explicit tensor(const std::array<size_t, D> &sizes_, const T &value = T()) :
        tensor_slice<T, D>(default_shape(sizes_)),
        _data(product(sizes_), value)
    {
        tensor_slice<T, D>::_ptr = _data.data();
    }

    tensor(const std::array<size_t, D> &sizes_, const std::initializer_list<T> &l) :
        tensor_slice<T, D>(default_shape(sizes_)),
        _data(l)
    {
        tensor_slice<T, D>::_ptr = _data.data();
    }

    template<typename A> tensor(const std::vector<A> &l) :
        tensor_slice<T, D>(default_shape(scan_sizes<D>(l))),
        _data(planarize<T, D>(l))
    {
        tensor_slice<T, D>::_ptr = _data.data();
    }

    tensor& operator=(const tensor &other) {
        return operator=<T, D>(other);
    }

    template<typename OTHER_T, size_t OTHER_D> tensor& operator=(
        const tensor_subslice<OTHER_T, OTHER_D> &other
    ) {
        element_wise_apply([](T &lhs, const OTHER_T &rhs) {
            lhs = rhs;
        }, *this, other);
        return *this;
    }
};


template<typename T> class scalar : public tensor_slice<T, 0> {
private:
    T _data;

public:
    explicit scalar(const T &value = T()) :
        tensor_slice<T, 0>({}, &_data),
        _data(value)
    {}
};
