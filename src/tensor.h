#pragma once

#include "tensor_subslice.h"
#include "tensor_container.h"


namespace tensors {

template<typename T, size_t D> class tensor;

namespace _details {

template<typename T, size_t D> tensor<T, D> construct_tensor(sizes_and_values<T, D> &&x) {
    return tensor<T, D>(x.sizes, tensor_container<T>(std::move(x.values)));
}

}  // namespace _details


template<typename T, size_t D> class tensor_slice : public tensor_subslice<T, D> {
private:
    const std::array<_details::size_and_step, D> _shape;

protected:
    template<
        typename OTHER_OP, size_t OTHER_D
    > friend class tensor_subslice;

    tensor_slice(const std::array<_details::size_and_step, D> &shape, T *ptr_ = nullptr) :
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


template<typename T, size_t D> class tensor : public tensor_slice<T, D> {
private:
    _details::tensor_container<T> _data;

    template<
        typename OTHER_T, size_t OTHER_D
    > friend tensor<OTHER_T, OTHER_D> _details::construct_tensor(
        _details::sizes_and_values<OTHER_T, OTHER_D> &&x
    );

    explicit tensor(
        const std::array<size_t, D> &sizes_, _details::tensor_container<T> &&values
    ) :
        tensor_slice<T, D>(_details::default_shape(sizes_), values.data()),
        _data(std::move(values))
    {}

public:
    explicit tensor(const std::array<size_t, D> &sizes_, const T &value = T()) :
        tensor{sizes_, _details::tensor_container<T>(_details::product(sizes_), value)}
    {}

    template<typename OTHER_T> tensor(tensor_subslice<OTHER_T, D> other) :
        tensor{other.sizes()}
    {
        *this = other;
    }

    tensor(const tensor &other) :
        tensor{static_cast<const tensor_subslice<T, D>&>(other)}
    {}

    tensor(tensor &&other) :
        tensor{other.sizes(), std::move(other._data)}
    {}

    tensor operator=(const tensor &other) {
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


template<typename T, size_t D, typename A> tensor<T, D> make_tensor(A &&a) {
    return _details::construct_tensor(_details::sizes_and_values<T, D>(std::forward<A>(a)));
}

template<typename T, size_t D> tensor<T, D> make_tensor(_details::multidimentional_list<T, D> &&a) {
    return _details::construct_tensor(_details::sizes_and_values<T, D>(std::move(a)));
}

template<typename A> auto make_scalar(A &&a) {
    return make_tensor<std::remove_const_t<std::remove_reference_t<A>>, 0>(std::forward<A>(a));
}

}  // namespace tensors
