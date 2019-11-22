#pragma once

#include "tensor_subslice.h"
#include "tensor_container.h"


namespace tensors {
namespace _details {

template<size_t N> class dummy_argument {
private:
    template<size_t OTHER_N> friend dummy_argument<OTHER_N> make_dummy_argument();

    dummy_argument() {}
};

template<size_t N> dummy_argument<N> make_dummy_argument() {
    return dummy_argument<N>();
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
        typename OTHER_T, size_t OTHER_D, typename OTHER_A
    > friend tensor<OTHER_T, OTHER_D> make_tensor(OTHER_A&&);
    template<
        typename OTHER_T, size_t OTHER_D
    > friend tensor<OTHER_T, OTHER_D> make_tensor(_details::multidimentional_list<OTHER_T, OTHER_D>&&);

    explicit tensor(_details::dummy_argument<0>,
        const std::array<size_t, D> &sizes_, _details::tensor_container<T> &&values
    ) :
        tensor_slice<T, D>(_details::default_shape(sizes_), values.data()),
        _data(std::move(values))
    {}

    explicit tensor(_details::dummy_argument<1>,
        _details::sizes_and_values<T, D> &&x
    ) :
        tensor{_details::make_dummy_argument<0>(),
            x.sizes, _details::tensor_container<T>(std::move(x.values))
        }
    {}

    template<typename A> explicit tensor(_details::dummy_argument<2>,
        A &&a
    ) :
        tensor{_details::make_dummy_argument<1>(),
            _details::sizes_and_values<T, D>(std::forward<A>(a))
        }
    {}

    explicit tensor(_details::dummy_argument<3>,
        _details::multidimentional_list<T, D> &&a
    ) :
        tensor{_details::make_dummy_argument<1>(),
            _details::sizes_and_values<T, D>(std::move(a))
        }
    {}

public:
    explicit tensor(const std::array<size_t, D> &sizes_, const T &value = T()) :
        tensor{_details::make_dummy_argument<0>(),
            sizes_, _details::tensor_container<T>(_details::product(sizes_), value)
        }
    {}

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


template<typename T, size_t D, typename A> tensor<T, D> make_tensor(A &&a) {
    return tensor(_details::make_dummy_argument<2>(), std::forward<A>(a));
}

template<typename T, size_t D> tensor<T, D> make_tensor(_details::multidimentional_list<T, D> &&a) {
    return tensor(_details::make_dummy_argument<3>(), std::move(a));
}

template<typename T> tensor<T, 0> make_scalar(const T &a) {
    return tensor<T, 0>({}, a);
}

}  // namespace tensors
