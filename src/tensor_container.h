#pragma once

#include "tensor_common.h"


namespace tensors {
namespace _details {

template<typename T> using value_store_type = typename std::conditional<std::is_same<T, bool>::value, char, T>::type;


template<
    typename T, size_t D
> class multidimentional_list : public std::initializer_list<
    multidimentional_list<T, D - 1>
> {
public:
    multidimentional_list(std::initializer_list<multidimentional_list<T, D - 1>> &&a) :
        std::initializer_list<multidimentional_list<T, D - 1>>(a)
    {}
};

template<
    typename T
> class multidimentional_list<T, 1> : public std::initializer_list<T> {
public:
    multidimentional_list(std::initializer_list<T> &&a) :
        std::initializer_list<T>(a)
    {}
};


template<typename T, size_t D> struct sizes_and_values {
    std::array<size_t, D> sizes;
    std::vector<value_store_type<T>> values;

    template<typename A> explicit sizes_and_values(A &&a) {
        sizes.fill(_details::npos);
        if (std::is_lvalue_reference<A>::value) {
            fill<0, const std::remove_reference_t<A>>(a);
        } else {
            fill<0>(a);
        }
    }

    template<size_t I, typename A> void fill(A &a, REQUEST_ARG(I == D)) {
        values.emplace_back(std::move(a));
    }

    template<size_t I, typename A> void fill(A &a, REQUEST_ARG(I < D)) {
        size_t size = 0;
        for (auto &b : a) {
            fill<I + 1>(b);
            ++size;
        }
        if (sizes[I] == _details::npos) {
            sizes[I] = size;
        } else {
            assert(sizes[I] == size);
        }
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


template<typename T> class tensor_container : private std::vector<value_store_type<T>> {
public:
    tensor_container(size_t n, const T &val) :
        std::vector<value_store_type<T>>(n, val)
    {}

    explicit tensor_container(std::vector<value_store_type<T>> &&a) :
        std::vector<value_store_type<T>>(std::move(a))
    {}

    T* data() {
        return reinterpret_cast<T*>(std::vector<value_store_type<T>>::data());
    }
};

}  // namespace _details
}  // namespace tensors
