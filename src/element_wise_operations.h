#include "tensor_basic_utils.h"


template<typename T, size_t D> class tensor;
template<typename T, size_t D> class tensor_subslice;


template<
    size_t K = 0, typename ...T, size_t ...D, REQUEST_TPL(constexpr_max(D...) >= K)
> std::array<size_t, constexpr_max(D...) - K> common_shape(
    tensor_subslice<T, D> ...a
) {
    constexpr size_t M = constexpr_max(D...);
    std::array<size_t, M - K> shape;
    for (size_t i = 0; i < M; ++i) {
        size_t res = npos;
        for (size_t val : {(i < M - D ? npos : a.size(D - M + i))...}) {
            if (res == npos) {
                res = val;
            } else {
                assert(val == npos || res == val);
            }
        }
        assert(res != npos);
        if (i >= K) {
            shape[i] = res;
        }
    }
    return shape;
}

template<typename OP, size_t D, typename ...T> struct element_wise_apply_equal_impl;

template<typename OP, typename ...T> struct element_wise_apply_equal_impl<OP, 0, T...> {
    static void apply(const size_t *shape, const OP &op, tensor_subslice<T, 0> ...a) {
        UNUSED(shape);
        op(a.get()...);
    }
};

template<typename OP, size_t D, typename ...T> struct element_wise_apply_equal_impl {
    static void apply(const size_t *shape, const OP &op, tensor_subslice<T, D> ...a) {
        for (size_t i = 0; i < *shape; ++i) {
            element_wise_apply_equal_impl<OP, D - 1, T...>::apply(shape + 1, op, a[i]...);
        }
    }
};

template<typename OP, size_t D, typename ...T> void element_wise_apply_equal (
    const OP &op, tensor_subslice<T, D> ...a
) {
    std::array<size_t, D> shape = common_shape(a...);
    element_wise_apply_equal_impl<OP, D, T...>::apply(shape.data(), op, a...);
}

template<typename OP, typename ...T, size_t ...D> void element_wise_apply_impl (
    const OP &op, tensor_subslice<T, D> ...a
) {
    constexpr size_t M = constexpr_max(D...);
    std::array<size_t, M> shape = common_shape(a...);
    element_wise_apply_equal_impl<OP, M, T...>::apply(shape.data(), op, (a.template expand<M - D, 0>(shape))...);
}

template<typename OP, typename ...T> void element_wise_apply (
    const OP &op, T &...a
) {
    element_wise_apply_impl(op, a.forward()...);
}

template<
    typename R, size_t K = 0, typename OP, typename ...T, size_t ...D, REQUEST_TPL(constexpr_max(D...) >= K)
> auto element_wise_calc_reduce_impl(
    const OP &op, tensor_subslice<T, D> ...a
) {
    constexpr size_t M = constexpr_max(D...);
    std::array<size_t, M> shape = common_shape(a...);
    tensor<R, M - K> result(common_shape<K>(a...));
    element_wise_apply_impl(
        op, result.template expand<0, K>(shape), (a.template expand<M - D, 0>(shape))...
    );
    return result;
}

template<
    typename R, size_t K = 0, typename OP, typename ...T
> auto element_wise_calc_reduce (
    const OP &op, T &...a
) {
    return element_wise_calc_reduce_impl<R, K>(op, a.forward()...);
}

template<typename OP, typename ...T, size_t ...D> auto element_wise_calc_impl(
    const OP &op, tensor_subslice<T, D> ...a
) {
    using R = decltype(op(std::declval<T>()...));

    return element_wise_calc_reduce_impl<R>([&op](R &res, T &...vals) {
        res = op(vals...);
    }, a...);
}

template<typename OP, typename ...T> auto element_wise_calc (
    const OP &op, T &...a
) {
    return element_wise_calc_impl(op, a.forward()...);
}