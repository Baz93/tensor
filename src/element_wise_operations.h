#include "tensor_common.h"


namespace tensors {

template<typename T, size_t D> class tensor;
template<typename T, size_t D> class tensor_subslice;

namespace _details {

template<typename T> constexpr T constexpr_max(T x) {
    return x;
}

template<typename T0, typename T1> constexpr T0 constexpr_max(T0 x0, T1 x1) {
    return x0 > x1 ? x0 : x1;
}

template<
    typename T0, typename T1, typename T2, typename ...T
> constexpr T0 constexpr_max(T0 x0, T1 x1, T2 x2, T ...x) {
    return constexpr_max(x0, constexpr_max(x1, x2, x...));
}


template<
    size_t K = 0, typename ...T, size_t ...D, REQUEST_TPL(constexpr_max(D...) >= K)
> std::array<size_t, constexpr_max(D...) - K> common_shape(
    tensor_subslice<T, D> ...a
) {
    constexpr size_t M = constexpr_max(D...);
    std::array<size_t, M - K> shape;
    for (size_t i = 0; i < M; ++i) {
        size_t res = _details::npos;
        for (size_t val : {(i < M - D ? _details::npos : a.size(D - M + i))...}) {
            if (res == _details::npos) {
                res = val;
            } else {
                assert(val == _details::npos || res == val);
            }
        }
        assert(res != _details::npos);
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

template<typename OP, typename ...T, size_t ...D> auto element_wise_calc_impl(
    const OP &op, tensor_subslice<T, D> ...a
) {
    using R = decltype(op(std::declval<T>()...));

    return element_wise_calc_reduce_impl<R>([&op](R &res, T &...vals) {
        res = op(vals...);
    }, a...);
}

}  // namespace _details


template<typename OP, typename ...T> void element_wise_apply (
    const OP &op, T &...a
) {
    _details::element_wise_apply_impl(op, a.forward()...);
}

template<
    typename R, size_t K = 0, typename OP, typename ...T
> auto element_wise_calc_reduce (
    const OP &op, T &...a
) {
    return _details::element_wise_calc_reduce_impl<R, K>(op, a.forward()...);
}

template<typename OP, typename ...T> auto element_wise_calc (
    const OP &op, T &...a
) {
    return _details::element_wise_calc_impl(op, a.forward()...);
}

}  // namespace tensors
