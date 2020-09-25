#pragma once

#include "element_wise_operations.h"


#define __TENSOR_CHANGE_ASSIGN_OPERATOR(symbol)                 \
template<                                                       \
    typename T1, size_t D1, typename T2, size_t D2              \
> tensor_subslice<T1, D1> operator symbol (                     \
    tensor_subslice<T1, D1> &lhs, tensor_subslice<T2, D2> rhs   \
) {                                                             \
    element_wise_apply([](T1 &v1, const T2 &v2) {               \
        v1 symbol v2;                                           \
    }, lhs, rhs);                                               \
    return lhs;                                                 \
}                                                               \
                                                                \
template<                                                       \
    typename T1, size_t D1, typename T2, size_t D2              \
> tensor_subslice<T1, D1> operator symbol (                     \
    tensor_subslice<T1, D1> &&lhs, tensor_subslice<T2, D2> rhs  \
) {                                                             \
    element_wise_apply([](T1 &v1, const T2 &v2) {               \
        v1 symbol v2;                                           \
    }, lhs, rhs);                                               \
    return lhs;                                                 \
}                                                               \


#define __TENSOR_BINARY_OPERATOR(symbol)                        \
template<                                                       \
    typename T1, size_t D1, typename T2, size_t D2              \
> auto operator symbol (                                        \
    tensor_subslice<T1, D1> lhs, tensor_subslice<T2, D2> rhs    \
) {                                                             \
    return element_wise_calc([](const T1 &v1, const T2 &v2) {   \
        return v1 symbol v2;                                    \
    }, lhs, rhs);                                               \
}                                                               \


#define __TENSOR_UNARY_OPERATOR(symbol)                         \
template<                                                       \
    typename T, size_t D                                        \
> auto operator symbol (                                        \
    tensor_subslice<T, D> val                                   \
) {                                                             \
    return element_wise_calc([](const T &v) {                   \
        return symbol v;                                        \
    }, val);                                                    \
}                                                               \


namespace tensors {

__TENSOR_CHANGE_ASSIGN_OPERATOR(+=)
__TENSOR_CHANGE_ASSIGN_OPERATOR(-=)
__TENSOR_CHANGE_ASSIGN_OPERATOR(*=)
__TENSOR_CHANGE_ASSIGN_OPERATOR(/=)
__TENSOR_CHANGE_ASSIGN_OPERATOR(%=)
__TENSOR_CHANGE_ASSIGN_OPERATOR(&=)
__TENSOR_CHANGE_ASSIGN_OPERATOR(|=)
__TENSOR_CHANGE_ASSIGN_OPERATOR(^=)
__TENSOR_CHANGE_ASSIGN_OPERATOR(<<=)
__TENSOR_CHANGE_ASSIGN_OPERATOR(>>=)

__TENSOR_BINARY_OPERATOR(+)
__TENSOR_BINARY_OPERATOR(-)
__TENSOR_BINARY_OPERATOR(*)
__TENSOR_BINARY_OPERATOR(/)
__TENSOR_BINARY_OPERATOR(%)
__TENSOR_BINARY_OPERATOR(==)
__TENSOR_BINARY_OPERATOR(!=)
__TENSOR_BINARY_OPERATOR(>)
__TENSOR_BINARY_OPERATOR(<)
__TENSOR_BINARY_OPERATOR(>=)
__TENSOR_BINARY_OPERATOR(<=)
__TENSOR_BINARY_OPERATOR(&&)
__TENSOR_BINARY_OPERATOR(||)
__TENSOR_BINARY_OPERATOR(&)
__TENSOR_BINARY_OPERATOR(|)
__TENSOR_BINARY_OPERATOR(^)
__TENSOR_BINARY_OPERATOR(<<)
__TENSOR_BINARY_OPERATOR(>>)

__TENSOR_UNARY_OPERATOR(+)
__TENSOR_UNARY_OPERATOR(-)
__TENSOR_UNARY_OPERATOR(!)
__TENSOR_UNARY_OPERATOR(~)

#undef __TENSOR_CHANGE_ASSIGN_OPERATOR
#undef __TENSOR_BINARY_OPERATOR
#undef __TENSOR_UNARY_OPERATOR

template<
    typename T1, size_t D1, typename T2, size_t D2, REQUEST_TPL(D1 > 1 && D2 > 1)
> auto matmul(
    tensor_subslice<T1, D1> lhs, tensor_subslice<T2, D2> rhs
) {
    using R = decltype(std::declval<T1>() * std::declval<T2>());

    return element_wise_calc_reduce<1>(
        [](R &res, const T1 &l, const T2 &r) {
            res += l * r;
        }, R(),
        lhs.extend(rhs.size(D2 - 1)).transpose(D1 - 1, D1),
        rhs.extend(lhs.size(D1 - 2)).transpose(D2 - 2, D2)
    );
}

template<
    typename T1, size_t D1, typename T2, REQUEST_TPL(D1 > 1)
> auto matmul(
    tensor_subslice<T1, D1> lhs, tensor_subslice<T2, 1> rhs
) {
    using R = decltype(std::declval<T1>() * std::declval<T2>());

    return element_wise_calc_reduce<1>(
        [](R &res, const T1 &l, const T2 &r) {
            res += l * r;
        }, R(),
        lhs,
        rhs.extend(lhs.size(D1 - 2)).transpose(0, 1)
    );
}

template<
    typename T1, typename T2, size_t D2, REQUEST_TPL(D2 > 1)
> auto matmul(
    tensor_subslice<T1, 1> lhs, tensor_subslice<T2, D2> rhs
) {
    using R = decltype(std::declval<T1>() * std::declval<T2>());

    return element_wise_calc_reduce<1>(
        [](R &res, const T1 &l, const T2 &r) {
            res += l * r;
        }, R(),
        lhs.extend(rhs.size(D2 - 1)).transpose(0, 1),
        rhs.transpose(D2 - 2, D2 - 1)
    );
}

template<
    typename T1, typename T2
> auto matmul(
    tensor_subslice<T1, 1> lhs, tensor_subslice<T2, 1> rhs
) {
    using R = decltype(std::declval<T1>() * std::declval<T2>());

    return element_wise_calc_reduce<1>(
        [](R &res, const T1 &l, const T2 &r) {
            res += l * r;
        }, R(),
        lhs,
        rhs
    );
}

}  // namespace tensors
