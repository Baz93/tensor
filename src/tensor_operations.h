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

}  // namespace tensors
