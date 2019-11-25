#pragma once

#include "element_wise_operations.h"


namespace tensors {

template<
    typename T1, size_t D1, typename T2, size_t D2
> tensor_subslice<T1, D1> operator+=(
    tensor_subslice<T1, D1> lhs, tensor_subslice<T2, D2> rhs
) {
    element_wise_apply([](T1 &v1, const T2 &v2) {
        v1 += v2;
    }, lhs, rhs);
    return lhs;
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> tensor_subslice<T1, D1> operator-=(
    tensor_subslice<T1, D1> lhs, tensor_subslice<T2, D2> rhs
) {
    element_wise_apply([](T1 &v1, const T2 &v2) {
        v1 -= v2;
    }, lhs, rhs);
    return lhs;
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> tensor_subslice<T1, D1> operator*=(
    tensor_subslice<T1, D1> lhs, tensor_subslice<T2, D2> rhs
) {
    element_wise_apply([](T1 &v1, const T2 &v2) {
        v1 *= v2;
    }, lhs, rhs);
    return lhs;
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> tensor_subslice<T1, D1> operator/=(
    tensor_subslice<T1, D1> lhs, tensor_subslice<T2, D2> rhs
) {
    element_wise_apply([](T1 &v1, const T2 &v2) {
        v1 /= v2;
    }, lhs, rhs);
    return lhs;
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> tensor_subslice<T1, D1> operator%=(
    tensor_subslice<T1, D1> lhs, tensor_subslice<T2, D2> rhs
) {
    element_wise_apply([](T1 &v1, const T2 &v2) {
        v1 %= v2;
    }, lhs, rhs);
    return lhs;
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> tensor_subslice<T1, D1> operator&=(
    tensor_subslice<T1, D1> lhs, tensor_subslice<T2, D2> rhs
) {
    element_wise_apply([](T1 &v1, const T2 &v2) {
        v1 &= v2;
    }, lhs, rhs);
    return lhs;
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> tensor_subslice<T1, D1> operator|=(
    tensor_subslice<T1, D1> lhs, tensor_subslice<T2, D2> rhs
) {
    element_wise_apply([](T1 &v1, const T2 &v2) {
        v1 |= v2;
    }, lhs, rhs);
    return lhs;
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> tensor_subslice<T1, D1> operator^=(
    tensor_subslice<T1, D1> lhs, tensor_subslice<T2, D2> rhs
) {
    element_wise_apply([](T1 &v1, const T2 &v2) {
        v1 ^= v2;
    }, lhs, rhs);
    return lhs;
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> tensor_subslice<T1, D1> operator<<=(
    tensor_subslice<T1, D1> lhs, tensor_subslice<T2, D2> rhs
) {
    element_wise_apply([](T1 &v1, const T2 &v2) {
        v1 <<= v2;
    }, lhs, rhs);
    return lhs;
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> tensor_subslice<T1, D1> operator>>=(
    tensor_subslice<T1, D1> lhs, tensor_subslice<T2, D2> rhs
) {
    element_wise_apply([](T1 &v1, const T2 &v2) {
        v1 >>= v2;
    }, lhs, rhs);
    return lhs;
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator+(
    tensor_subslice<T1, D1> lhs, tensor_subslice<T2, D2> rhs
) {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 + v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator-(
    tensor_subslice<T1, D1> lhs, tensor_subslice<T2, D2> rhs
) {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 - v2;
    }, lhs, rhs);
}

template<typename T, size_t D> auto operator+(
    tensor_subslice<T, D> &val
) {
    return element_wise_calc([](const T &v) {
        return +v;
    }, val);
}

template<typename T, size_t D> auto operator-(
    tensor_subslice<T, D> &val
) {
    return element_wise_calc([](const T &v) {
        return -v;
    }, val);
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator*(
    tensor_subslice<T1, D1> lhs, tensor_subslice<T2, D2> rhs
) {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 * v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator/(
    tensor_subslice<T1, D1> lhs, tensor_subslice<T2, D2> rhs
) {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 / v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator%(
    tensor_subslice<T1, D1> lhs, tensor_subslice<T2, D2> rhs
) {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 % v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator==(
    tensor_subslice<T1, D1> lhs, tensor_subslice<T2, D2> rhs
) {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 == v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator!=(
    tensor_subslice<T1, D1> lhs, tensor_subslice<T2, D2> rhs
) {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 != v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator>(
    tensor_subslice<T1, D1> lhs, tensor_subslice<T2, D2> rhs
) {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 > v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator<(
    tensor_subslice<T1, D1> lhs, tensor_subslice<T2, D2> rhs
) {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 < v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator>=(
    tensor_subslice<T1, D1> lhs, tensor_subslice<T2, D2> rhs
) {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 >= v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator<=(
    tensor_subslice<T1, D1> lhs, tensor_subslice<T2, D2> rhs
) {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 <= v2;
    }, lhs, rhs);
}

template<typename T, size_t D> auto operator!(
    tensor_subslice<T, D> &val
) {
    return element_wise_calc([](const T &v) {
        return !v;
    }, val);
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator&&(
    tensor_subslice<T1, D1> lhs, tensor_subslice<T2, D2> rhs
) {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 && v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator||(
    tensor_subslice<T1, D1> lhs, tensor_subslice<T2, D2> rhs
) {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 || v2;
    }, lhs, rhs);
}

template<typename T, size_t D> auto operator~(
    tensor_subslice<T, D> &val
) {
    return element_wise_calc([](const T &v) {
        return ~v;
    }, val);
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator&(
    tensor_subslice<T1, D1> lhs, tensor_subslice<T2, D2> rhs
) {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 & v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator|(
    tensor_subslice<T1, D1> lhs, tensor_subslice<T2, D2> rhs
) {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 | v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator^(
    tensor_subslice<T1, D1> lhs, tensor_subslice<T2, D2> rhs
) {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 ^ v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator<<(
    tensor_subslice<T1, D1> lhs, tensor_subslice<T2, D2> rhs
) {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 << v2;
    }, lhs, rhs);
}

template<
    typename T1, size_t D1, typename T2, size_t D2
> auto operator>>(
    tensor_subslice<T1, D1> lhs, tensor_subslice<T2, D2> rhs
) {
    return element_wise_calc([](const T1 &v1, const T1 &v2) {
        return v1 >> v2;
    }, lhs, rhs);
}

}  // namespace tensors
